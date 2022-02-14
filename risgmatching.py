# cython: language_level=3

from copy import deepcopy
import cv2,math
import numpy as np
import torch
from torch import nn

torch.set_grad_enabled(False)

def frame2tensor(frame, device):
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

####  SUPERGLUE  #### #############################################################################


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        path = "./db/sg.dat"
        self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #    self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        with torch.no_grad():
            desc0, desc1 = data['descriptors0'], data['descriptors1']
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']

            if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
                shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
                return {
                    'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                    'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                    'matching_scores0': kpts0.new_zeros(shape0),
                    'matching_scores1': kpts1.new_zeros(shape1),
                }

            # Keypoint normalization.
            s0 = torch.Size([1, 1, data['imageh0'], data['imagew0']])
            s1 = torch.Size([1, 1, data['imageh1'], data['imagew1']])

            kpts0 = normalize_keypoints(kpts0, s0)
            kpts1 = normalize_keypoints(kpts1, s1)

            # kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            # kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

            # Keypoint MLP encoder.
            desc0 = desc0 + self.kenc(kpts0, data['scores0'])
            desc1 = desc1 + self.kenc(kpts1, data['scores1'])
            # std = time.perf_counter()
            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1)
            # print('GNN:%6.2f' % (time.perf_counter() - std))

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / self.config['descriptor_dim'] ** .5

            # Run the optimal transport.
            scores = log_optimal_transport(
                scores, self.bin_score,
                iters=self.config['sinkhorn_iterations'])

            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            return {
                'matches0': indices0,  # use -1 for invalid match
                'matches1': indices1,  # use -1 for invalid match
                'matching_scores0': mscores0,
                'matching_scores1': mscores1,
            }


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=False, **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


####SuperPoint################################################################
# import joblib
# gPCA = joblib.load('./db/pca64xin.m')

class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = './db/sp.dat'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        #print('*  载入AI模型成功...')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        with torch.no_grad():
            x = self.relu(self.conv1a(data['image']))
            x = self.relu(self.conv1b(x))
            x = self.pool(x)
            x = self.relu(self.conv2a(x))
            x = self.relu(self.conv2b(x))
            x = self.pool(x)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool(x)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))

            # Compute the dense keypoint scores
            cPa = self.relu(self.convPa(x))
            scores = self.convPb(cPa)
            scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
            b, _, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
            scores = simple_nms(scores, self.config['nms_radius'])

            # Extract keypoints
            keypoints = [
                torch.nonzero(s > self.config['keypoint_threshold'])
                for s in scores]
            scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

            # Discard keypoints near the image borders
            keypoints, scores = list(zip(*[
                remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
                for k, s in zip(keypoints, scores)]))

            # Keep the k keypoints with highest score
            if self.config['max_keypoints'] >= 0:
                keypoints, scores = list(zip(*[
                    top_k_keypoints(k, s, self.config['max_keypoints'])
                    for k, s in zip(keypoints, scores)]))

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]

            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            descriptors = self.convDb(cDa)
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

            # Extract descriptors
            descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                           for k, d in zip(keypoints, descriptors)]
            scores = list(scores)
            # print(descriptors[0].shape)
            return {
                'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors,
            }


####SG################################################################
class SuperPointFeatureExtract(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, config={}, sp=''):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        with torch.no_grad():
            pred = {}
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
            return pred


class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": './db/sp.dat',
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'
        self.superpoint = SuperPoint(self.config).eval().to(self.device)

    def set_config(self,config):
        self.superpoint.config = {**self.superpoint.config, **config}

    def __call__(self, image_tensor, img_size, outflag='tensor'):

        pred = self.superpoint({'image': image_tensor})

        if outflag == 'tensor':
            ret_dict = {
                "imageh": img_size[0],
                "imagew": img_size[1],
                "keypoints": pred["keypoints"],
                "scores": pred["scores"],
                "descriptors": pred["descriptors"]
            }
        else:
            ret_dict = {
                "image_size": np.array([img_size[0], img_size[1]]),
                "torch": pred,
                "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
                "scores": pred["scores"][0].cpu().detach().numpy(),
                "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
            }
        return ret_dict


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, config={}):
        super().__init__()
        # self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config)
        self.config = config

    def forward(self, data):
        with torch.no_grad():
            pred = {}
            data = {**data, **pred}
            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])
            # Perform the matching
            pred = {**pred, **self.superglue(data)}
            return pred

class SuperGlueMatcher(object):
    default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.5,
        "path": './db/sg.dat',
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'
        print('*  --Device for matching：' + self.device)
        # self.superglue = SuperGlue(self.config).to(self.device)
        self.superglue = Matching(self.config).eval().to(self.device)

    def __call__(self, kptdescs0, kptdescs1):
        # setup data for superglue
        kptdescs00 = {**{k + '0': v for k, v in kptdescs0.items()}}
        kptdescs11 = {**{k + '1': v for k, v in kptdescs1.items()}}
        data = {**kptdescs00, **kptdescs11}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        pred = self.superglue(data)

        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}

        kpts0, kpts1 = kptdescs0['keypoints'][0].cpu().numpy(), kptdescs1['keypoints'][0].cpu().numpy()
        matches0 = pred['matches0']
        conf = pred['matching_scores0']
        valid0 = matches0 > -1
        index0, index1 = np.where(valid0)[0], matches0[valid0]
        mkpts0 = kpts0[valid0]
        mconf = conf[valid0]
        mkpts1 = kpts1[index1]

        pred.clear()
        torch.cuda.empty_cache()
        return mkpts0, mkpts1, mconf, index0, index1

def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x, np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def rotate_image_bound_with_M(image, angle):
    if angle == 0:
        return image,np.array([[1,0,0],[0,1,0]])

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int(0.5 + (h * sin) + (w * cos))
    nH = int(0.5 + (h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_AREA, borderValue=(255, 255, 255)),M

def rotate_image_bound(image, angle):
    img, M = rotate_image_bound_with_M(image, angle)
    return img

def calRotateAngleFromMatch(mkpts0, mkpts1):
    mat = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    return calRotateAngleFromMatrix(mat)

def calTransformationFromMatch(mkpts0, mkpts1):
    mat = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    return mat[0][:,2].T

def calRotateAngleFromMatrix(mat):
    rmat = mat[0][:2, :2]
    det = np.linalg.det(rmat)
    rmat_normal = rmat / (det ** 0.5)
    angle = math.asin(rmat_normal[1, 0]) * 180 / math.pi
    return angle

class RISGMatcher(object):
    def __init__(self, config={}):
        self.sp_detector = SuperPointDetector(config["detector"])
        self.sg_matcher = SuperGlueMatcher(config["matcher"])
        self.device = 'cuda' if torch.cuda.is_available() and config["detector"]["cuda"] else 'cpu'

    def match(self,img0, img1, nrotate=1):
        if (img0 is None) or (img1 is None):
            print('Error: Image file is not found!')
            return None

        #img0shape = img0.shape
        img1shape = img1.shape

        img1_tensor = frame2tensor(img1, self.device)
        fdata1 = self.sp_detector(img1_tensor, img1shape)

        # kpts0_all = np.empty((0,2))
        mkpts0_all = np.empty((0, 2))
        mkpts1_all = np.empty((0, 2))
        conf0_all = np.empty((0))

        step = 360.0 / nrotate

        for rot in range(nrotate):
            img_r, M = rotate_image_bound_with_M(img0, rot * step)
            M = np.row_stack((M, np.array([0, 0, 1])))
            M_inv = np.mat(np.linalg.inv(M))
            img0_tensor = frame2tensor(img_r, self.device)
            fdata0 =  self.sp_detector(img0_tensor, (img0_tensor.shape[2], img0_tensor.shape[3]))
            mkpts0, mkpts1, conf, idx0, idx1 =  self.sg_matcher(fdata0, fdata1)

            # unproject points
            hmkpts0 = add_ones(mkpts0)
            rhmkpts0 = (M_inv * hmkpts0.T).A.T[:, 0:2]

            mkpts0_all = np.vstack((mkpts0_all, rhmkpts0))
            mkpts1_all = np.vstack((mkpts1_all, mkpts1))
            conf0_all = np.hstack((conf0_all, conf))

        #get main direction
        maindirection = 0
        if len(conf0_all)>6:
            maindirection = calRotateAngleFromMatch(mkpts0_all, mkpts1_all)
        else:
            print('Waning: Matching points are not enough.')
        return mkpts0_all, mkpts1_all, conf0_all,maindirection

