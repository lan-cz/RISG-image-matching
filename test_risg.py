import yaml,cv2
import time,pickle
from risgmatching import RISGMatcher,rotate_image_bound
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

###################################################################################################
def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, conf, text, path=None,
                            show_keypoints=False, margin=10, show_line=True,
                            opencv_display=False, opencv_title='',
                            small_text=[],resize_win = None):
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]
    H, W = max(H0, H1), W0 + W1 + margin

    color = cm.jet(conf)
    if image0.ndim == 3:
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0 + margin:, :] = image1
    else:
        out = 255 * np.ones((H, W), np.uint8)
        out[:H0, :W0] = image0
        out[:H1, W0 + margin:] = image1
        out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 255)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        if show_line:
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                     color=c, thickness=2, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 5, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 5, c, -1,
                   lineType=cv2.LINE_AA)
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)
    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)
    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
    if resize_win is not None:
        out = cv2.resize(out, resize_win)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.resizeWindow(opencv_title, resize_win[0], resize_win[1])
        cv2.waitKey(1)
    return out
###################################################################################################

def initial(ax):
    #ax.axis("equal")  # 设置图像显示的时候XY轴比例
    ax.set_xlabel('Rotate angle(Deg)')
    ax.set_ylabel('Number of matching points')
    #ax.set_title('RISG Matching')
    ax.set_xticks(range(-180, 190, 45))
    ax.set_yticks(range(0, 1000, 50))

    return ax


if __name__ == '__main__':

    config_filename = './config.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
        risg = RISGMatcher(config)

        testid = 1
        img_filename0 = 'test/%02d/pair1.jpg'%testid
        img_filename1 = 'test/%02d/pair2.jpg'%testid

        img0 = cv2.imread(img_filename0)
        img1 = cv2.imread(img_filename1)

        if (img0 is None) or (img1 is None):
            print('Error: Image file not found.')
            exit()
        start_time = time.perf_counter()

        rotate_angel = 0
        step = 5
        score = []
        ##########################################################################
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax = initial(ax)
        plt.ion()  # interactive mode on

        obsX = []
        obsY = []
        obsYsg = []
        #############################################################################
        for rotate_angel in range(-180,180,5):
            print('\n')
            img1r = rotate_image_bound(img1,rotate_angel)
            start_time = time.perf_counter()
            mkpts0, mkpts1, conf, main_dir = risg.match(img0,img1r,nrotate = 5)
            print('RISG matching time: %6.3fs. Matching points num: %d, main diretion: %6.2f'%(time.perf_counter()-start_time,len(conf),main_dir))
            start_time = time.perf_counter()
            mkpts0sg, mkpts1sg, confsg, _ = risg.match(img0,img1r,nrotate = 1)
            print('SuperGlue matching time: %6.3fs. Matching points num: %d'%(time.perf_counter()-start_time,len(confsg)))

            #ransac
            if mkpts1.shape[0]>10:
                Affine, mask = cv2.estimateAffine2D(mkpts1, mkpts0, ransacReprojThreshold = 3)
                inlineNum = np.sum(mask)
                mm = np.where(mask > 0)
                mmkpts0 = mkpts0[mm[0], :]
                mmkpts1 = mkpts1[mm[0], :]
                ransac_mconf = conf[mm[0]]
                #score.append(inlineNum)
            else:
                mmkpts0,mmkpts1 = mkpts0,mkpts1
                inlineNum = mkpts1.shape[0]

            if mkpts1sg.shape[0]>10:
                Affine, mask = cv2.estimateAffine2D(mkpts1sg, mkpts0sg, ransacReprojThreshold = 3)
                inlineNumsg = np.sum(mask)
                mm = np.where(mask > 0)
                mmkpts0sg = mkpts0sg[mm[0], :]
                mmkpts1sg = mkpts1sg[mm[0], :]
            else:
                mmkpts0sg,mmkpts1sg = mkpts0sg,mkpts1sg
                inlineNumsg = mkpts1sg.shape[0]
            text = [
                'RISG:',
                'Angle{} '.format(rotate_angel),
                'matches: {}/{}'.format(mmkpts1.shape[0],mkpts1.shape[0]),
                'Main dir:{:.2f} '.format(main_dir)
            ]
            make_matching_plot_fast(img0, img1r,  mkpts0, mkpts1, mmkpts0, mmkpts1, conf, text,
                                    opencv_title='RISG',show_keypoints=True, opencv_display=True, show_line=True,resize_win=(600,300))
            text = [
                'SuperGlue:',
                'Angle {} '.format(rotate_angel),
                'matches: {}/{}'.format(mmkpts1sg.shape[0],mkpts1sg.shape[0])
            ]

            make_matching_plot_fast(img0, img1r,  mkpts0sg, mkpts1sg, mmkpts0sg, mmkpts1sg, confsg, text,
                                    opencv_title='SuperGlue', show_keypoints=True, opencv_display=True, show_line=True,resize_win=(600,300))
            inlineNum = mmkpts1.shape[0]
            inlineNumsg = mmkpts1sg.shape[0]
            obsX.append(rotate_angel)
            obsY.append(inlineNum)
            obsYsg.append(inlineNumsg)


            plt.cla()
            ax = initial(ax)
            ax.plot(obsX, obsY,'-r')
            ax.plot(obsX, obsYsg,'--g') 

            plt.legend(labels=['RISG','SuperGlue'], loc='best')
            plt.pause(0.001)

            rotate_angel = rotate_angel + step

        print('done!')
        obsY = np.array(obsY)
        mean = np.mean(obsY)
        plt.savefig('result/%d-%d' % (testid,mean) + '.png')
        print('average num: %d'%(mean))

        cv2.waitKey(0)


