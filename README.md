# RISG: A rotation invariant SuperGlue algorithm.


## Overview
A rotation invariant SuperGlue matching algorithm for cross modality image
[also see here, https://gitee.com/ssacn/RISG-image-matching](https://gitee.com/ssacn/RISG-image-matching)

## Some results： 

- 多时相谷歌地球影像，Optical-optical
![Optical-optical images](https://s3.bmp.ovh/imgs/2022/02/51f4addb4e8c4bb6.gif)

- 近红外与光学图像 near-infrared - optical images
![ir-optical images](https://s3.bmp.ovh/imgs/2022/02/9ffd48d859fefdaa.gif)

- SAR和光学图像，sar-optical
![sar-optical images](https://s3.bmp.ovh/imgs/2022/02/732ef2bbb5f9c47d.gif)

- 光学图像和夜光图像，optical- night light
![optical- night light images](https://s3.bmp.ovh/imgs/2022/02/07de5b0bdde92881.gif)

- 地图与光学图像，map - optical
![map - optical images](https://s3.bmp.ovh/imgs/2022/02/dc277d711915ae80.gif)

- 光学图像与激光雷达深度图，optical -lidar depth
![optical -lidar depth images](https://s3.bmp.ovh/imgs/2022/02/578b827c2d0bcd1c.gif)

## Getting start:

Python 3.7+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:
### Dependencies

- PyTorch-GPU 1.10.0+
- OpenCV
- SciPy
- Matplotlib
- pyymal
- pickle

### Dataset
We collected a set of test data, including images from space-borne SAR and visible light sensors, drone thermal infrared sensors, and Google Earth images. You may find them in the directory  "test" in this repository.

### Usage

#### just for test
`risgmatching.py` contains the majority of the code. Run `test_risg.py` for testing:
```bash
python3 test_risg.py
```

#### Using RISG in your code

```python
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        risg = RISGMatcher(config)

        img_filename0 = 'test/01/pair1.jpg'
        img_filename1 = 'test/01/pair2.jpg'

        img0 = cv2.imread(img_filename0)
        img1 = cv2.imread(img_filename1)
        # rotate is number of directions
        mkpts0, mkpts1, conf, main_dir = risg.match(img0,img1,nrotate = 5)

```

## RISG source code is coming soon...
