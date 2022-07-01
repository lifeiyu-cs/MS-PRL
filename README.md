# MS-PRL
Rethinking PRL: A Multi-Scale Progressively Residual Learning Network for Inverse Halftoning.

The code will release soon.

## Abstract

## Network architecture

## Contents
1. [Environment](#env)
2. [Demo](#demo)
3. [Dataset](#data)
4. [Train](#train)
5. [Test and Valid](#test)
6. [Model](#model)

## Environment <a name="env"></a>
```shell
python=3.8 numpy=1.21.2 opencv-python=4.5.5.64
pillow=8.4.0 numba=0.55.1 scikit-image=0.18.3
pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3
```

## Demo <a name="demo"></a>
```shell
python demo.py
```

## Train <a name="train"></a>
To train MS-PRL , run the command below:
```shell
python main.py --mode train --model_name=MS-PRL
```
if you want to train other model, pleace change --model_name= your model name. The model weights will be saved in checkpoint/model_name/model_name_iterations.pth folder.


## Test and Valid <a name="test"></a>
run test mode, images will be saved in resutls/model_name/test_name/ and the log will be saved in logs/model_name/test/test_name/log.txt
run valid mode, just the log will be saved in logs/model_name/test/test_name/log.txt

To test MS-PRL , run the command below:
```shell
python main.py --mode test --model_name=MS-PRL
```
To valid MS-PRL , run the command below:
```shell
python main.py --mode valid --model_name=MS-PRL
```
Please pay attention to the **dataset path**, refer to the details of the dataset (#data).

## Dataset <a name="data"></a>
Download VOC2012, Kodak25, Place365 dataset and 5 SR benckmark dataset. You can also down our 
The data folder should be like the format below:
```
dataset
├─ train
│ ├─ data     % 13841 halftone images
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ target   % 13841 gray images
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ valid
│ ├─ data     % 3000 halftone images
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ target   % 3000 gray images
│ │ ├─ xxxx.png
│ │ ├─ ......
|
├─ test
│ ├─ Class
| │ ├─ data     % halftone images
| │ │ ├─ xxxx.png
│ | │ ├─ ......
│ │
│ | ├─ target   % gray image
│ │ | ├─ xxxx.png
│ │ | ├─ ......
|
│ ├─ Kodak
| | ├─ ......

```

## Model <a name="model"></a>
We provide our all pre-trained models.
- MS-PRL, PRL-dt and other model: 
