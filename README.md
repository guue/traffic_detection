# Yolov7 + StrongSORT with OSNet





<div align="center">
<p>
<img src="MOT16_eval/track_pedestrians.gif" width="400"/> <img src="MOT16_eval/track_all.gif" width="400"/> 
</p>
<br>  
<a href="https://colab.research.google.com/drive/101f0PNBPx3245Hu710QAf2LXpf3E2uIk?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 



## Introduction

This repository contains a highly configurable two-stage-tracker that adjusts to different deployment scenarios. The detections generated by [YOLOv7](https://github.com/WongKinYiu/yolov7), a family of object detection architectures and models pretrained on the COCO dataset, are passed to [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/pdf/2202.13514.pdf) which combines motion and appearance information based on [OSNet](https://github.com/KaiyangZhou/deep-person-reid)[](https://arxiv.org/abs/1905.00953) in order to tracks the objects. It can track any object that your Yolov7 model was trained to detect.

## Before you run the tracker

1. Clone the repository recursively:


If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov7_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats

```bash
$ python track.py --source 0  # webcam
                           img.jpg  # image
                           vid.mp4  # video
                           path/  # directory
                           path/*.jpg  # glob
                           'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```


## Select object detection and ReID model

### Yolov7

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov7 family model for automatic download

```bash


$ python track.py --source 0 --yolo-weights yolov7.pt --img 640
                                            yolov7x.pt --img 640
                                            yolov7-e6e.pt --img 1280
                                            ...
```

### StrongSORT

The above applies to StrongSORT models as well. Choose a ReID model based on your needs from this ReID [model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)

```bash


$ python track.py --source 0 --strong-sort-weights osnet_x0_25_market1501.pt
                                                   osnet_x0_5_market1501.pt
                                                   osnet_x0_75_msmt17.pt
                                                   osnet_x1_0_msmt17.pt
                                                   ...
```


## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the MS COCO classes, add their corresponding index after the classes flag

```bash
python track.py --source 0 --yolo-weights yolov7.pt --classes 16 17  # tracks cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov7 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.


## MOT compliant results

Can be saved to your experiment folder `runs/track/<yolo_model>_<deep_sort_model>/` by 

```bash
python track.py --source ... --save-txt
```

## 测速实现
首先使用提前设定好的车辆真实宽度和检测出来的车辆像素宽度求出真实距离和像素距离的比值为c，再使用每辆车的前后两帧框的中心坐标计算出两帧之间移动的像素距离。利用这个比值和像素距离做映射，就可以求出两帧之间车辆移动的真实距离。然后距离除以两帧之间的时间，就是速度了。本测速算法中将车辆真实移动距离与像素移动距离看成是线性关系，仅在监控相机轴线与车辆移动方向垂直时才能成立，并且检测出来的车辆框在空间上会产生一定形变，使得真实距离和像素距离的映射关系不准确。
```bash

def Estimated_speed(outputs, output, id, fps, width):
    SpeedOver = False
    prev_IDs = []  # 之前的ids
    work_IDs = []  # 有效的ids
    work_locations = output  # 当前帧数据：中心点x坐标、中心点y坐标、目标序号、车辆类别、车辆像素宽度
    work_prev_locations = []  # 上一帧数据，数据格式相同
    for i in range(len(outputs)):
        prev_IDs.append(outputs[i][4])  # 获得前一帧中跟踪到车辆的ID

    for m, n in enumerate(prev_IDs):  # 进行筛选，找到在两帧图像中均被检测到的有效车辆ID，存入work_IDs中
        if id == n:
            work_IDs.append(m)
            work_prev_locations = outputs[m]  # 将当前帧有效检测车辆的信息存入work_locations中



    if len(work_IDs) > 0:
        locations = [0,0]
        prev_locations = [0,0]  # 存放中心点坐标
        bbox_prev = work_prev_locations[0:4]
        bbox = work_locations[0:4]
        p1, p2 = (int(bbox_prev[0]), int(bbox_prev[1])), (int(bbox_prev[2]), int(bbox_prev[3]))
        print(p1)
        prev_locations[0] = (p2[0] - p1[0]) / 2 + p1[0]
        prev_locations[1] = (p2[1] - p1[1]) / 2 + p1[1]
        x1, x2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        locations[0] = (x2[0] - x1[0]) / 2 + x1[0]
        locations[1] = (x2[1] - x1[1]) / 2 + x1[1]
        d=math.sqrt((locations[0] - prev_locations[0]) ** 2 + (locations[1] - prev_locations[1]) ** 2)

        speed = ((d /width ) *2.5*3.6* fps  )  #像素速度

        if speed > 60:
            SpeedOver = True

        print(speed)
        # speed = 11.3
        speed = str(round(speed, 1)) + "km/h"
        return speed,SpeedOver
    return " ",SpeedOver
```
###待实现
此算法无法检测的速度不准确 低于正常速度 还需通过**透视变换** 将测速时的照片与监控设备成垂直状态 测试结束后变回来

