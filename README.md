# Yolov9 + StrongSORT Traffic_tracker

## Before you run the tracker

安装依赖库，确保python>=3.8 （尽量用3.8版本 不要高版本）

如果没有cuda先去安装cuda

https://blog.csdn.net/chen565884393/article/details/127905428

安装**anaconda** **pytorch**

conda和torch安装: https://blog.csdn.net/MCYZSF/article/details/116525159


最后 在conda终端激活环境后 cd到本目录 
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple




## Tracking sources

```bash

$ python detect_v9.py --source 0  # webcam
                           img.jpg  # image
                           vid.mp4  # video
                           path/  # directory
                           path/*.jpg  # glob
                           'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```


## Select object detection and ReID model

### Yolo

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov9/7 family model for automatic download

```bash


$ python python detect_v9.py --source 0 --yolo-weights yolov7.pt --img 640
                                            yolov9-c.pt --img 640
                                           
```

### StrongSORT

The above applies to StrongSORT models as well. Choose a ReID model based on your needs from this ReID [model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)

```bash


$ python python detect_v9.py --source 0 --strong-sort-weights osnet_x0_25_market1501.pt
                                                   
```


## 选择跟踪类别


```bash
python python detect_v9.py --source 0 --yolo-weights yolov7.pt --classes 0  # tracks persons only
```








## 追踪
```bash
$ python detect_v9.py --source .....
```

目前可以通过yolov9进行检测，速度会比yolov7要更高，且检测精度更好,目前模型用的官方**yolov9-c.pt** 模型，数据集为coco


## 2024/3/30
**车牌检测线程**

**图片显示线程**

**斑马线检测线程**


## 前端要注意的点
**车辆截图保存**
目前所有图片都保存在后端，因此前端要考虑如何从后端读入相关保存的图片，或者提供一个网络上传接口，实时将后端图片进行上传至前端

**实时车流统计**
目前车流统计是通过字典格式保存 格式{类别：数量}

前端需要实时读入 并实时绘制相应统计图

**实时图片**
前端可直接读入**self.image_display_queue** 此队列里的图片进行显示


## 2024/4/3 
update:

更新车牌检测模型位yolov8

更改线程处理方式 修复并发线程无法及时检测车牌的问题

## 2024/4/4

在运行代码中 使用**self.is_running** True 为运行 False 为不在运行  指示运行状态 方便调整

同时新增**start_detection**函数 通过线程启动检测，可以在主线程中添加更多的处理细节 比如获取车辆信息表等
使用方法：
```python
    logger.info(f"Initializing VideoTracker with options: {opt}")
    video_tracker = VideoTracker(opt)
    logger.info('Init successfully')
    video_tracker.change_video("D:/traffic_detection/video_20s.mp4")
    video_tracker.start_detection()  # 开始检测
```




## 2024/4/8
给出多个返回值函数
```python
   def get_car_status(self):
        """
        返回最新的车辆状态列表
        ['id: 1, licence: , speed: , illegal: False, illegal: ',......]
        """

        return self.car_status

    def get_class_counts(self):
        """
        返回最新的类别统计
        {car: 5, truck: 7}
        """
        return self.class_counts

    def get_vehicle_count(self):
        # 返回当前的车流量计数 int
        return self.vehicle_count

    def get_illegal_car(self):
        """
        返回 违规车辆行为
        ['id: 5, licence: , illegal_behavior: 超速, 变道', 'id: 3, licence: , illegal_behavior: 超速']
        """
        return self.illegal_car_status
```
使用示例：
```python
    video_tracker = VideoTracker(opt)
    video_tracker.start_detection()  # 开始检测
    try:
        while video_tracker.is_running:
            # 每隔一定时间获取并打印实时数据
            car_status = video_tracker.get_car_status()
            class_counts = video_tracker.get_class_counts()
            vehicle_count = video_tracker.get_vehicle_count()
            illegal_car = video_tracker.get_illegal_car()
```
增加更换视频源函数
```python
    def change_video(self,vid_path):
        self.source=vid_path
```
使用示例：
```python 
    video_tracker = VideoTracker(opt)
    video_tracker.change_video("D:/traffic_detection/video_20s.mp4")
    video_tracker.start_detection()  # 开始检测
```
在违规车辆增加其图片crop type:image 
```python
    if SpeedOverFlag and self.names[int(cls)] in ['car', 'bus', 'truck']:
        car_status_dict['illegal'] = True
        car_status_dict['illegal_behavior'].append('超速')
        if id is not None:
            crop = save_one_box(torch.Tensor(bboxes), imc, BGR=True, save=False)
            path = Path(self.save_dir / 'speedover' / str(int(id)) / f'{speed}.jpg')
            path.parent.mkdir(parents=True, exist_ok=True)  # make directory
            Image.fromarray(crop[..., ::-1]).save(str(path), quality=95,
                                                  subsampling=0)  # save RGB
```


## 车牌识别
通过**Yolov8**定位车牌目标位置 **LPRNET**对车牌字符进行识别

红绿灯检测：只检测一次 获得坐标后 持续对图像进行颜色检测
```python
            if self.names[int(cls)] == 'traffic light':  # 假设交通灯的类别标签是'traffic light'
                # 获取交通灯图像区域
                tl_x1, tl_y1, tl_x2, tl_y2 = map(int, [781, 35, 807, 106])
                tl_img = im0s[tl_y1:tl_y2, tl_x1:tl_x2]

                # 检测交通灯颜色
                if tl_img.size > 0:  # 确保图像区域有效
                    traffic_light_color = self.TLL_DET(tl_img)
                    print(f"Detected traffic light color: {traffic_light_color}")
                    # 这里你可以根据颜色做进一步的处理
                else:
                    logger.info("Invalid traffic light image region.")
```
斑马线检测：只检测一次 获得坐标后 对图像绘制坐标
```python
                # 斑马线检测
                if not self.zebra_detected:
                    ret, zoo_location = ZebraDetection(im0)
                    if ret:
                        self.zebra_detected = True
                        box = [zoo_location[0][0], zoo_location[0][1], zoo_location[1][0], zoo_location[1][1]]
                        annotator.box_label(box, 'zoo_crossing', (0, 255, 0))
                    else:
                        self.zebra_detected = True
                        box = [399,494,1456,640]
                        annotator.box_label(box,'zoo_crossing',(0,255,0))
```

## 测速更新
利用消失点估算外参矩阵 用matlab得到内参矩阵 最后得到世界坐标

