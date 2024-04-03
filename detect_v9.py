import argparse
import shutil

import threading
import queue

import os
import platform
import time
from multiprocessing import Queue, Event

from licence import Licence
from datetime import datetime, timedelta
# limit the number of cpus used by high performance libraries
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov9') not in sys.path:
    sys.path.append(str(ROOT / 'yolov9'))  # add yolov9 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from YOLOv9.models.common import DetectMultiBackend
from YOLOv9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from YOLOv9.utils.general import (logging, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                  colorstr, cv2,
                                  increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
                                  xyxy2xywh)

from YOLOv9.utils.torch_utils import select_device, smart_inference_mode
from YOLOv9.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from calculateDirection import calculateDirection
from Estimated_speed import Estimated_speed

from draw import *
import logging
import matplotlib.pyplot as plt
from zebra_detector.test import detect_zoo
import pynvml

logger = logging.getLogger('detect')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("detect.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)  # 设置文件的log格式
logger.addHandler(handler)


def clear_directory(dir_path):
    """
    Clears all the contents of the given directory.

    Parameters:
    - dir_path: The path of the directory to clear.
    """
    if os.path.exists(dir_path):
        # 删除目录下的所有内容
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


class VideoTracker:
    def __init__(self, opt):

        self.license_output_queue = queue.Queue()
        self.license_input_queue = queue.Queue(7)
        self.car_license_history = {}  # {id: [last_license, second_last_license]

        self.vid_flag = 0
        self.im0_s = None
        self.pred = None
        self.webcam = None

        self.dnn = opt.dnn
        self.device = select_device(opt.device)
        self.half = opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        self.save_dir = opt.save_dir
        self.yolo_weights = opt.yolo_weights

        self.licence = Licence()

        self.carlicence = ''
        # Load YOLO model
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.yolo_model = self.loadmodel(Path(opt.yolo_weights))
        self.names = self.yolo_model.names

        self.stride = self.yolo_model.stride  # Model stride
        self.pt = self.yolo_model.pt

        self.source = str(opt.source)
        self.show_vid = opt.show_vid
        self.imgsz = check_img_size(opt.imgsz, s=self.stride)

        self.batchsize = 1
        self.vid_stride = opt.vid_stride
        self.dataset = self.dateset_Loader()
        self.vid_writer = None

        # Load StrongSORT
        self.strongsort_cfg = get_config()
        self.strongsort_cfg.merge_from_file(opt.config_strongsort)
        self.strongsort_list = []  # 被跟踪对象
        for i in range(self.batchsize):
            self.strongsort_list.append(
                StrongSORT(
                    opt.strong_sort_weights,
                    device=self.device,
                    fp16=self.half,
                    max_dist=self.strongsort_cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=self.strongsort_cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=self.strongsort_cfg.STRONGSORT.MAX_AGE,
                    n_init=self.strongsort_cfg.STRONGSORT.N_INIT,
                    nn_budget=self.strongsort_cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=self.strongsort_cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=self.strongsort_cfg.STRONGSORT.EMA_ALPHA,
                )
            )
            self.strongsort_list[i].model.warmup()

        # 存放预测结果   第一维是id索引
        self.outputs = [None] * self.batchsize
        self.outputs_prev = []

        self.zebra_input_queue = queue.Queue()
        self.zebra_output_queue = queue.Queue()



        self.image_display_queue = queue.Queue(maxsize=10)  # 前端可直接读此队列进行显示



        """
        储存格式应该是 ['id: license: speed: illegal: '] illegal 显示其违法信息  如果涉及违法 则在ui中标红
        在一开始 illegal 默认为 '' 
        如果有违规需要提出预警信息 前端ui相应车辆信息行列标红
        warning()
            ui 车辆信息标红
            并对车辆信息（车牌号 违法信息）保存
        最后将所有信息存储在car_status里，前端需要读取里面的信息，并根据相应的位置进行显示 每一帧图片之后这个信息都会更新 
        """
        self.car_licence = {}  # {id:{license:licence,conf:conf},....}
        self.car_status = []  # 存放每辆车的状态

        # Other parameters
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.save_txt = opt.save_txt
        self.save_vid = opt.save_vid
        self.hide_labels = opt.hide_labels


        self.augment = opt.augment
        self.max_det = opt.max_det
        self.class_counts = {}
        self.vehicle_count = 0
        self.vehicle_counts_over_time = {}  # 时间戳: 车辆总数


        self.last_time_stamp = 0

        # Placeholder for FPS calculation
        self.fps = 0

    def license_plate_detection(self, input_queue, output_queue):
        while True:
            item = input_queue.get()
            if item is None:  # None用来指示线程退出
                break
            id, image = item


            # 检查车辆是否已经连续两次检测到相同的车牌，如果是，则跳过检测
            if id in self.car_license_history and self.car_license_history[id][0] == self.car_license_history[id][1]:
                output_queue.put((id, None))  # 使用None表示跳过检测
                continue
            detection_result = self.licence.detectLicence(image)
            try:
                label,conf = detection_result
            except Exception as e:
                logger.error(e)
            # 更新车牌检测历史
            if label != '':
                if id not in self.car_license_history:
                    self.car_license_history[id] = [label, None]
                else:
                    self.car_license_history[id] = [label, self.car_license_history[id][0]]
            output_queue.put((id, detection_result))
    def zebra_detection_thread(self, input_queue, output_queue):
        while True:
            item = input_queue.get()  # 从队列中获取图像
            if item is None:  # None表示接收到线程结束信号
                output_queue.put(None)  # 确保主线程也能接收到结束信号
                break
            # 进行斑马线检测
            image = item
            detection_result = detect_zoo(image)
            output_queue.put(detection_result)  # 将检测结果放入输出队列

    def image_detect(self, image):
        # while True:
        #     item = input_queue.get()  # 从队列中获取图像
        #     if item is None:  # None表示接收到线程结束信号
        #         output_queue.put(None)  # 确保主线程也能接收到结束信号
        #         break

        # Inference
        self.pred = self.yolo_model(image, augment=self.augment)
        self.pred = self.pred[0][1]
        # Apply NMS
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes,
                                        self.agnostic_nms, max_det=self.max_det)

    def plot_counts(self, count, save_path, title, xlabel):
        times = list(count.keys())
        counts = list(count.values())

        plt.figure(figsize=(10, 6))
        plt.plot(times, counts, marker='o', linestyle='-', color='b')
        plt.title(title + ' Count')
        plt.xlabel(xlabel)
        plt.xticks(rotation=270)

        plt.ylabel(title + ' Count')
        plt.grid(True)

        # 保存图表到指定路径
        plt.savefig(save_path)
        plt.close()  # 关闭图表窗口，释放资源

    def update_or_add(self, car_status_dict):
        """
        car_status_dict即将要添加的车辆状态信息
        """
        # 准备更新或添加的车辆状态信息字符串，移除字典表示的开头和结尾的 '{' 和 '}'
        car_status_str = ", ".join([f"{k}: {v}" for k, v in car_status_dict.items()])
        updated = False
        for i, status_str in enumerate(self.car_status):
            # 从当前列表中的车辆状态字符串中提取车牌信息
            licence_existing = [pair.split(": ")[1] for pair in status_str.split(", ") if "licence" in pair][0]

            # 检查是否满足更新条件：车牌号匹配且不为空
            if licence_existing == car_status_dict['licence'] and licence_existing != '' and car_status_dict[
                'licence'] != '':
                # 更新现有条目
                self.car_status[i] = car_status_str
                updated = True
                break
        if not updated:
            # 如果未找到匹配的车牌号来更新，则添加新的车辆状态信息
            self.car_status.append(car_status_str)

    def loadmodel(self, weights):
        model = DetectMultiBackend(Path(weights), device=self.device, dnn=self.dnn,  fp16=self.half)
        return model

    def image_display(self):
        while  not self.image_display_queue.empty():
            # 从队列中获取图像
            image = self.image_display_queue.get()
            if image is None:
                continue  # None用作占位符，不显示
            # 显示图像
            cv2.imshow('Detection', image)
            if cv2.waitKey(1) == ord('q'):  # 按q退出
                self.display_thread_running = False
                break

    def dateset_Loader(self):

        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        screenshot = self.source.lower().startswith('screen')
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Dataloader
        if self.webcam:
            self.show_vid = check_imshow(warn=True)
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
                                  vid_stride=1)
            self.batchsize = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
                                 vid_stride=1)

        vid_path, vid_writer, txt_path = [None] * self.batchsize, [None] * self.batchsize, [
            None] * self.batchsize  # 多视频源才需要用列表初始化每个视频源的路径
        return dataset

    def preprocess(self, im):
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    @smart_inference_mode()
    def run(self):

        video_start_time = datetime.now()

        # 上次记录时间
        last_record_time = video_start_time
        # 记录间隔，这里设置为2秒
        record_interval = timedelta(seconds=15)

        license_thread = threading.Thread(
            target=self.license_plate_detection,
            args=(self.license_input_queue, self.license_output_queue))

        # zebra_thread = threading.Thread(target=self.zebra_detection_thread,
        #                                 args=(self.zebra_input_queue, self.zebra_output_queue))

        license_thread.start()

        # zebra_thread.start()

        gpu_memory = {}

        self.yolo_model.warmup(imgsz=(1 if self.pt or self.yolo_model.triton else self.bs, 3, *self.imgsz))
        frame_count = 0
        global cardiection, CarLicence, label

        curr_frames, prev_frames = [None] * self.batchsize, [None] * self.batchsize


        for path, im, im0s, vid_cap, s in self.dataset:
            self.fps = vid_cap.get(cv2.CAP_PROP_FPS)
            current_time = datetime.now() # 当前帧开始检测的时间

            time_elapsed = current_time - last_record_time


            # s = ''

            im = self.preprocess(im)

            # 斑马线检测
            # self.zebra_input_queue.put(im)
            # if not self.zebra_output_queue.empty():
            #     zebra_detection_result = self.zebra_output_queue.get()
            #     # 处理斑马线检测结果
            #     zoo_locations, scale_h, scale_w = zebra_detection_result
            #     print(f'zoo:{zoo_locations}')
            #     for box in zoo_locations[0]:
            #         box = box.tolist()
            #         x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            #         x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)  # 解析返回的位置信息
            #         conf_zoo = box[4]
            #         boxes = [x1, y1, x2, y2]
            #         # 设置斑马线的类别标签为 100
            #         zebra_label = 100
            #         label_text = f'Zebra Crossing: {zebra_label} {conf_zoo}'
            #         # annotator.box_label(boxes,label_text,color=(0,255,0))   #斑马线绘制

            if time_elapsed >= record_interval:  # 每2秒记录一次
                self.vehicle_counts_over_time[current_time.strftime('%H:%M:%S')] = self.vehicle_count
                self.vehicle_count = 0  # 重置计数
                last_record_time = current_time

            if frame_count % self.vid_stride == 0:

                frame_start_time = time.time()  # 每一帧检测开始时间

                self.car_status.clear()
                self.class_counts.clear()

                self.image_detect(im)

                # Process detections
                # 这里的i就是1 其实不用index 为多视频源保留一个口子
                for i, det in enumerate(self.pred):  # detections per image  这里的i是索引视频源的 当多视频源时 就需要i来进行索引 目前系统不需要多视频源
                    if self.webcam:  # bs >= 1
                        p, im0, frame = path[i], im0s[i].copy(), self.dataset.count

                    else:
                        p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    txt_path = str(self.save_dir / 'labels' / p.stem) + (
                        '' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt output/track/exp/labels
                    (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
                    curr_frames[i] = im0
                    imc = im0.copy()
                    annotator = Annotator(im0, line_width=1, example=str(self.names), pil=True, font='simkai.ttf',
                                          font_size=20)

                    if self.strongsort_cfg.STRONGSORT.ECC:  # camera motion compensation
                        self.strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            # 统计类别计数
                            if self.names[int(c)] in ['person','bicycle','biker', 'car', 'pedestrian','motorcycle','bus','truck']:  # 红绿灯不统计
                                self.class_counts[self.names[int(c)]] = n.item()
                        total_vehicles = (self.class_counts.get('car', 0) +
                                          self.class_counts.get('bus', 0) +
                                          self.class_counts.get('truck', 0))

                        # print(self.class_counts)

                        # 向前端传递类别数量 是一个字典

                        ###################
                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]
                        # pass detections to strongsort
                        self.outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                        """
                        outputs[i] 包含这张图片中所有目标的信息  i是视频源索引    
                        """

                        # 这里进行测速代码
                        if len(self.outputs_prev) < 2:
                            self.outputs_prev.append(self.outputs[i])
                        else:
                            self.outputs_prev[:] = [self.outputs_prev[-1], self.outputs[i]]
                        # draw boxes for visualization
                        if len(self.outputs[i]) > 0:
                            # 单一检测目标处理
                            for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):

                                # output 每一个跟踪对象的信息
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]
                                car_status_dict = {"id": int(id), "licence": '', "speed": '',
                                                   "illegal": ''}
                                if  self.names[int(cls)] in ['car','bus','truck']:
                                    self.vehicle_count += 1


                                    x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 确保坐标是整数
                                    if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                                        cropped_image = im0s[y1:y2, x1:x2]
                                        if cropped_image.size > 0:  # 检查图像尺寸
                                            self.license_input_queue.put((id, cropped_image)) # 将车牌图像放入车牌识别队列


                                            # t = self.licence.detectLicence(cropped_image)
                                            # if t != None:
                                            #     label_li, conf = t
                                            #
                                            #     if (id not in self.car_licence or conf > self.car_licence[id][
                                            #         'confidence']) and label_li != '':
                                            #         label_li = list(label_li)
                                            #         label_li[0] = '川'
                                            #         label_li = ''.join(label_li)
                                            #         self.car_licence[id] = {'licence': label_li,
                                            #                                 'confidence': conf}








                                        else:
                                            logger.warning("Invalid cropped image size.")
                                    else:
                                        logger.warning("Invalid crop coordinates.")


                                    if len(self.outputs_prev) == 2:
                                        cardiection = calculateDirection(self.outputs_prev[-2], output, id)
                                        if cardiection == 'LEFT' or cardiection == 'RIGHT':
                                            speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id,
                                                                                   self.fps, self.vid_stride, 1)
                                        else:
                                            speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id,
                                                                                   self.fps, self.vid_stride)
                                        if speed != ' ':
                                            speed = round(speed, 1)
                                            bbox_speed = str(speed) + "km/h"
                                        else:
                                            bbox_speed = ''
                                        car_status_dict['speed'] = bbox_speed

                                while not self.license_output_queue.empty():
                                    thread_id, license_detection_result = self.license_output_queue.get()
                                    if license_detection_result is not None:
                                        label_licence,label_conf = license_detection_result
                                        # 检查是否已存在该车辆的车牌信息，且新识别的车牌置信度是否更高
                                        if (thread_id not in self.car_licence or label_conf > self.car_licence[thread_id][
                                            'confidence']) and label_licence != '':
                                            # 更新车牌信息和置信度
                                            label_licence = list(label_licence)
                                            label_licence[0] = '鲁'
                                            label_licence = ''.join(label_licence)
                                            self.car_licence[thread_id] = {'licence': label_licence,
                                                                           'confidence': label_conf}

                                if SpeedOverFlag and self.names[int(cls)] in ['car','bus','truck']:

                                    if id is not None:
                                        save_one_box(torch.Tensor(bboxes), imc,
                                                     file=self.save_dir / 'speedover' / self.names[int(cls)] /
                                                          f"{str(int(id))}" / f'{speed}.jpg',
                                                     BGR=True)
                                if id in self.car_licence:
                                    car_status_dict['licence'] = self.car_licence[id]['licence']

                                if self.save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(
                                            ('%g ' * 12 + '\n') % (frame + 1, id, conf, cls, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                                if self.save_vid or self.show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id
                                    if self.names[c] in ['car','bus','truck']:
                                        label = None if self.hide_labels else (
                                            f' {self.names[c]}')
                                        label += f" {bbox_speed}  "
                                        if id in self.car_licence:
                                            label += f"{self.car_licence[id]['licence']}"
                                    else:
                                        label = None if self.hide_labels else (
                                            f' {self.names[c]}' )

                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                # 将车辆状态信息放到一个汇总列表里 以字符串形式
                                self.update_or_add(car_status_dict)

                    else:
                        self.strongsort_list[i].increment_ages()
                        logger.warning('No detections')

                    prev_frames[i] = curr_frames[i]
                    im0 = annotator.result()

                    # 创建一个可写的副本
                    im0_writable = im0.copy()
                    draw_class_counts(im0_writable, self.class_counts)
                    if len(self.car_status) > 0:
                        # print(self.car_status)

                        """
                        向前端传递 汽车信息表  type:列表
                        """

                        ''''''''''''''''''''''''''''''''
                        height, width = im0.shape[:2]  # 获取图像的高度和宽度
                        # 设定一个从右边界向左的偏移量
                        right_margin = 10
                        # 计算绘制文本的起始位置
                        # 假设每行文本的大概宽度是400像素，这个值可以根据实际情况调整
                        text_width = 800
                        start_x = width - text_width - right_margin
                        start_y = 20  # 从图像顶部向下的偏移量
                        # 绘制车辆状态信息
                        draw_texts(im0_writable, self.car_status, start_pos=(start_x, start_y),
                                   color=(0, 255, 0))
                    self.im0_s = im0_writable.copy()



                    pynvml.nvmlInit()
                    handles = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handles)
                    gpu_memory[(current_time-video_start_time).seconds] = meminfo.used / 1024 ** 3

                    logger.info(f"\n显存占用:{meminfo.used / 1024 ** 3}g\n")  # 已用显存大小

            # Stream results
            if self.im0_s is None:
                self.im0_s = im0s.copy()

            if not self.image_display_queue.full():
                # 如果队列未满，直接放入新帧
                self.image_display_queue.put_nowait(self.im0_s)
            else:
                # 如果队列已满，先出队一个元素，再入队新帧
                # 这一步确保队列中总是有最新的帧，避免重复显示同一帧
                self.image_display_queue.get_nowait()  # 出队一个旧帧
                self.image_display_queue.put_nowait(self.im0_s)  # 入队一个新帧

            self.image_display()

            # Save results (image with detections)
            # 但视频源保存
            if self.save_vid:

                if self.vid_writer is None:
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = self.fps
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, self.im0_s.shape[1], self.im0_s.shape[0]

                    video_name = Path(path).name
                    vid_save_path = increment_path(
                        Path('D:/traffic_detection/output/track/video'), mkdir=True, exist_ok=True)
                    save_path = str(
                        Path('D:/traffic_detection/output/track/video/' + str(video_name)).with_suffix('.mp4'))

                    self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                self.vid_writer.write(self.im0_s)

        # 循环结束后，确保释放资源
        if self.vid_writer:
            self.vid_writer.release()
        self.license_input_queue.put(None)  # 发送停止信号
        self.zebra_input_queue.put(None)
        license_thread.join()

        cv2.destroyAllWindows()

        # Print results

        self.vehicle_counts_over_time[datetime.now().strftime('%H:%M:%S')] = self.vehicle_count

        self.plot_counts(self.vehicle_counts_over_time, (str(self.save_dir) + '/' + f"traffic car count {datetime.now().strftime('%Y-%m-%d')}"), 'vehicle',
                         'Time')

        self.plot_counts(gpu_memory, (str(self.save_dir) + '/' + 'gpu memory count'), 'gpu memory', 'time')

        if self.save_txt or self.save_vid:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} tracks saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov9-c.pt',
                        help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='video_10s.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')

    parser.add_argument('--augment', action='store_true', help='')

    parser.add_argument('--classes', default=[0, 1, 2, 3, 5, 7, 9], nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='output/track', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True, action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=2, help='video frame-rate stride')
    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if only one size


    opt.save_dir = increment_path(Path(opt.project), exist_ok=opt.exist_ok)  # increment run output/track/exp
    clear_directory(opt.save_dir)
    # make dir
    return opt


def main(opt):
    logger.info(f"Initializing VideoTracker with options: {opt}")
    video_tracker = VideoTracker(opt)
    logger.info('Init successfully')
    video_tracker.run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
