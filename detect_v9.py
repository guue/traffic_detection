import argparse
import math

import os
import platform
import threading
from queue import Queue

from licence import Licence

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from trafficLine import *
from draw import *
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('detect')
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("detect.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter) #设置文件的log格式
logger.addHandler(handler)
class VideoTracker:
    def __init__(self, opt):
        self.vid_flag = 0
        self.im0_s = None
        self.pred = None
        self.webcam = None
        self.data = opt.data
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

        """
        用outputs[i]索引分类和id 如果cls == car 将其以str append到每一个车的状态信息列表car_status里 再加速度和方向信息  通过license_id 索引其车牌信息
        储存格式应该是 ['id: license: dir: speed: illegal: '] illegal 显示其违法信息  如果涉及违法 则在ui中标红
        在一开始 illegal 默认为 None 
        先在run里定义一个car_status_dir key：id,license,dir,speed,illegal
        ilegal 默认为 None
        if SpeedOverFlag
        dir['illegal'] = SpeedOver 
        warning()
            ui 车辆信息标红
            并对车辆信息（车牌号 违法信息）保存在excel里 

        每一帧图片都在不断更新这个dir
        每更新一个dir
        就以str格式 append到car_status_list上 
        当这一帧所有list都更新完毕后    清空之前的ui显示  加入新ui显示
        """
        self.car_licence = {}  # {id:license}
        self.car_status = []  # 存放每辆车的状态

        # Other parameters
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.save_txt = opt.save_txt
        self.save_vid = opt.save_vid
        self.hide_labels = opt.hide_labels
        self.hide_conf = opt.hide_conf
        self.hide_class = opt.hide_class
        self.augment = opt.augment
        self.max_det = opt.max_det
        self.class_counts = {}
        self.vehicle_count = 0
        self.vehicle_counts_over_time = {}  # 时间戳: 车辆总数
        self.last_time_stamp = 0

        # Placeholder for FPS calculation
        self.fps = 0

    def plot_vehicle_counts(self, vehicle_counts, save_path):
        times = list(vehicle_counts.keys())
        counts = list(vehicle_counts.values())

        plt.figure(figsize=(10, 6))
        plt.plot(times, counts, marker='o', linestyle='-', color='b')
        plt.title(' Vehicle Count')
        plt.xlabel('Time (seconds)')
        plt.ylabel(' Vehicle Count')
        plt.grid(True)

        # 保存图表到指定路径
        plt.savefig(save_path)
        plt.close()  # 关闭图表窗口，释放资源

    def update_or_add(self, car_status_dict):

        """
        car_status_dict即将要添加的车辆状态信息
        """

        updated = False
        for i, status_str in enumerate(self.car_status):
            status_dict = eval(status_str)  # 假设每个字符串都能安全地转换回字典

            if status_dict['licence'] == car_status_dict['licence'] and status_dict['licence'] != '' and \
                    car_status_dict['licence'] != '':
                # 找到相同license的字典，更新值
                self.car_status[i] = str(car_status_dict)
                updated = True
                break
        if not updated:
            # 没有找到相同license的字典，添加新的字典
            self.car_status.append(str(car_status_dict))

    def loadmodel(self, weights):
        model = DetectMultiBackend(Path(weights), device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        return model

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
                                  vid_stride=self.vid_stride)
            self.batchsize = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
                                 vid_stride=self.vid_stride)

        vid_path, vid_writer, txt_path = [None] * self.batchsize, [None] * self.batchsize, [
            None] * self.batchsize  # 多视频源才需要用列表初始化每个视频源的路径
        return dataset

    @smart_inference_mode()
    def run(self):
        self.yolo_model.warmup(imgsz=(1 if self.pt or self.yolo_model.triton else self.bs, 3, *self.imgsz))
        frame_count = 0
        global cardiection, CarLicence, label

        curr_frames, prev_frames = [None] * self.batchsize, [None] * self.batchsize

        start_time = time.time()
        for path, im, im0s, vid_cap, s in self.dataset:

            current_time = time.time() - start_time
            frame_count += 1
            # s = ''
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            if current_time - self.last_time_stamp >= 2:  # 每2秒记录一次
                self.vehicle_counts_over_time[int(current_time)] = self.vehicle_count
                self.vehicle_count = 0  # 重置计数
                self.last_time_stamp = current_time

            if frame_count % 2 == 0:
                self.car_status.clear()
                self.class_counts.clear()
                # Inference
                self.pred = self.yolo_model(im, augment=self.augment)
                self.pred = self.pred[0][1]
                # Apply NMS
                self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes,
                                                self.agnostic_nms, max_det=self.max_det)
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
                    SpeedOver = False
                    # 斑马线检测
                    ret, location = detectTrafficLine(im0)

                    # s += '%gx%g ' % im.shape[2:]  # print string
                    imc = im0.copy()
                    annotator = Annotator(im0, line_width=1, example=str(self.names), pil=True, font='simkai.ttf')

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
                            if c != 9:  # 红绿灯不统计
                                self.class_counts[self.names[int(c)]] = n.item()
                        # draw_class_counts(im0, self.class_counts)
                        print(self.class_counts)

                        # 像前端传递类别数量 是一个字典

                        ###################

                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]

                        # pass detections to strongsort

                        self.outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                        """
                        outputs[i] 是检测结果 id为i的目标信息  
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
                                car_status_dict = {"id": id, "licence": '', "speed": '',
                                                   "illegal": ''}

                                if int(cls) in [2, 5, 7]:
                                    self.vehicle_count += 1
                                    if self.car_licence.get(id, 0) == 0:
                                        x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
                                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 确保坐标是整数
                                        if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                                            cropped_image = im0s[y1:y2, x1:x2]
                                            if cropped_image.size > 0:  # 检查图像尺寸
                                                t = self.licence.detectLicence(cropped_image)
                                                if t != None:
                                                    xyxy, label = t
                                                    self.car_licence[id] = label
                                        else:
                                            logger.warning("Invalid cropped image size.")
                                    else:
                                        logger.warning("Invalid crop coordinates.")

                                    if id in self.car_licence:
                                        car_status_dict['licence'] = self.car_licence[id]
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)

                                    if len(self.outputs_prev) == 2:
                                        cardiection = calculateDirection(self.outputs_prev[-2], output, id)
                                        if cardiection == 'LEFT' or cardiection == 'RIGHT':
                                            speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id,
                                                                                   fps, 2, 1)
                                        else:
                                            speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id,
                                                                                   fps, 2)
                                        if speed != ' ':
                                            speed = round(speed, 1)
                                            bbox_speed = str(speed) + "km/h"
                                        else:
                                            bbox_speed = ''
                                        car_status_dict['speed'] = bbox_speed
                                        if SpeedOverFlag:
                                            if id is not None:

                                                save_one_box(torch.Tensor(bboxes), imc,
                                                             file=self.save_dir / 'speedover' / self.names[int(cls)] /
                                                                  f'{self.car_licence[id] if id in self.car_licence else id}' / f'{speed}.jpg',
                                                             BGR=True)
                                            elif id is None or id == '':
                                                save_one_box(torch.Tensor(bboxes), imc,
                                                             file=self.save_dir / 'speedover' / self.names[
                                                                 int(cls)] / f'{speed}.jpg'
                                                             , BGR=True)
                                            car_status_dict['illegal'] += 'SpeedOver '
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

                                if self.save_vid or self.save_crop or self.show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id
                                    if c in [2, 5, 7]:
                                        label = None if self.hide_labels else (
                                            f'{id} {self.names[c]}' if self.hide_conf else
                                            (
                                                f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                                        label += f" {bbox_speed}  "
                                        if id in self.car_licence:
                                            label += f"{self.car_licence[id]}"
                                    else:
                                        label = None if self.hide_labels else (
                                            f'{id} {self.names[c]}' if self.hide_conf else
                                            (
                                                f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))

                                    annotator.box_label(bboxes, label, color=colors(c, True))

                                if ret:
                                    # 如果检测到斑马线，则在im0上绘制检测框
                                    # 假设location是一个包含两个坐标点的元组，表示检测框的对角线上的两个点
                                    cv2.rectangle(im0, location[0], location[1], (0, 255, 0), 2)
                                    # print(location)

                                # 将车辆状态信息放到一个汇总列表里 以字符串形式
                                self.update_or_add(car_status_dict)


                    else:
                        self.strongsort_list[i].increment_ages()
                        logger.warning('No detections')

                    if len(self.car_status) > 0:
                        print(self.car_status)

                        """
                        向前端传递 汽车信息表  type:列表
                        """

                        ''''''''''''''''''''''''''''''''
                    #     height, width = im0.shape[:2]  # 获取图像的高度和宽度
                    #
                    #     # 设定一个从右边界向左的偏移量
                    #     right_margin = 10
                    #
                    #     # 计算绘制文本的起始位置
                    #     # 假设每行文本的大概宽度是400像素，这个值可以根据实际情况调整
                    #     text_width = 800
                    #     start_x = width - text_width - right_margin
                    #     start_y = 20  # 从图像顶部向下的偏移量
                    #
                    #     # # 绘制车辆状态信息
                    #     # draw_texts(im0, self.car_status, start_pos=(start_x, start_y),
                    #     #            color=(255, 255, 255))
                    prev_frames[i] = curr_frames[i]
                    im0 = annotator.result()
                    self.im0_s = im0.copy()

            windows = []
            # Stream results
            if self.im0_s is None:
                self.im0_s = im0s.copy()

            if self.show_vid:
                if platform.system() == 'Linux' and 'detection' not in windows:
                    windows.append('detection')
                    cv2.namedWindow('detection',
                                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow('detection', self.im0_s.shape[1], self.im0_s.shape[0])
                cv2.imshow('detection', self.im0_s)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 但视频源保存
            if self.save_vid:

                if self.vid_writer is None:
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, self.im0_s.shape[1], self.im0_s.shape[0]



                    video_name = Path(path).name
                    save_path = str(
                        Path('D:/traffic_detection/output ' + str(video_name)).with_suffix('.mp4'))



                    self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))



                self.vid_writer.write(self.im0_s)



        # 循环结束后，确保释放资源
        if self.vid_writer:
            self.vid_writer.release()

        # Print results
        self.vehicle_counts_over_time[int(time.time() - start_time)] = self.vehicle_count

        self.plot_vehicle_counts(self.vehicle_counts_over_time, (str(self.save_dir) + '/' + 'traffic car count'))

        if self.save_txt or self.save_vid:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} tracks saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='YOLOv9/data/coco.yaml',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov9-c.pt',
                        help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='video_20s.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')
    parser.add_argument('--augment', action='store_true', help='')

    parser.add_argument('--classes', default=[0, 1, 2, 3, 5, 7, 9], nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='D:/traffic_detection/output/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=False, action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if only one size

    if not isinstance(opt.yolo_weights, list):  # single yolo model
        exp_name = opt.yolo_weights.stem
    elif type(opt.yolo_weights) is list and len(opt.yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(opt.yolo_weights[0]).stem

    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = opt.name if opt.name else exp_name + "_" + opt.strong_sort_weights.stem
    opt.save_dir = increment_path(Path(opt.project) / exp_name, exist_ok=opt.exist_ok)  # increment run output/track/exp
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
