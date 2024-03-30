import argparse

import os

from ultralytics.utils.plotting import save_one_box

from licence import Licence
from zebra_detector.test import detect_zoo

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
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from calculateDirection import calculateDirection
from Estimated_speed import Estimated_speed

from draw import *

import matplotlib.pyplot as plt

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
import pynvml





class VideoTrack:
    def __init__(self, opt):
        self.process_time = {}
        self.im0_s = None
        self.pred = None
        self.webcam = None
        self.device = select_device(opt.device)
        self.half = opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        self.save_dir = opt.save_dir
        self.yolo_weights = opt.yolo_weights

        self.licence = Licence()

        self.carlicence = ''
        # Load YOLO model
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.yolo_model = self.loadmodel(Path(opt.yolo_weights))
        self.names, = self.yolo_model.names,
        self.stride = self.yolo_model.stride.max().cpu().numpy()  # Model stride

        self.source = str(opt.source)
        self.show_vid = opt.show_vid
        self.imgsz = check_img_size(opt.imgsz[0], s=self.stride)

        self.dataset, self.vid_path, self.vid_writer, self.txt_path, self.nr_sources = self.dateset_Loader()

        # Load StrongSORT
        self.strongsort_cfg = get_config()
        self.strongsort_cfg.merge_from_file(opt.config_strongsort)
        self.strongsort_list = []  # 被跟踪对象
        for i in range(self.nr_sources):
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
        self.outputs = [None] * self.nr_sources
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
        self.save_crop = opt.save_crop
        self.save_txt = opt.save_txt
        self.save_vid = opt.save_vid
        self.hide_labels = opt.hide_labels
        self.hide_conf = opt.hide_conf
        self.hide_class = opt.hide_class

        self.class_counts = {}

        self.vehicle_count = 0
        self.vehicle_counts_over_time = {}  # 时间戳: 车辆总数
        self.last_time_stamp = 0

        # Placeholder for FPS calculation
        self.fps = 0
    def plot_counts(self, count, save_path,title,xlabel):
        times = list(count.keys())
        counts = list(count.values())

        plt.figure(figsize=(10, 6))
        plt.plot(times, counts, marker='o', linestyle='-', color='b')
        plt.title(title+' Count')
        plt.xlabel(xlabel)
        plt.ylabel(title+' Count')
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
        model = attempt_load(Path(weights), map_location=self.device)
        return model

    def dateset_Loader(self):
        is_file = Path(self.source).suffix[1:] in (VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Dataloader
        if self.webcam:
            self.show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
            nr_sources = len(dataset.sources)
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
        return dataset, vid_path, vid_writer, txt_path, nr_sources

    def run(self):
        gpu_memory = {}
        frame_count = 0
        global cardiection, CarLicence, label, save_path
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        curr_frames, prev_frames = [None] * self.nr_sources, [None] * self.nr_sources

        start_time = time.time()

        for frame_idx, (path, im, im0s, vid_cap) in enumerate(self.dataset):

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
                frame_start_time = time.time()  # 每一帧检测开始时间
                self.class_counts.clear()
                # Inference
                self.pred = self.yolo_model(im)
                # Apply NMS
                self.pred = non_max_suppression(self.pred[0], self.conf_thres, self.iou_thres, self.classes,
                                                self.agnostic_nms)
                # Process detections
                for i, det in enumerate(self.pred):  # detections per image  这里的i是索引视频源的 当多视频源时 就需要i来进行索引 目前系统不需要多视频源
                    if self.webcam:  # nr_sources >= 1
                        p, im0, _ = path[i], im0s[i].copy(), self.dataset.count
                        p = Path(p)  # to Path

                        txt_file_name = p.name
                        save_path = str(self.save_dir / p.name) + str(i)  # im.jpg, vid.mp4, ...
                    else:
                        p, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        # video file
                        if self.source.endswith(VID_FORMATS):
                            txt_file_name = p.stem
                            save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                        # folder with imgs
                        else:
                            txt_file_name = p.parent.name  # get folder name containing current img
                            save_path = str(self.save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

                    curr_frames[i] = im0
                    SpeedOver = False
                    # 斑马线检测
                    zoo_locations, scale_h, scale_w = detect_zoo(im0)
                    # print(f'zoo:{zoo_locations}')
                    for box in zoo_locations[0]:
                        box = box.tolist()
                        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)  # 解析返回的位置信息
                        conf_zoo = box[4]
                        boxes = [x1, y1, x2, y2]
                        # 设置斑马线的类别标签为 100
                        zebra_label = 100
                        label_text = f'Zebra Crossing: {zebra_label} {conf_zoo}'
                        # annotator.box_label(boxes,label_text,color=(0,255,0))   #斑马线绘制

                    txt_path = str(self.save_dir / 'tracks' / txt_file_name)  # im.txt
                    # s += '%gx%g ' % im.shape[2:]  # print string
                    imc = im0.copy() if self.save_crop else im0  # for save_crop

                    if self.strongsort_cfg.STRONGSORT.ECC:  # camera motion compensation
                        self.strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            # 统计类别计数
                            if c != 9:  # 红绿灯不统计
                                self.class_counts[self.names[int(c)]] = n
                        draw_class_counts(im0, self.class_counts)

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
                                            pass
                                            # print("Invalid cropped image size.")
                                    else:
                                        pass
                                        # print("Invalid crop coordinates.")
                                if id in self.car_licence:
                                    car_status_dict['licence'] = self.car_licence[id]

                                fps = vid_cap.get(cv2.CAP_PROP_FPS)

                                SpeedOverFlag = False

                                if len(self.outputs_prev) == 2 and int(cls) in [2, 5, 7]:
                                    cardiection = calculateDirection(self.outputs_prev[-2], output, id)

                                    if cardiection == 'LEFT' or cardiection == 'RIGHT':
                                        bbox_speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id,
                                                                                    fps,
                                                                                    2, 1)
                                    else:
                                        bbox_speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id,
                                                                                    fps, 2)

                                    car_status_dict['speed'] = bbox_speed

                                if SpeedOverFlag and c in [2, 5, 7]:
                                    # self.save_crop = True
                                    SpeedOver = True
                                    car_status_dict['illegal'] += 'SpeedOver '

                                if self.save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                       bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                                if self.save_vid or self.save_crop or self.show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id
                                    if c in [2, 5, 7]:
                                        label = None if self.hide_labels else (
                                            f'{id} {self.names[c]}' if self.hide_conf else
                                            (
                                                f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                                        label += f" {bbox_speed} {cardiection} "
                                        if id in self.car_licence:
                                            label += f"{self.car_licence[id]}"
                                    else:
                                        label = None if self.hide_labels else (
                                            f'{id} {self.names[c]}' if self.hide_conf else
                                            (
                                                f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))

                                    plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                                    if self.save_crop:
                                        txt_file_name = txt_file_name if (
                                                isinstance(path, list) and len(path) > 1) else ''
                                        if SpeedOver:
                                            save_one_box(torch.Tensor(bboxes), imc,
                                                         file=self.save_dir / 'speedover' / txt_file_name / self.names[
                                                             c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                                            self.save_crop = False

                                        else:
                                            save_one_box(torch.Tensor(bboxes), imc,
                                                         file=self.save_dir / 'crops' / txt_file_name / self.names[
                                                             c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                                            self.save_crop = False



                                # draw_text_on_frame(im0, f"overspeed:{' '.join(self.overspeedcar)}",
                                #                    background_color=(0, 0, 0))

                                # 将车辆状态信息放到一个汇总列表里 以字符串形式
                                self.update_or_add(car_status_dict)
                                # self.car_status.append(str(car_status_dict))

                    else:
                        self.strongsort_list[i].increment_ages()
                        print('No detections')

                    if len(self.car_status) > 0:
                        # print(self.car_status)
                        height, width = im0.shape[:2]  # 获取图像的高度和宽度

                        # 设定一个从右边界向左的偏移量
                        right_margin = 10

                        # 计算绘制文本的起始位置
                        # 假设每行文本的大概宽度是400像素，这个值可以根据实际情况调整
                        text_width = 800
                        start_x = width - text_width - right_margin
                        start_y = 20  # 从图像顶部向下的偏移量

                        # 绘制车辆状态信息
                        draw_texts(im0, self.car_status, start_pos=(start_x, start_y),
                                   color=(255, 255, 255))
                    prev_frames[i] = curr_frames[i]

                    self.im0_s = im0.copy()
                    frame_end_time = time.time()  # 结束处理帧的时间

                    frame_processing_time = frame_end_time - frame_start_time  # 计算这一帧的处理时间
                    print(f"Frame {frame_count} processing time: {frame_processing_time:.4f} seconds.")

                    self.process_time[int(frame_count)]=frame_processing_time

                    pynvml.nvmlInit()
                    handles = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handles)


                    gpu_memory[int(frame_count)]=meminfo.used / 1024 ** 3

                    print(f"\n显存占用:{meminfo.used / 1024 ** 3}g\n")  # 已用显存大小


            if self.im0_s is None:
                if self.webcam:  # nr_sources = 1
                    p, self.im0_s, _ = path, im0s.copy(), self.dataset.count
                    p = Path(p)  # to Path
                    save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, self.im0_s, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if self.source.endswith(VID_FORMATS):
                        save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        save_path = str(self.save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            # Stream results
            if self.show_vid:
                cv2.imshow(str(p), self.im0_s)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 但视频源保存
            if self.save_vid:
                if self.vid_path != save_path:  # new video
                    self.vid_path = save_path
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, self.im0_s.shape[1], self.im0_s.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                      (w, h))
                self.vid_writer.write(self.im0_s)


        # Print results
        self.vehicle_counts_over_time[int(time.time() - start_time)] = self.vehicle_count

        self.plot_counts(self.vehicle_counts_over_time, (str(self.save_dir) + '/' + 'traffic car count'),'vehicle','Time/ms')
        self.plot_counts(self.process_time,(str(self.save_dir) + '/' + 'process time count'),'process time','frame_idx')
        self.plot_counts(gpu_memory,(str(self.save_dir) + '/' + 'gpu memory count'),'gpu memory','frame_idx')

        if self.save_txt or self.save_vid:
            s = f"\n{len(list(self.save_dir.glob('tracks/*.txt')))} tracks saved to {self.save_dir / 'tracks'}" if self.save_txt else ''
            print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7-tiny.pt',
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
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=False, action='store_true', help='save cropped self.prediction boxes')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')

    parser.add_argument('--classes', default=[0, 1, 2, 3, 5, 7, 9], nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default=ROOT / 'output/track_v7', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True, action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if only one size

    if not isinstance(opt.yolo_weights, list):  # single yolo model
        exp_name = opt.yolo_weights.stem
    elif type(opt.yolo_weights) is list and len(opt.yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(opt.yolo_weights[0]).stem

    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = opt.name if opt.name else exp_name + "_" + opt.strong_sort_weights.stem
    opt.save_dir = Path(increment_path(Path(opt.project) / exp_name, exist_ok=opt.exist_ok))  # increment run
    (opt.save_dir / 'tracks' if opt.save_txt else opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    return opt


def main(opt):
    print(f"Initializing VideoTracker with options: {opt}")
    video_tracker = VideoTrack(opt)
    print('Init successfully')
    video_tracker.run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
