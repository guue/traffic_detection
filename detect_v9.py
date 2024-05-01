import argparse
import base64
import math
import shutil
import threading
import queue
import os
import time
import traceback

from connect import send_json_msg
from color_det import get_color
from velocity_cal.cal_velocity import calculate_average_speed
from zebra_det import ZebraDetection
from licence import Licence
from datetime import datetime, timedelta
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import pynvml
import logging
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'YOLOv9') not in sys.path:
    sys.path.append(str(ROOT / 'YOLOv9'))  # add yolov9 ROOT to PATH
if str(ROOT / 'ultralytics') not in sys.path:
    sys.path.append(str(ROOT / 'ultralytics'))  # add yolov8 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from YOLOv9.models.common import DetectMultiBackend
from YOLOv9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from YOLOv9.utils.general import (check_file, check_img_size, check_imshow,
                                  colorstr, increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from YOLOv9.utils.torch_utils import select_device, smart_inference_mode
from YOLOv9.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# from draw import *

# 日志
logger = logging.getLogger('detect')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("detect.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)  # 设置文件的log格式
logger.addHandler(handler)

import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: X={x}, Y={y}")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Detection', param)


def parse_str2dir(status_str):
    """
    将存储在car_status中的字符串转换为字典。
    特别处理illegal_behavior作为列表，并处理其内部的逗号分隔问题。

    参数:
        status_str (str): 字符串，格式为 "key: value, key: value, ..."

    返回:
        dict: 转换后的字典。
    """
    status_dict = {}

    # 尝试按正常键值对进行分割
    items = status_str.split(", ")
    for item in items:
        if ": " in item:
            key, value = item.split(": ", 1)
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif key == 'illegal_behavior':
                # 处理illegal_behavior为列表的情况
                # 由于illegal_behavior的值可能已被逗号分隔，我们需要特别处理这个字段
                # 这里不使用分隔符替换的方法，而是直接分割可能存在的单一或多个违规行为
                value = [v.strip() for v in value.split("|")]
            status_dict[key] = value
        else:
            # 对于无法正常解析的情况，尤其是illegal_behavior，我们将它直接作为列表的单个元素处理
            # 这里的处理意味着我们假定无法解析的项都属于illegal_behavior
            if 'illegal_behavior' not in status_dict:
                status_dict['illegal_behavior'] = [item.strip()]
            else:
                status_dict['illegal_behavior'].append(item.strip())

    return status_dict


def clear_directory(dir_path):
    """
    清空保存文件夹 覆盖保存

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
        self.platelength_car = {}
        self.client_socket = None
        self.traffic_light_color = 'green'
        self.zoo_box = []
        self.zebra_detected = False
        self.detection_thread = None

        self.is_running = False
        self.lock = threading.Lock()

        self.image_display_queue = queue.Queue(maxsize=10)  # 前端可直接读此队列进行显示
        self.license_output_queue = queue.Queue()
        self.license_input_queue = queue.Queue(10)
        self.car_license_history = {}  # {id: [last_license, second_last_license]

        self.im0_s = None
        self.pred = None
        self.webcam = None
        self.dnn = opt.dnn
        self.device = select_device(opt.device)
        self.half = opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        self.save_dir = opt.save_dir
        self.yolo_weights = opt.yolo_weights
        self.licence = Licence()

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

        self.car_locations_history = {}  # {id: [(frame_idx, (x1, y1)), ...], ...}
        self.car_history = {}  # 新增一个字典来保存每辆车的历史状态信息

        self.potential_red_light_runners = {}  # {car_id: {"entered": False, "red_light": False, "positions": []}}

        self.car_licence = {}  # {id:{license:licence,conf:conf},....}

        self.car_status = []  # 存放每辆车的状态 ["id: license: speed: illegal: ","id: license: speed: illegal: ",..]
        self.illegal_car_status = []  # 存放违规车辆信息
        self.class_counts = {}  # {car:2,person:1,...}

        # Other parameters
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms

        self.save_vid = opt.save_vid
        self.hide_labels = opt.hide_labels
        self.augment = opt.augment
        self.max_det = opt.max_det

        # 车流统计参数
        self.vehicle_count = 0
        self.counted_ids = set()  # 初始化一个集合，用于存储已经计数过的ID
        self.vehicle_counts_over_time = {}  # 时间戳: 车辆总数
        self.fps = 0

    def update_red_light_runners(self, car_id, bbox, im0):
        try:
            """
            根据车辆位置和红绿灯状态更新可能闯红灯的车辆信息。
    
            Parameters:
            - car_id: 车辆ID。
            - bbox: 车辆的边界框 [x1, y1, x2, y2]。
            - traffic_light_color: 红绿灯颜色。
            - zebra_box: 斑马线区域 [x1, y1, x2, y2]。
            - im0:一张干净的图片
            """
            x1, y1, x2, y2 = bbox
            zebra_x1, zebra_y1, zebra_x2, zebra_y2 = self.zoo_box

            # 初始化车辆信息
            if car_id not in self.potential_red_light_runners:
                self.potential_red_light_runners[car_id] = {"start_record": False, "entered": False,
                                                            "red_light_detected": False, "positions": [],
                                                            "initial_img": None}

            car_info = self.potential_red_light_runners[car_id]
            # 检查车辆左上角是否进入斑马线区域
            if x1 >= zebra_x1 and y1 >= zebra_y1 and not car_info["entered"]:
                car_info["start_record"] = True
                if self.traffic_light_color in ["red1", "red2"]:
                    car_info["red_light_detected"] = True
            # 记录车辆位置
            if car_info["start_record"]:
                car_info["positions"].append((x1, y1, x2, y2))

            # 检查车辆是否完全进入斑马线区域
            if x2 <= zebra_x2 and y2 <= zebra_y2 and car_info["start_record"]:
                car_info["entered"] = True
            if car_info["start_record"] and not car_info.get("initial_img_saved", False):
                car_info["initial_img"] = im0.copy()  # 保存当前帧作为初始截图
                car_info["initial_bbox"] = bbox  # 保存初始位置的边界框
                car_info["initial_img_saved"] = True
        except Exception as e:
            logger.warning(e)

    def check_red_light_violations(self):
        """
        当检测到车辆左上角进入斑马线区域时为红灯 开始记录其信息
        一直记录到当其右下角进入斑马线区域 即车身全部进入斑马线
        如果记录中红绿灯一直为红 则视为闯红灯
        """
        chuanghongdeng = []
        for car_id, info in list(self.potential_red_light_runners.items()):
            if info["entered"] and info["red_light_detected"]:
                chuanghongdeng.append(car_id)
                # 获取初始和最终截图以及对应的边界框
                initial_img = info["initial_img"]
                initial_bbox = info["initial_bbox"]
                final_img = self.im0_c.copy()
                final_bbox = info["positions"][-1]  # 最后记录的位置

                # 保存截图
                self.save_violation_snapshot(car_id, initial_img, initial_bbox, "initial", 'chuanghongdeng')
                self.save_violation_snapshot(car_id, final_img, final_bbox, "final", 'chuanghongdeng')
                # 清除记录，避免重复处理
                del self.potential_red_light_runners[car_id]

        return chuanghongdeng

    def save_violation_snapshot(self, car_id, img, bbox, snapshot_type, behavior):
        """
        保存具有边界框标注的违规车辆截图。
        Parameters:
        - car_id: 车辆ID。
        - img: 要保存的图像。
        - bbox: 车辆的边界框 [x1, y1, x2, y2]。
        - snapshot_type: 截图类型（'initial'或'final'）。
        """
        x1, y1, x2, y2 = map(int, bbox)
        save_path = os.path.join(str(self.save_dir), 'violation', str(car_id), f"{behavior}_{snapshot_type}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 创建带有边界框的图像副本
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色边界框
        font = cv2.FONT_HERSHEY_SIMPLEX
        if behavior == 'SpeedOver':
            text = f"Violation: {behavior} speed:{snapshot_type}km/h"
        else:
            text = f"Violation: {behavior}"
        text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        text_x = x1
        text_y = y2 + 20  # 边界框下方留出一些空间
        cv2.putText(marked_img, text, (text_x, text_y), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # 白色文本
        cv2.rectangle(marked_img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 0, 255),
                      -1)
        cv2.putText(marked_img, text, (text_x, text_y), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # 重新绘制文本以确保其在背景上
        # 保存图像
        cv2.imwrite(save_path, marked_img)

    def change_video(self, vid_path):
        self.source = vid_path

    def get_centers(self, box):

        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        centers = (x_center, y_center)
        return centers

    def get_size_from_bbox(self, bbox):
        """
        根据边界框计算尺寸。
        :param bbox: 边界框，格式为 (x1, y1, x2, y2)
        :return: (width, height) 宽度和高度
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width, height

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

    def update_location_history(self, car_id, frame_idx, center, size):
        if car_id not in self.car_locations_history:
            self.car_locations_history[car_id] = []
        self.car_locations_history[car_id].append((frame_idx, center, size))
        if len(self.car_locations_history[car_id]) > 20:
            self.car_locations_history[car_id].pop(0)


    def calculate_direction_for_car(self, car_id):
        if car_id not in self.car_locations_history or len(self.car_locations_history[car_id]) < 2:
            return ''
        # 获取车辆的最新两个位置
        prev_location = self.car_locations_history[car_id][-2][1]  # 倒数第二个位置
        current_location = self.car_locations_history[car_id][-1][1]  # 最后一个位置
        # 计算中心点的变化
        dx = current_location[0] - prev_location[0]
        dy = current_location[1] - prev_location[1]
        # 根据dx和dy的关系判断方向
        if dy > 0 and 3 * abs(dy) >= abs(dx):
            return 'DOWN'
        if dy < 0 and 1.5 * abs(dy) >= abs(dx):
            return 'UP'
        if dx > 0 and abs(dx) >= abs(dy):
            return 'RIGHT'
        if dx < 0 and abs(dx) >= abs(dy):
            return "LEFT"
        return 'UP'

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
                label, conf, licence_boxes = detection_result
            except Exception as e:
                logger.error(e)
                label = ''
            # 更新车牌检测历史
            if label != '':
                if id not in self.car_license_history:
                    self.car_license_history[id] = [label, None]
                else:
                    self.car_license_history[id] = [label, self.car_license_history[id][0]]
            output_queue.put((id, detection_result))


    def image_detect(self, image):
        # Inference
        self.pred = self.yolo_model(image, augment=self.augment)
        self.pred = self.pred[0][1]
        # Apply NMS
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes,
                                        self.agnostic_nms, max_det=self.max_det)

    def plot_counts(self, count, save_path, title, xlabel):
        xtick = list(count.keys())
        ytick = list(count.values())
        plt.plot(xtick, ytick, marker='o', linestyle='-', color='b')
        plt.title(title + 'Count')
        plt.xlabel(xlabel)
        plt.xticks(rotation=270)
        plt.ylabel(title)
        plt.grid(True)
        # 保存图表到指定路径
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # 关闭图表窗口，释放资源

    def TLL_DET(self, image):
        traffic_light_color = get_color(image)
        return traffic_light_color

    def update_or_add(self, car_status_dict):
        """
        car_status_dict即将要添加的车辆状态信息。
        无论是否有现存的车牌信息，都应保留违规为True的信息。
        根据新的车辆状态信息更新或添加车辆状态，
        同时考虑历史违规信息。
        """
        car_id = car_status_dict['id']
        new_illegal = car_status_dict.get('illegal', False)
        new_illegal_behavior = car_status_dict['illegal_behavior']

        # 检查该车辆是否已有历史记录
        if car_id in self.car_history:
            history = self.car_history[car_id]
            prev_illegal = history.get('illegal', False)
            prev_illegal_behavior = history['illegal_behavior']

            # 如果之前的违规信息为True，且新的信息中违规状态不是True，则保留历史中的违规信息
            if prev_illegal and not new_illegal:
                car_status_dict['illegal'] = True
                car_status_dict['illegal_behavior'] = prev_illegal_behavior
            elif prev_illegal and new_illegal:
                # 将新的违规行为记录进表中
                unique_new_illegal_behaviors = [behavior for behavior in new_illegal_behavior if
                                                behavior not in prev_illegal_behavior]
                car_status_dict['illegal_behavior'] = prev_illegal_behavior + unique_new_illegal_behaviors

        # 更新车辆的历史记录为最新的状态
        self.car_history[car_id] = car_status_dict
        updated = False
        for i, status_str in enumerate(self.car_status):
            # 从当前列表中的车辆状态字符串中解析出字典
            existing_info = parse_str2dir(status_str)
            licence_existing = existing_info.get('licence', '')

            # 检查是否满足更新条件：车牌号匹配且不为空
            if licence_existing == car_status_dict['licence'] and licence_existing != '' and car_status_dict[
                'licence'] != '':
                # 根据car_status_dict重新构建状态字符串
                illegal_behaviors_str = ", ".join(car_status_dict['illegal_behavior'])
                car_status_str = f"id: {car_status_dict['id']}, licence: {car_status_dict['licence']}, speed: {car_status_dict['speed']}, illegal: {car_status_dict['illegal']}, illegal_behavior: {illegal_behaviors_str}"
                self.car_status[i] = car_status_str
                updated = True
                break

        if not updated:
            illegal_behaviors_str = ", ".join(car_status_dict['illegal_behavior'])
            car_status_str = f"id: {car_status_dict['id']}, licence: {car_status_dict['licence']}, speed: {car_status_dict['speed']}, illegal: {car_status_dict['illegal']}, illegal_behavior: {illegal_behaviors_str}"
            self.car_status.append(car_status_str)

    def update_illegal_car_status(self):
        """
        从car_status中整合违规信息到illegal_car_status列表。
        """
        for status_str in self.car_status:
            car = parse_str2dir(status_str)  # 将字符串转换为字典

            if car['illegal']:  # 检查是否违规
                updated = False

                for i, existing_str in enumerate(self.illegal_car_status):
                    existing = parse_str2dir(existing_str)  # 将字符串转换为字典
                    if existing['id'] == car['id'] or (existing['licence'] == car['licence'] and car[
                        'licence'] != ''):  # 查找是否已存在于illegal_car_status中
                        # 如果已存在，则更新违规信息
                        illegal_behaviors_str = ", ".join(car['illegal_behavior'])
                        updated_str = f"id: {car['id']}, licence: {car['licence']}, illegal_behavior: {illegal_behaviors_str}"
                        self.illegal_car_status[i] = updated_str  # 更新列表中的字符串
                        updated = True
                        break
                if not updated:
                    # 如果不存在，则添加新的违规车辆信息
                    illegal_behaviors_str = ", ".join(car['illegal_behavior'])
                    new_str = f"id: {car['id']}, licence: {car['licence']}, illegal_behavior: {illegal_behaviors_str}"
                    self.illegal_car_status.append(new_str)

    def loadmodel(self, weights):
        model = DetectMultiBackend(Path(weights), device=self.device, dnn=self.dnn, fp16=self.half)
        return model



    def dateset_Loader(self):

        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'https://', 'https://'))
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

    def post_car_status(self):
        if self.car_status:
            for car_status in self.car_status:
                car_status_dict = parse_str2dir(car_status)
                send_json_msg({
                    "action": "new_vehicle_update",
                    "data": {'id': car_status_dict['id'], "licence": car_status_dict["licence"],
                             "illegal": car_status_dict["illegal"],
                             "illegal_behavior": car_status_dict["illegal_behavior"]}
                }, self.client_socket)
                time.sleep(0.0001)

    @smart_inference_mode()
    def run(self):
        new_id_dict, new_id_index = dict(), 1
        self.dataset = self.dateset_Loader()
        t = 0
        while self.is_running:
            global cardiection, CarLicence, label, speed, x1, y1
            video_start_time = datetime.now()
            # 上次记录时间
            last_record_time = video_start_time
            # 记录间隔
            record_interval = timedelta(seconds=15)
            self.license_thread = threading.Thread(
                target=self.license_plate_detection,
                args=(self.license_input_queue, self.license_output_queue))

            self.license_thread.start()

            self.gpu_memory = {}
            self.yolo_model.warmup(imgsz=(1 if self.pt or self.yolo_model.triton else self.bs, 3, *self.imgsz))

            frame_count = 0
            curr_frames, prev_frames = [None] * self.batchsize, [None] * self.batchsize

            for path, im, im0s, vid_cap, s in self.dataset:
                frame_start_time = time.time()


                frame_count += 1

                self.fps = vid_cap.get(cv2.CAP_PROP_FPS)
                current_time = datetime.now()  # 当前帧开始检测的时间
                time_elapsed = current_time - last_record_time

                if time_elapsed >= record_interval:
                    self.vehicle_counts_over_time[current_time.strftime('%H:%M:%S')] = self.vehicle_count
                    self.vehicle_count = 0  # 重置计数
                    self.counted_ids.clear()  # 清空已计数的车辆ID集合
                    last_record_time = current_time

                if frame_count % self.vid_stride == 0:


                    self.car_status.clear()
                    self.class_counts.clear()

                    im = self.preprocess(im)
                    self.image_detect(im)

                    # Process detections
                    # 这里的i就是1 其实不用index 为多视频源保留一个口子
                    for i, det in enumerate(self.pred):  # detections per image
                        if self.webcam:  # bs >= 1
                            p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                        self.im0_c = im0.copy()  # 干净的图

                        p = Path(p)  # to Path
                        self.save_dir.mkdir(parents=True, exist_ok=True)
                        curr_frames[i] = im0
                        imc = im0.copy()
                        annotator = Annotator(im0, line_width=1, example=str(self.names), pil=True, font='simkai.ttf',
                                              font_size=20)

                        # 斑马线检测
                        if not self.zebra_detected:
                            ret, zoo_location = ZebraDetection(im0)
                            if ret:
                                self.zebra_detected = True
                                self.zoo_box = []
                            else:
                                self.zebra_detected = True
                                self.zoo_box = []

                        # annotator.box_label(self.zoo_box,'zebra',(0,255,0))



                        if self.strongsort_cfg.STRONGSORT.ECC:  # camera motion compensation
                            self.strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                        if det is not None and len(det):
                            # im大小变换为原来的大小
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Filter detections with y1 < 340
                            det = det[det[:, 1] >= 350]
                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class

                                # 统计类别计数

                                self.class_counts[self.names[int(c)]] = n.item()

                            xywhs = xyxy2xywh(det[:, 0:4])
                            confs = det[:, 4]
                            clss = det[:, 5]
                            # pass detections to strongsort
                            self.outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                            """
                            outputs 包含这张图片中所有目标的信息  i是视频源索引    
                            """
                            # 红绿灯检测
                            # if self.names[int(cls)] == 'traffic light':  # 假设交通灯的类别标签是'traffic light'
                            # 获取交通灯图像区域 要改的
                            # tl_x1, tl_y1, tl_x2, tl_y2 = map(int, [781, 35, 807, 106])
                            # tl_img = im0s[tl_y1:tl_y2, tl_x1:tl_x2]
                            #
                            # # 检测交通灯颜色
                            # if tl_img.size > 0:  # 确保图像区域有效
                            #     self.traffic_light_color = self.TLL_DET(tl_img)
                            #
                            # else:
                            #     logger.info("Invalid traffic light image region.")

                            if len(self.outputs[i]) > 0:

                                # 单一检测目标处理
                                for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):

                                    # output 每一个跟踪对象的信息
                                    bboxes = output[0:4]

                                    if int(bboxes[1]) >= 350:
                                        id = output[4]
                                        if id in new_id_dict.keys():
                                            id = new_id_dict[id]
                                        else:
                                            new_id_dict[id] = new_id_index
                                            id = new_id_index
                                            new_id_index += 1

                                        cls = output[5]
                                        car_status_dict = {"id": int(id), "licence": '', "speed": '',
                                                           "illegal": False, "illegal_behavior": []}

                                        self.update_red_light_runners(int(id), bboxes, imc)


                                        car_center = self.get_centers(bboxes)

                                        size = self.get_size_from_bbox(bboxes)

                                        self.update_location_history(id, frame_count, car_center, size)

                                        if id not in self.counted_ids:
                                            self.vehicle_count += 1
                                            self.counted_ids.add(id)  # 添加ID到已计数集合中

                                        x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
                                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 确保坐标是整数
                                        if x2 > x1 and y2 > y1 and y2 > 430:  # 确保裁剪区域有效车牌检测区
                                            cropped_image = im0s[y1:y2, x1:x2]

                                            if cropped_image.size > 0:  # 检查图像尺寸
                                                self.license_input_queue.put((id, cropped_image))  # 将车牌图像放入车牌识别队列
                                            else:
                                                logger.warning("Invalid cropped image size.")
                                        else:
                                            logger.warning("Invalid crop coordinates.")

                                        # 测速
                                        cardiection = self.calculate_direction_for_car(id)
                                        speed = None
                                        SpeedOverFlag = False
                                        # 超过一定区域后因检测框的浮动会不准
                                        if y2 >= 400:
                                            speed, SpeedOverFlag = calculate_average_speed(id, self.fps,
                                                                                           self.car_locations_history)


                                        if speed != None:
                                            speed = round(speed, 1)
                                            bbox_speed = str(speed) + "km/h"
                                        else:
                                            bbox_speed = ''
                                        car_status_dict['speed'] = bbox_speed
                                        # 车牌
                                        while not self.license_output_queue.empty():
                                            thread_id, license_detection_result = self.license_output_queue.get()
                                            if license_detection_result is not None:
                                                label_licence, label_conf, label_boxes = license_detection_result
                                                # 检查是否已存在该车辆的车牌信息，且新识别的车牌置信度是否更高
                                                if (thread_id not in self.car_licence or label_conf >
                                                    self.car_licence[thread_id][
                                                        'confidence']) and label_licence != '':
                                                    # 更新车牌信息和置信度
                                                    label_licence = list(label_licence)
                                                    label_licence[0] = '京'
                                                    label_licence = ''.join(label_licence)
                                                    self.car_licence[thread_id] = {'licence': label_licence,
                                                                                   'confidence': label_conf}

                                        # 超速处理
                                        if SpeedOverFlag :
                                            car_status_dict['illegal'] = True
                                            car_status_dict['illegal_behavior'].append('超速')
                                            if id is not None:
                                                self.save_violation_snapshot(int(id), self.im0_c, bboxes, f"{speed}",
                                                                             'SpeedOver')

                                        if id in self.car_licence:
                                            car_status_dict['licence'] = self.car_licence[id]['licence']

                                        # 图像检测信息框绘制

                                        if self.save_vid or self.show_vid :  # Add bbox to image

                                            c = int(cls)  # integer class
                                            id = int(id)  # integer id

                                            label = None if self.hide_labels else (
                                                f' {id} {self.names[c]}')
                                            label += f" {bbox_speed}  "
                                            if id in self.car_licence:
                                                label += f"{self.car_licence[id]['licence']}"


                                            annotator.box_label(bboxes, label, color=colors(c, True))

                                        # 将车辆状态信息放到一个汇总列表里 以字符串形式
                                        chuanghongdeng_car = self.check_red_light_violations()
                                        if len(chuanghongdeng_car):
                                            for index in chuanghongdeng_car:
                                                if index == int(id):
                                                    car_status_dict['illegal'] = True
                                                    car_status_dict['illegal_behavior'].append('闯红灯')
                                        self.update_or_add(car_status_dict)
                                        self.update_illegal_car_status()




                        else:
                            self.strongsort_list[i].increment_ages()
                            logger.warning('No detections')

                        # 更新上一帧图片
                        prev_frames[i] = curr_frames[i]
                        im0 = annotator.result()

                        # 创建一个可写的副本 显示车辆信息和统计信息 但最终不需要这部分
                        im0_writable = im0.copy()
                        # 最终显示的图片
                        self.im0_s = im0_writable.copy()
                        # 显存占用统计
                        pynvml.nvmlInit()
                        handles = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handles)
                        self.gpu_memory[(current_time - video_start_time).seconds] = meminfo.used / 1024 ** 3
                        logger.info(f"\n显存占用:{meminfo.used / 1024 ** 3}g\n")  # 已用显存大小
                        frame_end_time = time.time()
                        # print(f"Processing time for {frame_count}: {frame_end_time - frame_start_time:.2f} seconds")
                        t += float(f'{frame_end_time - frame_start_time:.2f}')

                # 显示
                if self.im0_s is None:
                    self.im0_s = im0s.copy()
                cv2.imshow('Detection', self.im0_s)
                cv2.imwrite('s.jpg',self.im0_c)
                cv2.setMouseCallback('Detection', click_event, self.im0_s)  # 设置鼠标回调
                if cv2.waitKey(1) == ord('q'):  # 按q退出
                    self.display_thread_running = False
                    break

                # if not self.image_display_queue.full():
                #     # 如果队列未满，直接放入新帧
                #     self.image_display_queue.put_nowait(self.im0_s)
                # else:
                #     # 如果队列已满，先出队一个元素，再入队新帧
                #     # 这一步确保队列中总是有最新的帧，避免重复显示同一帧
                #     self.image_display_queue.get_nowait()  # 出队一个旧帧
                #     self.image_display_queue.put_nowait(self.im0_s)  # 入队一个新帧
                # self.image_display()

                # 视频保存
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
            self.is_running = False

        # 循环结束后，确保释放资源
        if self.vid_writer:
            self.vid_writer.release()
        self.license_input_queue.put(None)  # 发送停止信号

        self.license_thread.join()
        cv2.destroyAllWindows()

        # 统计
        print(f'haoshi:{t}ms')
        self.vehicle_counts_over_time[datetime.now().strftime('%H:%M:%S')] = self.vehicle_count
        self.plot_counts(self.vehicle_counts_over_time,
                         (str(self.save_dir) + '/' + f"traffic car count {datetime.now().strftime('%Y-%m-%d')}"),
                         'vehicle',
                         'Time')
        self.plot_counts(self.gpu_memory, (str(self.save_dir) + '/' + 'gpu memory count'), 'gpu memory', 'time')
        if self.save_vid:
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def start_detection(self):
        # 在独立线程中启动检测过程
        self.is_running = True
        self.detection_thread = threading.Thread(target=self.run)
        self.detection_thread.start()

    def stop_detection(self):
        with self.lock:
            self.is_running = False
        self.detection_thread.join()

    def is_detection_running(self):
        # 提供一个方法来检查检测是否正在运行
        return self.is_running


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='VID20240425144841.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--augment', action='store_true', help='')
    parser.add_argument('--classes', default=[0,3],nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='output/track', help='保存路径')
    parser.add_argument('--exist-ok', default=True, action='store_true',
                        help='保存路径覆盖')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--half', action='store_true', help='半精度')
    parser.add_argument('--dnn', action='store_true', help='onnx模型')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if only one size
    opt.save_dir = increment_path(Path(opt.project), exist_ok=opt.exist_ok)  # increment run output/track
    clear_directory(opt.save_dir)
    return opt


def main(opt):
    logger.info(f"Initializing VideoTracker with options: {opt}")
    video_tracker = VideoTracker(opt)
    logger.info('Init successfully')
    # video_tracker.change_video("chuanghongdeng.mp4")
    video_tracker.start_detection()  # 开始检测

    try:
        while video_tracker.is_running:
            # 每隔一定时间获取并打印实时数据
            car_status = video_tracker.get_car_status()
            class_counts = video_tracker.get_class_counts()
            vehicle_count = video_tracker.get_vehicle_count()
            illegal_car = video_tracker.get_illegal_car()
            time.sleep(1)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
