import argparse

import os

from ultralytics.utils.plotting import save_one_box

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
from trafficLine import *
from draw import draw_text_on_frame, draw_class_counts

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


# Assuming the functions from your provided code snippets are correctly defined and imported here

class VideoTrack:
    def __init__(self, opt):

        self.webcam = None
        self.device = select_device(opt.device)
        self.half = opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        self.save_dir = opt.save_dir
        self.yolo_weights = opt.yolo_weights

        # Load YOLO model
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.yolo_model = attempt_load(Path(opt.yolo_weights), map_location=self.device)
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

        # 未实现
        self.license_id = {}  # 预想存放id：车牌
        """
        用outputs[i]索引分类和id 如果cls == car 将其以str append到每一个车的状态信息列表里 再加速度和方向信息  通过license_id 索引其车牌信息
        储存格式应该是 ['id: license: dir: speed: illegal: '] illegal 显示其违法信息  如果涉及违法 则在ui中标红
        在一开始 illegal 默认为 None 
        先在run里定义一个car_status_dir key：id,license,dir,speed,illegal
        ilegal 默认为 None
        if SpeedOverFlag
        dir['illegal'] = SpeedOver 
        warning()
            ui 车辆信息标红
            并对车辆信息（车牌号 违法信息）保存在excel里 截图：savecrops
            
        每一帧图片都在不断更新这个dir
        每更新一个dir
        就以str格式 append到car_status_list上 
        当这一帧所有list都更新完毕后    清空之前的ui显示  加入新ui显示
        """
        self.car_status = []  # 存放每辆车的状态 以id作为第一维索引

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

        self.overspeedcar = []  # 预期存放格式 ['id:  license:  ']
        self.class_counts = {}

        self.visualize = opt.visualize

        # Placeholder for FPS calculation
        self.fps = 0

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

        global cardiection
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * self.nr_sources, [None] * self.nr_sources
        for frame_idx, (path, im, im0s, vid_cap) in enumerate(self.dataset):
            self.overspeedcar.clear()  # 每一帧开始前清空列表
            self.class_counts.clear()

            s = ''
            t1 = time_synchronized()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_synchronized()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(self.save_dir / Path(path[0]).stem, mkdir=True) if self.visualize else False
            pred = self.yolo_model(im)
            t3 = time_synchronized()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred[0], self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms)
            dt[2] += time_synchronized() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                seen += 1
                if self.webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), self.dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
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
                ret, location = detectTrafficLine(im0)

                txt_path = str(self.save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if self.save_crop else im0  # for save_crop

                if self.strongsort_cfg.STRONGSORT.ECC:  # camera motion compensation
                    self.strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # 统计类别计数
                        self.class_counts[self.names[int(c)]] = n
                    draw_class_counts(im0, self.class_counts)

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to strongsort
                    t4 = time_synchronized()
                    self.outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_synchronized()
                    dt[3] += t5 - t4


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
                        for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):



                            # output 每一个跟踪对象的信息
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)

                            if len(self.outputs_prev) == 2 and int(cls) in [2, 5, 7]:
                                cardiection = calculateDirection(self.outputs_prev[-2], output, id)

                                if cardiection == 'LEFT' or cardiection == 'RIGHT':
                                    bbox_speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id, fps,
                                                                                1)
                                else:
                                    bbox_speed, SpeedOverFlag = Estimated_speed(self.outputs_prev[-2], output, id, fps)

                            if SpeedOverFlag and c in [2, 5, 7]:

                                # self.save_crop = True
                                SpeedOver = True

                                #str(dic['id'])
                                self.overspeedcar.append(str(int(id)))

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
                                    label += f' {bbox_speed} {cardiection}'
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

                            if ret:
                                # 如果检测到斑马线，则在im0上绘制检测框
                                # 假设location是一个包含两个坐标点的元组，表示检测框的对角线上的两个点
                                cv2.rectangle(im0, location[0], location[1], (0, 255, 0), 2)
                                print(location)


                            draw_text_on_frame(im0, f"overspeed:{' '.join(self.overspeedcar)}",background_color=(0, 0, 0))

                    print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

                else:
                    self.strongsort_list[i].increment_ages()
                    print('No detections')

                # Stream results
                if self.show_vid:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                if self.save_vid:
                    if self.vid_path[i] != save_path:  # new video
                        self.vid_path[i] = save_path
                        if isinstance(self.vid_writer[i], cv2.VideoWriter):
                            self.vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    self.vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, self.imgsz, self.imgsz)}' % t)
        if self.save_txt or self.save_vid:
            s = f"\n{len(list(self.save_dir.glob('tracks/*.txt')))} tracks saved to {self.save_dir / 'tracks'}" if self.save_txt else ''
            print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        print(self.overspeedcar)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7-tiny.pt',
                        help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='video-02.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=False, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')

    parser.add_argument('--classes', default=[0,1,2,3,5,7,9], nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'output/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=False, action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')

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
