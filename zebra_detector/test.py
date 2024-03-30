import os
import cv2

import argparse

import torch
from zebra_detector.model import detector
from zebra_detector.utils import utils

def detect_zoo(img):
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='zebra_detector/data/coco.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='zebra_detector/weights/coco-190-epoch-0.481495ap-model.pth',
                        help='The path of the .pth model to be transformed')


    opt = parser.parse_args()
    cfg = utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"


    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    #sets the module in eval node
    model.eval()
    
    #数据预处理
    ori_img = img
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0,3, 1, 2))
    img = img.to(device).float() / 255.0

    #模型推理

    preds = model(img)



    #特征图后处理
    output = utils.handel_preds(preds, cfg, device)
    output_boxes = utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]
    return output_boxes,scale_h,scale_w






    

