import sys



sys.path.append('./LPRNet')


from LPRNet.LPRNet_Test import *

import torch.nn as nn
import torch

import numpy as np
import cv2
from ultralytics.test import predict, random_color

import re

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(), #13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=256+class_num+128+64, out_channels=self.class_num, kernel_size=(1,1), stride=(1,1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                #print("intermediate feature map {} shape is: ".format(i), x.shape)
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            #print("after globel context {} shape is: ".format(i), f.shape)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        #print("after container shape is: ", x.shape)
        logits = torch.mean(x, dim=2)

        return logits





class Licence(object):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo_model = 'weights/licence_yolov8.pt'
        self.STN = STNet()
        self.STN.to(self.device)
        self.STN.load_state_dict(torch.load('weights/stn_93.12_model.pth', map_location=lambda storage, loc: storage))
        self.STN.eval()
        self.lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(torch.load('weights/lprnet_93.12._model.pth', map_location=lambda storage, loc: storage))
        self.lprnet.eval()
        self.CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
        '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
        '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
        '新',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z', 'I', 'O', '-'
        ]



    def convert_image(self,inp):
        # convert a Tensor to numpy image
        inp = inp.squeeze(0).cpu()
        inp = inp.detach().numpy().transpose((1,2,0))
        inp = 127.5 + inp/0.0078125
        inp = inp.astype('uint8') 

        return inp



    def decode(self,preds, CHARS):
        # greedy decode
        pred_labels = list()
        labels = list()
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = list()
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = pred_label[0]
            for c in pred_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            pred_labels.append(no_repeat_blank_label)
            
        for i, label in enumerate(pred_labels):
            lb = ""
            for i in label:
                lb += CHARS[i]
            labels.append(lb)
        
        return labels, np.array(pred_labels)

    def detectLicence(self, img):
        # 检查并转换图像通道数
        if img.shape[2] == 4:  # 如果图像是4通道的
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 将图像从BGRA转换为BGR
        boxes, names = predict(self.yolo_model, img)
        final_labels = []
        final_confidences = []

        for obj in boxes:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            detection_confidence = obj[4]  # YOLO检测置信度

            img_box = img[top:bottom, left:right]
            if img_box is None:
                continue

            try:
                im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
            except:
                continue

            im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
            data = torch.from_numpy(im).float().unsqueeze(0).to(self.device)
            transfer = self.STN(data)

            preds = self.lprnet(transfer)
            preds_softmax = torch.softmax(preds, dim=2)  # 对预测结果应用softmax获取概率分布
            preds_max_conf, preds_index = torch.max(preds_softmax, dim=2)  # 获取每个位置最大概率及其索引
            preds_conf = preds_max_conf.detach().cpu().numpy()[0]  # 转换为numpy数组，获取第一项
            average_pred_conf = np.mean(preds_conf)  # 计算平均预测字符置信度

            # 计算最终置信度：检测置信度与平均预测字符置信度的加权平均
            final_confidence = (detection_confidence + average_pred_conf) / 2

            labels, pred_labels = self.decode(preds.cpu().detach().numpy(), self.CHARS)
            if re.match(r'^[\u4e00-\u9fa5][A-Za-z0-9]{6,7}$', labels[0]) is not None:
                final_labels.append(labels[0])
                final_confidences.append(final_confidence)
            else:
                final_labels.append('')
                final_confidences.append(0)

        # 假设选择置信度最高的车牌返回
        if final_confidences:  # 如果有识别到车牌
            max_conf_index = np.argmax(final_confidences)  # 获取最大置信度索引
            return final_labels[max_conf_index], final_confidences[max_conf_index]

if __name__=="__main__":
    lc=Licence()
    image = cv2.imdecode(np.fromfile(r"D:\traffic_detection\ccpd\images\val\3056141493055555554-88_93-205&455_603&597-603&575_207&597_205&468_595&455-0_0_3_24_32_27_31_33-90-213.jpg", dtype=np.uint8), -1)
    label,conf = lc.detectLicence(image)
    print(label)



            




