import cv2
import torch

from ccpd.data2coco import random_color
from ultralytics.utils.torch_utils import select_device

from ultralytics import YOLO


# def hsv2bgr(h, s, v):
#     h_i = int(h * 6)
#     f = h * 6 - h_i
#     p = v * (1 - s)
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
#
#     r, g, b = 0, 0, 0
#
#     if h_i == 0:
#         r, g, b = v, t, p
#     elif h_i == 1:
#         r, g, b = q, v, p
#     elif h_i == 2:
#         r, g, b = p, v, t
#     elif h_i == 3:
#         r, g, b = p, q, v
#     elif h_i == 4:
#         r, g, b = t, p, v
#     elif h_i == 5:
#         r, g, b = v, p, q
#
#     return int(b * 255), int(g * 255), int(r * 255)
#
#
# def random_color(id):
#     h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
#     s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
#     return hsv2bgr(h_plane, s_plane, 1)


def predict(model_path, img):
    model = YOLO(model_path)
    device = select_device('')
    # metrics = model.val(batch=1)  # evaluate model performance on the validation set

    results = model.predict(img,device=device)[0]
    names = results.names
    boxes = results.boxes.data.tolist()

    # for obj in boxes:
    #     x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
    #     confidence = obj[4]
    #     label = int(obj[5])
    #     cropped_image = img[y1:y2, x1:x2]
    #     cv2.imwrite(f'licence_pci]/label/{x1}.jpg',cropped_image)


    # w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
    # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    #
    # cropped_image = img[top:bottom,left:right]
    # print(caption)

    return boxes, names


if __name__ == '__main__':
    model_path = 'ultralytics/runs/detect/train8/weights/best.pt'
    img = cv2.imread(
        r'img_3.png')
    BOXES=predict(model_path, img)
    print(BOXES)
