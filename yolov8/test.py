import cv2

import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

def draw_boxes(img, boxes,model):
    """
    绘制检测框和标签。
    """
    names = model.names  # 假设模型的类别名称可以通过这种方式获取
    for obj in boxes:
        left, top, right, bottom, confidence, label = obj
        color = random_color(int(label))
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), color=color, thickness=2, lineType=cv2.LINE_AA)
        caption = f"{names[int(label)]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        cv2.rectangle(img, (int(left) - 3, int(top) - 33), (int(left) + w + 10, int(top)), color, -1)
        cv2.putText(img, caption, (int(left), int(top) - 5), 0, 1, (0, 0, 0), 2, 16)
    return img

def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32)  # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image


def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)

    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)

    img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM


def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    cross = max((right - left), 0) * max((bottom - top), 0)
    union = area_box(box1) + area_box(box2) - cross
    if cross == 0 or union == 0:
        return 0
    return cross / union


def NMS(boxes, iou_thres):
    remove_flags = [False] * len(boxes)

    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue

        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue

            jbox = boxes[j]
            if (ibox[5] != jbox[5]):
                continue
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes


def postprocess(pred, IM=[], conf_thres=0.25, iou_thres=0.45):
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        label = item[4:].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left = cx - w * 0.5
        top = cy - h * 0.5
        right = cx + w * 0.5
        bottom = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label])

    # 检查boxes是否为空
    if not boxes:
        return []

    boxes = np.array(boxes)

    # 应用逆变换矩阵IM
    lr = boxes[:, [0, 2]]
    tb = boxes[:, [1, 3]]
    boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]

    boxes = sorted(boxes.tolist(), key=lambda x: x[4], reverse=True)

    # 应用NMS
    return NMS(boxes, iou_thres)


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

def main(video,image,path,model):
    if video:
        detect_video(path,model)
    if image:
        img = detect_image(cv2.imread(path),model)
        cv2.imwrite("infer.jpg", img)
        print("save done")


def detect_video(path, model):
    # 打开视频文件
    video_path = path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    frame_count = 0  # 帧计数器
    last_boxes = []  # 存储上一次检测的结果

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 只有当frame_count为10的倍数时，才执行检测
        if frame_count % 1 == 0:
            frame, last_boxes = detect_image(frame, model, return_boxes=True)
        else:
            # 对未进行检测的帧，使用上一次的检测结果
            frame = draw_boxes(frame, last_boxes,model)  # 使用一个新函数draw_boxes来绘制框和标签

        out.write(frame)

        # 显示预测结果
        cv2.imshow("Predictions", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1  # 更新帧计数器

    print('Video processing completed. Video saved as output.mp4.')
    # 释放读取和写入对象
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def detect_image(img,model,return_boxes=False):
    img_pre, IM = preprocess_warpAffine(img)
    img_pre = img_pre.to(device)
    result = model(img_pre)[0].transpose(-1, -2)  # 1,8400,84
    boxes = postprocess(result, IM)

    # 绘制预测结果
    img = draw_boxes(img,boxes,model)
    if return_boxes:
        return img, boxes
    else:

        return img







if __name__ == "__main__":

    model = AutoBackend(weights="runs/detect/train/weights/best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    main(True,False,r"50fe93af9f87621dd9d99c85afe6b709.mp4",model)

















