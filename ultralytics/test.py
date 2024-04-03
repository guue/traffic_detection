import cv2
from ultralytics import YOLO


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


def predict(model_path, img):
    model = YOLO(model_path)

    results = model(img)[0]
    names = results.names
    boxes = results.boxes.data.tolist()

    for obj in boxes:
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
        caption = f"{names[label]} {confidence:.2f}"
        # w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)

        cropped_image = img[top:bottom,left:right]
        # print(caption)


    return boxes,names

if __name__=='__main__':
    model_path = 'weights/licence_yolov8.pt'
    img = cv2.imread(r'D:\traffic_detection\ccpd\images\val\3047552083333333335-88_98-138&431_470&575-458&558_138&575_144&432_470&431-0_0_3_24_32_29_30_25-106-198.jpg')
    img = predict(model_path,img)
    cv2.imshow('crop',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



