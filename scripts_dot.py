

import cv2

# 载入图像
image_path = r'img_1.png'  # 替换为你的图片路径
img = cv2.imread(image_path)
clone = img.copy()

def click_event(event, x, y, flags, param):
    # 如果是左键点击，则打印坐标并在图像上绘制圆点标记
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        cv2.circle(clone, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("image", clone)

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
