import cv2


def draw_text_on_frame(frame, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=1, font_color=(255, 255, 255), thickness=2, background_color=None):
    """
    在图像帧上绘制文本。

    参数:
    - frame: 图像帧，即在其上绘制文本的图像。
    - text: 要绘制的文本字符串。
    - position: 文本在图像上的位置，以像素为单位。（默认左上角）
    - font: OpenCV中的字体类型。
    - font_scale: 字体比例因子，控制字体的大小。
    - font_color: 字体颜色，格式为(B, G, R)。
    - thickness: 文本的线条厚度。
    - background_color: 文本背景颜色，如果设置，则在文本周围绘制一个背景框以提高可读性。
    """
    if background_color is not None:
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, position, (position[0] + text_width, position[1] - text_height - 10), background_color, -1)

    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


# def draw_class_counts(im0, counts):
#     """在帧上绘制每个类别的计数"""
#     position = (10, 70)  # 初始位置
#     font_scale = 1
#     color = (255, 0, 0)  # 蓝色
#     thickness = 1
#     line_type = cv2.LINE_AA
#     for cls, count in counts.items():
#         text = f"{cls}: {count}"
#         # 后期直接把text显示在ui上就好
#         cv2.putText(im0, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type)
#         position = (position[0], position[1] + 20)  # 更新位置，向下移动
def draw_class_counts(im0, counts):
    """在帧上绘制每个类别的计数，带有黑色背景"""
    position = (10, 70)  # 初始位置
    font_scale = 1
    color = (255, 255, 255)  # 白色
    background_color = (0, 0, 0)  # 黑色背景
    thickness = 1
    line_type = cv2.LINE_AA
    font = cv2.FONT_HERSHEY_SIMPLEX

    for cls, count in counts.items():
        text = f"{cls}: {count}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        # 绘制背景矩形
        background_top_left = (position[0], position[1] - text_height - 5)
        background_bottom_right = (position[0] + text_width, position[1] + 5)
        cv2.rectangle(im0, background_top_left, background_bottom_right, background_color, cv2.FILLED)
        # 绘制文本
        cv2.putText(im0, text, position, font, font_scale, color, thickness, line_type)
        position = (position[0], position[1] + text_height + 10)  # 更新位置，向下移动


if __name__ == "__main__":
    # 假设`frame`是您正在处理的当前图像帧
    frame = cv2.imread("yolov7/dataset_car/test/images/MVI_20011__img00045.jpg")  # 仅作为示例，实际情况下您会在循环中处理每一帧

    # 要显示的文本信息
    text_info = "FPS: 30"

    # 在图像的左上角绘制文本
    draw_text_on_frame(frame, text_info, position=(10, 30), font_scale=1, font_color=(0, 255, 0), thickness=2,
                       )

    # 显示图像帧以验证文本已正确绘制
    cv2.imshow("Frame with Text", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
