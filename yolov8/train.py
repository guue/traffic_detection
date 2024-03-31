from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）

    # Use the model
    model.train(data=r"D:\traffic_detection\ccpd\ccpd.yaml", epochs=50, batch=16,resume=True)  # 训练模型
