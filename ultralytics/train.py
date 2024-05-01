from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'cfg/models/v8/yolov8n.yaml')  # 从头开始构建新模型


    # Use the model
    model.train(data=r"D:\traffic_detection\ccpd\ccpd.yaml", epochs=10, batch=32,optimizer='Adam')  # 训练模型
