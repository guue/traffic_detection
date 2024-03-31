import os
from ultralytics import YOLO

# model = YOLO("yolov8s.yaml")
model = YOLO("../train/weights/best.pt")
# success=model.export(format="onnx")
success = model.export(format="onnx", half=False, dynamic=True)

