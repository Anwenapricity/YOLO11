from ultralytics import YOLO

# 1. 加载yolov8s模型配置+预训练权重
# model = YOLO("yolov8s.yaml")
model.load("yolov11m.pt")

# 2. 训练，coco128是官方自带小数据集，自动下载
model.train(
    data="coco128.yaml",
    epochs=10,        # 只训10轮快速验证
    imgsz=640,
    batch=8,
    device=0          # 有GPU写0，CPU运行写cpu
)