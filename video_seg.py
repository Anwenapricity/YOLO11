from ultralytics import YOLO

# 分割模型
model = YOLO("yolo11m-seg.pt")

video_path = "694755.mp4"
out_dir = "./output/VIS"

results = model.predict(
    source=video_path,
    save=True,
    save_dir=out_dir,
    save_txt=True,
    retina_masks=True,  # 输出单通道实例掩码
    overlap_mask=True,
    exist_ok=True,
    conf=0.3,
    device=0,
    vid_stride=1
)