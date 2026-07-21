from ultralytics import YOLO

model = YOLO("yolo11m.pt")

img_dir = "test.jpg"
out_dir = "./output/OD"

results = model.predict(
    source = img_dir,
    save = True,
    save_dir = out_dir,
    save_txt = True,
    exist_ok = True,
    conf = 0.25,
    device = 0
    )