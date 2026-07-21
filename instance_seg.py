import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("yolo11m-seg.pt")

img_dir = "VCG211.png"
out_dir = "./output/IS"

mask_out_dir = os.path.join(out_dir, "masks")
os.makedirs(mask_out_dir, exist_ok=True)

results = model.predict(
    source=img_dir,
    save=False,            # 关闭自动保存，手动控制
    retina_masks=True,
    overlap_mask=True,
    conf=0.25,
    device=0
)

for idx, res in enumerate(results):
    masks = res.masks

    # --- 1. 保存无边框的分割结果图 ---
    annotated = res.plot(boxes=True, conf=True, masks=True, labels=True)
    cv2.imwrite(os.path.join(out_dir, f"img{idx}_seg.jpg"), annotated)

    # --- 2. 保存每张掩码的灰度图 ---
    if masks is not None:
        for i, mask in enumerate(masks):
            mask_np = (mask.data[0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(mask_out_dir, f"img{idx}_mask{i}.png"),
                mask_np
            )
        print(f"img{idx}: 保存了 {len(masks)} 张掩码")