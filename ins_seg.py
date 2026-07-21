import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import random

# ===================== 配置区 =====================
model_path  = "yolo11m-seg.pt"
img_path    = "VCG211.png"                  # 支持改为文件夹路径批量处理
out_dir     = "./output/IS"
conf_thresh = 0.25
device      = 0
# ==================================================


def get_stem(path: str) -> str:
    """获取文件名（不含扩展名）"""
    return os.path.splitext(os.path.basename(path))[0]


def load_color_table(path: str) -> dict:
    """加载颜色表，不存在则返回空字典"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_color_table(path: str, table: dict):
    """保存颜色表"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)


def ensure_color(table: dict, class_name: str) -> list:
    """若该类别在颜色表中不存在，随机生成并追加；返回 [B, G, R]"""
    if class_name not in table:
        table[class_name] = [random.randint(30, 255) for _ in range(3)]
    return table[class_name]


def draw_mask_on_canvas(mask_bool: np.ndarray, color: list,
                        canvas: np.ndarray = None) -> np.ndarray:
    """将单个掩码以指定颜色绘制到画布上"""
    h, w = mask_bool.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask_bool] = color
    return canvas


# ===================== 主流程 =====================
# 创建输出目录
for sub in ["overlap", "masks", "labels"]:
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

color_table_path = os.path.join(out_dir, "color_table.json")
color_table = load_color_table(color_table_path)

model = YOLO(model_path)

results = model.predict(
    source=img_path,
    save=False,
    retina_masks=True,
    overlap_mask=True,
    conf=conf_thresh,
    device=device
)

for res in results:
    # ---------- 基本信息 ----------
    stem = get_stem(res.path)
    masks = res.masks
    if masks is None:
        print(f"[{stem}] 未检测到目标，跳过")
        continue

    boxes = res.boxes
    num = len(masks)
    h, w = masks.data.shape[1], masks.data.shape[2]
    orig_h, orig_w = res.orig_shape
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    cls_names = [model.names[c] for c in cls_ids]
    print(f"[{stem}] 检测到 {num} 个目标: {cls_names}")

    # ---------- 1. 保存无边框分割叠加图 ----------
    annotated = res.plot(boxes=False, masks=True, labels=True, conf=True)
    overlap_path = os.path.join(out_dir, "overlap", f"{stem}.jpg")
    cv2.imwrite(overlap_path, annotated)
    print(f"  叠加图 -> {overlap_path}")

    # ---------- 2. 保存 YOLO 标签 ----------
    txt_path = os.path.join(out_dir, "labels", f"{stem}.txt")
    with open(txt_path, "w") as f:
        for i in range(num):
            cls_id = cls_ids[i]
            # 检测框归一化坐标（boxes.xyxy 为原图像素坐标）
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cx_n = ((x1 + x2) / 2) / orig_w
            cy_n = ((y1 + y2) / 2) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            # 分割归一化坐标
            seg_points = masks.xyn[i].flatten().tolist()
            seg_str = " ".join(f"{v:.6f}" for v in seg_points)
            f.write(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {bw:.6f} {bh:.6f} {seg_str}\n")
    print(f"  标签   -> {txt_path}")

    # ---------- 3. 掩码目录 ----------
    mask_dir = os.path.join(out_dir, "masks", stem)
    os.makedirs(mask_dir, exist_ok=True)

    # ---------- 4. 保存合并总掩码 mask0 ----------
    merged = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num):
        mask_bool = masks.data[i].cpu().numpy() > 0.5
        color = ensure_color(color_table, cls_names[i])
        merged = draw_mask_on_canvas(mask_bool, color, merged)

    merged_path = os.path.join(mask_dir, f"{stem}_mask0.png")
    cv2.imwrite(merged_path, merged)
    print(f"  总掩码 -> {merged_path}")

    # ---------- 5. 保存单个目标掩码 mask1, mask2, ... ----------
    for i in range(num):
        mask_bool = masks.data[i].cpu().numpy() > 0.5
        color = ensure_color(color_table, cls_names[i])
        single = draw_mask_on_canvas(mask_bool, color)

        single_path = os.path.join(mask_dir, f"{stem}_mask{i + 1}.png")
        cv2.imwrite(single_path, single)
    print(f"  单掩码 -> {mask_dir}/{stem}_mask1~{num}.png")

# ---------- 6. 回写颜色表（含新增类别） ----------
save_color_table(color_table_path, color_table)
print(f"\n颜色表已保存至 {color_table_path}")
print("请根据需要修改颜色表中的 BGR 值后重新运行")