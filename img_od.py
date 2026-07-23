import os
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.disabled = True

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]

    return sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_detection_labels(result, label_path: Path) -> None:
    """
    保存 YOLO 格式检测标签：
    class_id x_center y_center width height confidence
    坐标均为归一化坐标。
    """

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        if result.boxes is None:
            return

        boxes = result.boxes

        xywhn = boxes.xywhn.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(xywhn, cls, conf):

            x_center, y_center, width, height = box

            f.write(
                f"{int(class_id)} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f} "
                f"{confidence:.6f}\n"
            )


def draw_detections(image, result):
    """
    在图像上绘制检测结果。
    """
    if result.boxes is None:
        return image

    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    names = result.names

    for box, class_id, confidence in zip(xyxy, cls, conf):

        x1, y1, x2, y2 = map(int, box)

        class_id = int(class_id)
        class_name = names[class_id]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


def run_detection(
    source: Path,
    save_dir: Path,
    model_path: str = "yolo11m.pt",
    conf: float = 0.25,
    device: str | int = 0,
) -> None:

    image_save_dir = save_dir / "images"
    label_save_dir = save_dir / "labels"

    image_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)

    model = YOLO(model_path)
    total = len(images)

    print(f"model: {model_path}")
    print(f"input: {source} (total: {total})")
    print(f"images output: {image_save_dir}")
    print(f"labels output: {label_save_dir}")
    print(f"confidence threshold: {conf} | device: {device}")
    print("=" * 60)

    for idx, img_path in enumerate(images, start=1):

        print(f"[{idx}/{total}] processing: {img_path.name}")

        # 读取原始图像
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Warning: failed to read {img_path}")
            continue

        # 只进行推理，不使用 YOLO 自动保存
        results = model.predict(
            source=image,
            conf=conf,
            device=device,
            verbose=False,
            save=False,
            save_txt=False,
        )

        result = results[0]

        num_boxes = (len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # =========================
        # 1. 保存检测标签
        # =========================

        label_path = label_save_dir / f"{img_path.stem}.txt"
        save_detection_labels(result, label_path)

        # =========================
        # 2. 绘制检测框
        # =========================

        result_image = image.copy()

        result_image = draw_detections(result_image, result)

        # =========================
        # 3. 自定义保存图像
        # =========================

        image_save_path = (image_save_dir / img_path.name)

        cv2.imwrite(str(image_save_path), result_image)

        print(f"  detections: {num_boxes}")

    print("=" * 60)
    print("Done!")
    print(f"images saved to: {image_save_dir}")
    print(f"labels saved to: {label_save_dir}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="YOLO11 图像目标检测")
    parser.add_argument("--dataset", dest="source", type=str, default="./dataset", help="输入图片路径或文件夹路径")
    parser.add_argument("--savedir", type=str, default="./output/OD", help="检测结果输出目录")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", type=str, default="0", help="推理设备，如 0、cuda:0 或 cpu")
    return parser.parse_args()


def parse_device(device: str) -> str | int:
    if device.isdigit():
        return int(device)
    return device


if __name__ == "__main__":
    args = parse_args()
    run_detection(
        source=Path(args.source),
        save_dir=Path(args.savedir),
        model_path=args.model,
        conf=args.conf,
        device=parse_device(args.device),
    )