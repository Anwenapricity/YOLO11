import os
import argparse
from pathlib import Path

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


def run_detection(
    source: Path,
    save_dir: Path,
    model_path: str = "yolo11m.pt",
    conf: float = 0.25,
    device: str | int = 0,
) -> None:
    """对单张图片或文件夹内图片顺序执行目标检测"""
    save_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(source)

    model = YOLO(model_path)
    total = len(images)

    print(f"model: {model_path}")
    print(f"input: {source} (total: {total})")
    print(f"output: {save_dir}")
    print(f"confidence threshold: {conf} | decice: {device}")
    print("=" * 60)

    for idx, img_path in enumerate(images, start=1):
        print(f"[{idx}/{total}] processing: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            save=True,
            save_dir=str(save_dir),
            save_txt=True,
            exist_ok=True,
            conf=conf,
            device=device,
            verbose=False,
        )
        num_boxes = len(results[0].boxes) if results and results[0].boxes is not None else 0

    print("=" * 60)
    print("Down!")
    print(f"images saved to: {save_dir}")
    print(f"labels saved to: {save_dir / 'labels'}")


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