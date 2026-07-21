<div align="center">

# 🎯 YOLO11 目标检测项目

<p align="center">
  <img src="https://img.shields.io/badge/YOLO11-🚀-FF6F00?style=for-the-badge&logo=ultralytics" alt="YOLO11">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-green?style=for-the-badge" alt="License">
</p>

**🤖 基于 Ultralytics YOLO11 的计算机视觉目标检测解决方案**

<p align="center">
  <img src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" width="600" alt="YOLO Detection Demo">
</p>

</div>

---

## ✨ 项目简介

本项目基于 **Ultralytics YOLO11** 🚀 构建，提供快速、准确且易于使用的目标检测能力。支持多种视觉任务：

| 任务类型 | 描述 | 模型示例 |
|---------|------|---------|
| 🔍 目标检测 | 检测图像中的物体边界框 | `yolo11n.pt` |
| 🎨 实例分割 | 精确分割每个物体的轮廓 | `yolo11n-seg.pt` |
| 🏷️ 图像分类 | 对整张图片进行分类 | `yolo11n-cls.pt` |
| 🏃 姿态估计 | 检测人体关键点 | `yolo11n-pose.pt` |
| 📐 旋转框检测 | 检测带角度的物体 | `yolo11n-obb.pt` |

---

## 🚀 快速开始

### 📦 安装依赖

```bash
pip install ultralytics
```

> 💡 需要 Python ≥ 3.8，PyTorch ≥ 1.8

### 🔥 一行命令检测

```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

### 🐍 Python 代码示例

```python
from ultralytics import YOLO

# 📥 加载预训练模型
model = YOLO("yolo11n.pt")

# 🖼️ 对单张图片进行推理
results = model("path/to/your/image.jpg")
results[0].show()  # 🎬 显示检测结果

# 🎥 对视频进行推理
results = model("path/to/your/video.mp4", save=True)

# 🏋️ 在自定义数据集上训练
model.train(data="custom_data.yaml", epochs=100, imgsz=640)

# 📊 模型评估
metrics = model.val()

# 📤 导出为 ONNX 格式
model.export(format="onnx")
```

---

## 📊 模型性能对比

| 模型 | 大小 | mAP@50-95 | 参数量 | FLOPs |
|------|------|-----------|--------|-------|
| ⚡ YOLO11n | 640 | 39.5 | 2.6M | 6.5B |
| 🔥 YOLO11s | 640 | 47.0 | 9.4M | 21.5B |
| 🚀 YOLO11m | 640 | 51.5 | 20.1M | 68.0B |
| 💪 YOLO11l | 640 | 53.4 | 25.3M | 86.9B |
| 🏆 YOLO11x | 640 | 54.7 | 56.9M | 194.9B |

> 📌 所有模型首次使用时会自动下载

---

## 🗂️ 项目结构

```
YOLO11/
├── 📁 data/              # 数据集配置
├── 📁 models/            # 模型定义文件
├── 📁 runs/              # 训练/检测结果
├── 📁 weights/           # 预训练权重
├── 📄 custom_data.yaml   # 自定义数据集配置
└── 📄 README.md          # 本文件
```

---

## 🎯 使用场景

- 🚗 **自动驾驶** — 车辆、行人、交通标志检测
- 🏭 **工业质检** — 产品缺陷自动识别
- 🏥 **医疗影像** — 病灶区域定位与分割
- 🛡️ **安防监控** — 异常行为实时预警
- 🌾 **智慧农业** — 作物病虫害识别

---

## 🔧 高级功能

| 功能 | 命令/代码 |
|------|----------|
| 🎬 视频检测 | `yolo predict model=yolo11n.pt source=video.mp4` |
| 📹 摄像头实时检测 | `yolo predict model=yolo11n.pt source=0` |
| 🏋️ 自定义训练 | `yolo train model=yolo11n.pt data=custom.yaml epochs=100` |
| 📊 模型评估 | `yolo val model=yolo11n.pt data=coco.yaml` |
| 📤 模型导出 | `yolo export model=yolo11n.pt format=onnx` |
| 🔍 跟踪 | `yolo track model=yolo11n.pt source=video.mp4` |

---

## 📝 自定义数据集

1. 📂 准备图片和标注文件（YOLO 格式）
2. ✏️ 创建 `data.yaml` 配置文件：

```yaml
path: ./dataset          # 📁 数据集根目录
train: images/train      # 🏋️ 训练图片路径
val: images/val          # 📊 验证图片路径
nc: 3                    # 🔢 类别数量
names: ['cat', 'dog', 'bird']  # 🏷️ 类别名称
```

3. 🚀 开始训练：

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！让我们一起让这个项目变得更好 💪

1. 🍴 Fork 本仓库
2. 🌿 创建新分支：`git checkout -b feature/amazing-feature`
3. ✏️ 提交更改：`git commit -m 'Add amazing feature'`
4. 📤 推送分支：`git push origin feature/amazing-feature`
5. 🔀 提交 Pull Request

---

## 📜 许可证

本项目采用 [AGPL-3.0](LICENSE) 开源协议。

---

## 🙏 致谢

- 🌟 [Ultralytics](https://github.com/ultralytics/ultralytics) — 提供强大的 YOLO 框架
- 🔥 [PyTorch](https://pytorch.org/) — 深度学习框架
- 🐍 [Python](https://www.python.org/) — 编程语言

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！**

<p align="center">
  <img src="https://img.shields.io/github/stars/Anwenapricity/YOLO11?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/Anwenapricity/YOLO11?style=social" alt="Forks">
</p>

Made with ❤️ by <a href="https://github.com/Anwenapricity">@Anwenapricity</a>

</div>
