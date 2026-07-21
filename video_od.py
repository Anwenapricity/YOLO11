from ultralytics import YOLO
import os
import shutil

# =============================================
VIDEO_PATH = "694755.mp4"    # 视频路径
MODEL_WEIGHT = "yolo11m.pt"  # 模型权重
CONF_THRESH = 0.3           # 置信度过滤阈值
ROOT_OUTPUT = "./output/VOD"     # 输出根目录
# =============================================

# 定义子文件夹
img_output = os.path.join(ROOT_OUTPUT, "images")
label_output = os.path.join(ROOT_OUTPUT, "labels")

# 新建文件夹
os.makedirs(img_output, exist_ok=True)
os.makedirs(label_output, exist_ok=True)

# 加载模型
model = YOLO(MODEL_WEIGHT)

print(f"开始推理视频：{VIDEO_PATH}")
results = model.predict(
    source = VIDEO_PATH,
    stream = False,
    conf = CONF_THRESH,
    save = True,          # 生成带检测框的视频
    save_txt = True,      # 每帧保存txt标签
    save_dir = ROOT_OUTPUT,
    exist_ok = True,
    verbose = False,      # 关闭冗余打印
    show = False          # 云服务器必须关闭可视化窗口
)

# 自动把根目录生成的帧图片移动到 output/images
for filename in os.listdir(ROOT_OUTPUT):
    suffix = os.path.splitext(filename)[1].lower()
    if suffix in [".jpg", ".jpeg", ".png"]:
        src = os.path.join(ROOT_OUTPUT, filename)
        dst = os.path.join(img_output, filename)
        shutil.move(src, dst)

print("推理完成！")
print(f"带框视频文件：{ROOT_OUTPUT}")
print(f"逐帧检测图片：{img_output}")
print(f"逐帧标注txt：{label_output}")