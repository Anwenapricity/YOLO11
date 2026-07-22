set -euo pipefail
CONF=0.30
DEVICE=0

# ========== 图像目标检测测试 ==========
DATASET=dataset
DATASETNAME=cat
SAVE=OD
MODEL=yolo11m.pt
python img_od.py                        \
    --dataset $DATASET/$DATASETNAME/    \
    --savedir output/$SAVE/             \
    --model $MODEL                      \
    --conf $CONF                        \
    --device $DEVICE                    \
    >> logs/IOD_$DATASETNAME.txt



# ========== 图像实例分割测试 ==========
# python img_seg.py

# ========== 视频目标检测测试 ==========
# python video_od.py

# ========== 视频实例分割测试 ==========
# python video_seg.py

# ========== 训练测试 ==========
# python train.py
