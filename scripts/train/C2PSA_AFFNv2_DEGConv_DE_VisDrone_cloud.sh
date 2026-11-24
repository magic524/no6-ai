#!/bin/bash
# C2PSA_AFFNv2_DEGConv_DE — VisDrone n/s/m/l/x 串行
# Cloud1 RTX 3090, device=0

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0

echo "[1/5] yolo11n-C2PSA_AFFNv2_DEGConv_DE — batch 16"
$YOLO detect train data=$DATA model=yolo11n-C2PSA_AFFNv2_DEGConv_DE.yaml batch=16 device=$DEVICE name=C2PSA_AFFNv2_DEGConv_DE/VisDrone/yolo11n-C2PSA_AFFNv2_DEGConv_DE
echo "yolo11n DONE at $(date)"

echo "[2/5] yolo11s-C2PSA_AFFNv2_DEGConv_DE — batch 16"
$YOLO detect train data=$DATA model=yolo11s-C2PSA_AFFNv2_DEGConv_DE.yaml batch=16 device=$DEVICE name=C2PSA_AFFNv2_DEGConv_DE/VisDrone/yolo11s-C2PSA_AFFNv2_DEGConv_DE
echo "yolo11s DONE at $(date)"

echo "[3/5] yolo11m-C2PSA_AFFNv2_DEGConv_DE — batch 8"
$YOLO detect train data=$DATA model=yolo11m-C2PSA_AFFNv2_DEGConv_DE.yaml batch=8 device=$DEVICE name=C2PSA_AFFNv2_DEGConv_DE/VisDrone/yolo11m-C2PSA_AFFNv2_DEGConv_DE
echo "yolo11m DONE at $(date)"

echo "[4/5] yolo11l-C2PSA_AFFNv2_DEGConv_DE — batch 8"
$YOLO detect train data=$DATA model=yolo11l-C2PSA_AFFNv2_DEGConv_DE.yaml batch=8 device=$DEVICE name=C2PSA_AFFNv2_DEGConv_DE/VisDrone/yolo11l-C2PSA_AFFNv2_DEGConv_DE
echo "yolo11l DONE at $(date)"

echo "[5/5] yolo11x-C2PSA_AFFNv2_DEGConv_DE — batch 4"
$YOLO detect train data=$DATA model=yolo11x-C2PSA_AFFNv2_DEGConv_DE.yaml batch=4 device=$DEVICE name=C2PSA_AFFNv2_DEGConv_DE/VisDrone/yolo11x-C2PSA_AFFNv2_DEGConv_DE
echo "yolo11x DONE at $(date)"

echo "全部完成！$(date)"
