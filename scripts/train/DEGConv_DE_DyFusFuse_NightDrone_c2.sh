#!/bin/bash
# DEGConv_DE_DyFusFuse — NightDrone n/s/m/l/x 串行
# Cloud2 RTX 3090, device=0

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=NightDrone.yaml
DEVICE=0

echo "[1/5] yolo11n-DEGConv_DE_DyFusFuse — batch 16"
$YOLO detect train data=$DATA model=yolo11n-DEGConv_DE_DyFusFuse.yaml batch=16 device=$DEVICE name=DEGConv_DE_DyFusFuse/NightDrone/yolo11n-DEGConv_DE_DyFusFuse
echo "yolo11n DONE at $(date)"

echo "[2/5] yolo11s-DEGConv_DE_DyFusFuse — batch 16"
$YOLO detect train data=$DATA model=yolo11s-DEGConv_DE_DyFusFuse.yaml batch=16 device=$DEVICE name=DEGConv_DE_DyFusFuse/NightDrone/yolo11s-DEGConv_DE_DyFusFuse
echo "yolo11s DONE at $(date)"

echo "[3/5] yolo11m-DEGConv_DE_DyFusFuse — batch 8"
$YOLO detect train data=$DATA model=yolo11m-DEGConv_DE_DyFusFuse.yaml batch=8 device=$DEVICE name=DEGConv_DE_DyFusFuse/NightDrone/yolo11m-DEGConv_DE_DyFusFuse
echo "yolo11m DONE at $(date)"

echo "[4/5] yolo11l-DEGConv_DE_DyFusFuse — batch 8"
$YOLO detect train data=$DATA model=yolo11l-DEGConv_DE_DyFusFuse.yaml batch=8 device=$DEVICE name=DEGConv_DE_DyFusFuse/NightDrone/yolo11l-DEGConv_DE_DyFusFuse
echo "yolo11l DONE at $(date)"

echo "[5/5] yolo11x-DEGConv_DE_DyFusFuse — batch 4"
$YOLO detect train data=$DATA model=yolo11x-DEGConv_DE_DyFusFuse.yaml batch=4 device=$DEVICE name=DEGConv_DE_DyFusFuse/NightDrone/yolo11x-DEGConv_DE_DyFusFuse
echo "yolo11x DONE at $(date)"

echo "全部完成！$(date)"
