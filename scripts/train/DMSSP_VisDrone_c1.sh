#!/bin/bash
# DMSSP VisDrone — n/s/m/l/x 串行
# Cloud1 RTX 3090, device=0

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0

echo "[1/5] yolo11n-DMSSP — batch 16"
$YOLO detect train data=$DATA model=yolo11n-DMSSP.yaml batch=16 device=$DEVICE name=DMSSP/VisDrone/yolo11n-DMSSP
echo "yolo11n DONE at $(date)"

echo "[2/5] yolo11s-DMSSP — batch 16"
$YOLO detect train data=$DATA model=yolo11s-DMSSP.yaml batch=16 device=$DEVICE name=DMSSP/VisDrone/yolo11s-DMSSP
echo "yolo11s DONE at $(date)"

echo "[3/5] yolo11m-DMSSP — batch 8"
$YOLO detect train data=$DATA model=yolo11m-DMSSP.yaml batch=8 device=$DEVICE name=DMSSP/VisDrone/yolo11m-DMSSP
echo "yolo11m DONE at $(date)"

echo "[4/5] yolo11l-DMSSP — batch 8"
$YOLO detect train data=$DATA model=yolo11l-DMSSP.yaml batch=8 device=$DEVICE name=DMSSP/VisDrone/yolo11l-DMSSP
echo "yolo11l DONE at $(date)"

echo "[5/5] yolo11x-DMSSP — batch 4"
$YOLO detect train data=$DATA model=yolo11x-DMSSP.yaml batch=4 device=$DEVICE name=DMSSP/VisDrone/yolo11x-DMSSP
echo "yolo11x DONE at $(date)"

echo "全部完成！$(date)"