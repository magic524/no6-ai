#!/bin/bash
# TGADetect — VisDrone n/s/m/l/x 串行
# Cloud1 RTX 3090, device=0

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0

echo "[1/5] yolo11n-TGADetect — batch 16"
$YOLO detect train data=$DATA model=yolo11n-TGADetect.yaml batch=16 device=$DEVICE name=TGADetect/VisDrone/yolo11n-TGADetect
echo "yolo11n DONE at $(date)"

echo "[2/5] yolo11s-TGADetect — batch 16"
$YOLO detect train data=$DATA model=yolo11s-TGADetect.yaml batch=16 device=$DEVICE name=TGADetect/VisDrone/yolo11s-TGADetect
echo "yolo11s DONE at $(date)"

echo "[3/5] yolo11m-TGADetect — batch 8"
$YOLO detect train data=$DATA model=yolo11m-TGADetect.yaml batch=8 device=$DEVICE name=TGADetect/VisDrone/yolo11m-TGADetect
echo "yolo11m DONE at $(date)"

echo "[4/5] yolo11l-TGADetect — batch 8"
$YOLO detect train data=$DATA model=yolo11l-TGADetect.yaml batch=8 device=$DEVICE name=TGADetect/VisDrone/yolo11l-TGADetect
echo "yolo11l DONE at $(date)"

echo "[5/5] yolo11x-TGADetect — batch 4"
$YOLO detect train data=$DATA model=yolo11x-TGADetect.yaml batch=4 device=$DEVICE name=TGADetect/VisDrone/yolo11x-TGADetect
echo "yolo11x DONE at $(date)"

echo "全部完成！$(date)"