#!/bin/bash
# Cloud2: WDAM VisDrone (n/s/m/l/x)
# RTX 3090, device=0
set -e
cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DEVICE=0
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "Cloud2: WDAM VisDrone"
echo "Start: $(date)"
echo "=========================================="

$YOLO detect train data=VisDrone.yaml model=yolo11n-C2PSA_WDAM.yaml epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ name=WDAM/VisDrone/yolo11n

echo "yolo11n DONE at $(date)" && echo ""

$YOLO detect train data=VisDrone.yaml model=yolo11s-C2PSA_WDAM.yaml epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ name=WDAM/VisDrone/yolo11s

echo "yolo11s DONE at $(date)" && echo ""

$YOLO detect train data=VisDrone.yaml model=yolo11m-C2PSA_WDAM.yaml epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ name=WDAM/VisDrone/yolo11m

echo "yolo11m DONE at $(date)" && echo ""

$YOLO detect train data=VisDrone.yaml model=yolo11l-C2PSA_WDAM.yaml epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ name=WDAM/VisDrone/yolo11l

echo "yolo11l DONE at $(date)" && echo ""

$YOLO detect train data=VisDrone.yaml model=yolo11x-C2PSA_WDAM.yaml epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ name=WDAM/VisDrone/yolo11x

echo "yolo11x DONE at $(date)" && echo ""

echo "=========================================="
echo "Cloud2: WDAM VisDrone 全部完成！"
echo "End: $(date)"
echo "=========================================="
