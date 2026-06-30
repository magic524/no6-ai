#!/bin/bash
# Cloud2: WDAM on VisDrone (n/s/m/l/x)
# RTX 3090, device=0
set -euo pipefail
cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DEVICE=0
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "Cloud2: WDAM VisDrone n/s/m/l/x"
echo "Start: $(date)"
echo "=========================================="

# ==================== VisDrone ====================

# n
echo "[1/5] yolo11n-C2PSA_WDAM -- VisDrone -- batch 16"
$YOLO train model=yolo11n-C2PSA_WDAM.yaml data=VisDrone.yaml \
  epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/VisDrone/yolo11n
echo "yolo11n DONE at $(date)" && echo ""

# s
echo "[2/5] yolo11s-C2PSA_WDAM -- VisDrone -- batch 16"
$YOLO train model=yolo11s-C2PSA_WDAM.yaml data=VisDrone.yaml \
  epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/VisDrone/yolo11s
echo "yolo11s DONE at $(date)" && echo ""

# m
echo "[3/5] yolo11m-C2PSA_WDAM -- VisDrone -- batch 8"
$YOLO train model=yolo11m-C2PSA_WDAM.yaml data=VisDrone.yaml \
  epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/VisDrone/yolo11m
echo "yolo11m DONE at $(date)" && echo ""

# l
echo "[4/5] yolo11l-C2PSA_WDAM -- VisDrone -- batch 8"
$YOLO train model=yolo11l-C2PSA_WDAM.yaml data=VisDrone.yaml \
  epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/VisDrone/yolo11l
echo "yolo11l DONE at $(date)" && echo ""

# x
echo "[5/5] yolo11x-C2PSA_WDAM -- VisDrone -- batch 4"
$YOLO train model=yolo11x-C2PSA_WDAM.yaml data=VisDrone.yaml \
  epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/VisDrone/yolo11x
echo "yolo11x DONE at $(date)" && echo ""

echo "=========================================="
echo "Cloud2: ALL WDAM RUNS COMPLETE"
echo "End: $(date)"
echo "=========================================="