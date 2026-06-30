#!/bin/bash
# Cloud1: WDAM on RSOD + NWPU_VHR-10 (n/s/m/l/x)
# RTX 3090, device=0
set -euo pipefail
cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DEVICE=0
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "Cloud1: WDAM RSOD + NWPU_VHR-10"
echo "Start: $(date)"
echo "=========================================="

# Batch: n/s=16, m/l=8, x=4

# ==================== RSOD ====================

# n
echo "[1/10] yolo11n-C2PSA_WDAM -- RSOD -- batch 16"
$YOLO train model=yolo11n-C2PSA_WDAM.yaml data=RSOD.yaml \
  epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/RSOD/yolo11n
echo "yolo11n DONE at $(date)" && echo ""

# s
echo "[2/10] yolo11s-C2PSA_WDAM -- RSOD -- batch 16"
$YOLO train model=yolo11s-C2PSA_WDAM.yaml data=RSOD.yaml \
  epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/RSOD/yolo11s
echo "yolo11s DONE at $(date)" && echo ""

# m
echo "[3/10] yolo11m-C2PSA_WDAM -- RSOD -- batch 8"
$YOLO train model=yolo11m-C2PSA_WDAM.yaml data=RSOD.yaml \
  epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/RSOD/yolo11m
echo "yolo11m DONE at $(date)" && echo ""

# l
echo "[4/10] yolo11l-C2PSA_WDAM -- RSOD -- batch 8"
$YOLO train model=yolo11l-C2PSA_WDAM.yaml data=RSOD.yaml \
  epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/RSOD/yolo11l
echo "yolo11l DONE at $(date)" && echo ""

# x
echo "[5/10] yolo11x-C2PSA_WDAM -- RSOD -- batch 4"
$YOLO train model=yolo11x-C2PSA_WDAM.yaml data=RSOD.yaml \
  epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/RSOD/yolo11x
echo "yolo11x DONE at $(date)" && echo ""

# ==================== NWPU_VHR-10 ====================

# n
echo "[6/10] yolo11n-C2PSA_WDAM -- NWPU_VHR-10 -- batch 16"
$YOLO train model=yolo11n-C2PSA_WDAM.yaml data=NWPU_VHR-10.yaml \
  epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/NWPU_VHR-10/yolo11n
echo "yolo11n DONE at $(date)" && echo ""

# s
echo "[7/10] yolo11s-C2PSA_WDAM -- NWPU_VHR-10 -- batch 16"
$YOLO train model=yolo11s-C2PSA_WDAM.yaml data=NWPU_VHR-10.yaml \
  epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/NWPU_VHR-10/yolo11s
echo "yolo11s DONE at $(date)" && echo ""

# m
echo "[8/10] yolo11m-C2PSA_WDAM -- NWPU_VHR-10 -- batch 8"
$YOLO train model=yolo11m-C2PSA_WDAM.yaml data=NWPU_VHR-10.yaml \
  epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/NWPU_VHR-10/yolo11m
echo "yolo11m DONE at $(date)" && echo ""

# l
echo "[9/10] yolo11l-C2PSA_WDAM -- NWPU_VHR-10 -- batch 8"
$YOLO train model=yolo11l-C2PSA_WDAM.yaml data=NWPU_VHR-10.yaml \
  epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/NWPU_VHR-10/yolo11l
echo "yolo11l DONE at $(date)" && echo ""

# x
echo "[10/10] yolo11x-C2PSA_WDAM -- NWPU_VHR-10 -- batch 4"
$YOLO train model=yolo11x-C2PSA_WDAM.yaml data=NWPU_VHR-10.yaml \
  epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
  name=WDAM/NWPU_VHR-10/yolo11x
echo "yolo11x DONE at $(date)" && echo ""

echo "=========================================="
echo "Cloud1: ALL WDAM RUNS COMPLETE"
echo "End: $(date)"
echo "=========================================="