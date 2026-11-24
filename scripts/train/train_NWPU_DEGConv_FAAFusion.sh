#!/bin/bash
# NWPU_VHR-10: DEGConv V3 (s/m/l/x) + FAAFusion (n/s/m/l/x)
# GPU0 (RTX 3080, 20GB) — NWPU 训练图 ~650 (≤1000)
# epochs=200, imgsz=640, data=NWPU_VHR-10.yaml

set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATA=NWPU_VHR-10.yaml
DEVICE=0
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "NWPU_VHR-10: DEGConv V3 + FAAFusion"
echo "Start: $(date)"
echo "=========================================="

# ============ DEGConv V3 (full) s/m/l/x ============
echo "========== DEGConv V3 =========="

# s — batch 16
echo "[1/9] yolo11s-DEGConv-full — batch 16"
$YOLO train model=yolo11s-DEGConv-full.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=DEGConv/NWPU_VHR-10/yolo11s-V3_full
echo "yolo11s-DEGConv DONE at $(date)"
echo ""

# m — batch 8
echo "[2/9] yolo11m-DEGConv-full — batch 8"
$YOLO train model=yolo11m-DEGConv-full.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=DEGConv/NWPU_VHR-10/yolo11m-V3_full
echo "yolo11m-DEGConv DONE at $(date)"
echo ""

# l — batch 8
echo "[3/9] yolo11l-DEGConv-full — batch 8"
$YOLO train model=yolo11l-DEGConv-full.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=DEGConv/NWPU_VHR-10/yolo11l-V3_full
echo "yolo11l-DEGConv DONE at $(date)"
echo ""

# x — batch 4
echo "[4/9] yolo11x-DEGConv-full — batch 4"
$YOLO train model=yolo11x-DEGConv-full.yaml data=$DATA epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
  name=DEGConv/NWPU_VHR-10/yolo11x-V3_full
echo "yolo11x-DEGConv DONE at $(date)"
echo ""

# ============ FAAFusion n/s/m/l/x ============
echo "========== FAAFusion =========="

# n — batch 16
echo "[5/9] yolo11n-FAAFusion — batch 16"
$YOLO train model=yolo11n-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=FAAFusion/NWPU_VHR-10/yolo11n-FAAFusion
echo "yolo11n-FAAFusion DONE at $(date)"
echo ""

# s — batch 16
echo "[6/9] yolo11s-FAAFusion — batch 16"
$YOLO train model=yolo11s-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
  name=FAAFusion/NWPU_VHR-10/yolo11s-FAAFusion
echo "yolo11s-FAAFusion DONE at $(date)"
echo ""

# m — batch 8
echo "[7/9] yolo11m-FAAFusion — batch 8"
$YOLO train model=yolo11m-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=FAAFusion/NWPU_VHR-10/yolo11m-FAAFusion
echo "yolo11m-FAAFusion DONE at $(date)"
echo ""

# l — batch 8
echo "[8/9] yolo11l-FAAFusion — batch 8"
$YOLO train model=yolo11l-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
  name=FAAFusion/NWPU_VHR-10/yolo11l-FAAFusion
echo "yolo11l-FAAFusion DONE at $(date)"
echo ""

# x — batch 4
echo "[9/9] yolo11x-FAAFusion — batch 4"
$YOLO train model=yolo11x-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
  name=FAAFusion/NWPU_VHR-10/yolo11x-FAAFusion
echo "yolo11x-FAAFusion DONE at $(date)"
echo ""

echo "=========================================="
echo "NWPU_VHR-10 全部完成！"
echo "Start:  $(cat /proc/$$/status | grep Seccomp)"
echo "End: $(date)"
echo "=========================================="
