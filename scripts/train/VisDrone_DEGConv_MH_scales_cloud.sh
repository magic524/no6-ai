#!/bin/bash
# ============================================================
# VisDrone_DEGConv_MH_scales_cloud.sh
# Date: 2026-06-10
# Purpose: Train yolo11-DEGConv_MH on VisDrone for s/m/l/x scales
# Machine: Cloud GPU (RTX 3090, 24GB)
# Env: no6-ai
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0

echo "============================================"
echo "DEGConv_MH VisDrone s/m/l/x scale training"
echo "============================================"
echo ""

# === 1. s-scale (batch=16) ===
echo "========== 1. VisDrone yolo11s-DEGConv_MH =========="
$YOLO detect train model=yolo11s-DEGConv_MH.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11s-MH
echo "=== VisDrone yolo11s-MH COMPLETE ==="
echo ""

# === 2. m-scale (batch=8) ===
echo "========== 2. VisDrone yolo11m-DEGConv_MH =========="
$YOLO detect train model=yolo11m-DEGConv_MH.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11m-MH
echo "=== VisDrone yolo11m-MH COMPLETE ==="
echo ""

# === 3. l-scale (batch=8) ===
echo "========== 3. VisDrone yolo11l-DEGConv_MH =========="
$YOLO detect train model=yolo11l-DEGConv_MH.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11l-MH
echo "=== VisDrone yolo11l-MH COMPLETE ==="
echo ""

# === 4. x-scale (batch=4) ===
echo "========== 4. VisDrone yolo11x-DEGConv_MH =========="
$YOLO detect train model=yolo11x-DEGConv_MH.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11x-MH
echo "=== VisDrone yolo11x-MH COMPLETE ==="
echo ""

echo ""
echo "============================================"
echo "ALL VISDRONE DEGConv_MH SCALES DONE!"
echo "============================================"
