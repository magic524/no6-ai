#!/bin/bash
# ============================================================
# VisDrone_DEGConv_improvements_cloud.sh
# Date: 2026-06-08
# Purpose: Train 3 DEGConv improvement variants on VisDrone n-scale
#          AP (Adaptive Patch), DE (Detail-Enhanced), MH (Multi-Head)
# Machine: Cloud GPU (RTX 3090, 24GB)
# Env: no6-ai
# ============================================================
set -euo pipefail

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0

echo "========== DEGConv Improvement Experiments on VisDrone n-scale =========="
echo ""

# === 1. AP-DEGConv (Adaptive Patch) ===
echo "========== 1. VisDrone yolo11n-DEGConv_AP =========="
$YOLO detect train model=yolo11n-DEGConv_AP.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11n-AP
echo "=== yolo11n-AP COMPLETE ==="
echo ""

# === 2. DE-DEGConv (Detail-Enhanced) ===
echo "========== 2. VisDrone yolo11n-DEGConv_DE =========="
$YOLO detect train model=yolo11n-DEGConv_DE.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11n-DE
echo "=== yolo11n-DE COMPLETE ==="
echo ""

# === 3. MH-DEGConv (Multi-Head) ===
echo "========== 3. VisDrone yolo11n-DEGConv_MH =========="
$YOLO detect train model=yolo11n-DEGConv_MH.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=DEGConv_improvement/VisDrone/yolo11n-MH
echo "=== yolo11n-MH COMPLETE ==="
echo ""

echo ""
echo "============================================"
echo "ALL 3 VISDRONE DEGConv IMPROVEMENTS DONE!"
echo "============================================"
