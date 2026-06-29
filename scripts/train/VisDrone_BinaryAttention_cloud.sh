#!/bin/bash
# ============================================================
# VisDrone_BinaryAttention_cloud.sh
# Date: 2026-06-03
# Purpose: Train YOLO11 BinaryAttention (1-bit) on VisDrone, all 5 scales
# Machine: Cloud-1 (3090-1, port 44908)
# Env: no6-ai (torch 2.11.0+cu128)
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml

# === 1. n-scale (batch=16) ===
echo "========== 1. VisDrone yolo11n-BinaryAttention =========="
$YOLO detect train model=yolo11n-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=0 \
  name=BinaryAttention/VisDrone/yolo11n
echo "=== VisDrone yolo11n-BinaryAttention COMPLETE ==="

# === 2. s-scale (batch=16) ===
echo "========== 2. VisDrone yolo11s-BinaryAttention =========="
$YOLO detect train model=yolo11s-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=0 \
  name=BinaryAttention/VisDrone/yolo11s
echo "=== VisDrone yolo11s-BinaryAttention COMPLETE ==="

# === 3. m-scale (batch=8) ===
echo "========== 3. VisDrone yolo11m-BinaryAttention =========="
$YOLO detect train model=yolo11m-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttention/VisDrone/yolo11m
echo "=== VisDrone yolo11m-BinaryAttention COMPLETE ==="

# === 4. l-scale (batch=8) ===
echo "========== 4. VisDrone yolo11l-BinaryAttention =========="
$YOLO detect train model=yolo11l-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttention/VisDrone/yolo11l
echo "=== VisDrone yolo11l-BinaryAttention COMPLETE ==="

# === 5. x-scale (batch=4) ===
echo "========== 5. VisDrone yolo11x-BinaryAttention =========="
$YOLO detect train model=yolo11x-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=0 \
  name=BinaryAttention/VisDrone/yolo11x
echo "=== VisDrone yolo11x-BinaryAttention COMPLETE ==="

echo ""
echo "============================================"
echo "ALL VISDRONE BINARYATTENTION RUNS COMPLETED!"
echo "============================================"
