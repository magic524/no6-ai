#!/bin/bash
# ============================================================
# VisDrone_BinaryAttentionV1_ns_cloud.sh
# Date: 2026-06-10
# Purpose: Train BinaryAttentionV1 on VisDrone for n-scale and s-scale
# Machine: Cloud GPU (RTX 3090, 24GB)
# Env: no6-ai
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0

echo "============================================"
echo "BinaryAttentionV1 VisDrone n + s training"
echo "============================================"
echo ""

# === 1. n-scale (batch=16) ===
echo "========== 1. VisDrone yolo11n-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11n-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV1/VisDrone/yolo11n
echo "=== VisDrone yolo11n-BinaryAttentionV1 COMPLETE ==="
echo ""

# === 2. s-scale (batch=16) ===
echo "========== 2. VisDrone yolo11s-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11s-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV1/VisDrone/yolo11s
echo "=== VisDrone yolo11s-BinaryAttentionV1 COMPLETE ==="
echo ""

echo ""
echo "============================================"
echo "ALL BINARYATTENTIONV1 VISDRONE RUNS DONE!"
echo "============================================"
