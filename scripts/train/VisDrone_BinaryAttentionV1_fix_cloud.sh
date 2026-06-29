#!/bin/bash
# ============================================================
# VisDrone_BinaryAttentionV1_fix_cloud.sh
# Date: 2026-06-05
# Purpose: Fix m/l/x scales that returned mAP=0.000 (OOM/NaN)
#          BinaryAttentionV1: float32+adaptive clip+smooth STE
# Machine: Cloud-1 (3090-1, port 44908)
# Env: no6-ai (torch 2.11.0+cu128)
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml

# === 1. m-scale (batch=8) — 原版 mAP=0.000 ===
echo "========== 1. VisDrone yolo11m-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11m-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttentionV1/VisDrone/yolo11m
echo "=== VisDrone yolo11m-BinaryAttentionV1 COMPLETE ==="

# === 2. l-scale (batch=8) — 原版 mAP=0.000 ===
echo "========== 2. VisDrone yolo11l-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11l-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttentionV1/VisDrone/yolo11l
echo "=== VisDrone yolo11l-BinaryAttentionV1 COMPLETE ==="

# === 3. x-scale (batch=4) — 原版 mAP=0.000 ===
echo "========== 3. VisDrone yolo11x-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11x-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=0 \
  name=BinaryAttentionV1/VisDrone/yolo11x
echo "=== VisDrone yolo11x-BinaryAttentionV1 COMPLETE ==="

echo ""
echo "============================================"
echo "VISDRONE BINARYATTENTIONV1 FIX RUNS COMPLETE!"
echo "============================================"
