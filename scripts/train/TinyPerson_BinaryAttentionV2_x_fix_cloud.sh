#!/bin/bash
# ============================================================
# TinyPerson_BinaryAttentionV2_x_fix.sh
# Date: 2026-06-15
# Purpose: Re-run BAV2 TinyPerson x-scale with STE gradient fix (V2.1)
#   - Gradient window widened from 2x to 4x threshold
#   - Added 0.02 gradient floor
#   - Fixes EMA NaN that caused best.pt corruption
# Machine: Cloud 2 (AutoDL 3090, port 10951)
# Env: no6-ai
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=TinyPerson.yaml
DEVICE=0

echo "============================================"
echo "BAV2 TinyPerson x-scale V2.1 STE fix re-run"
echo "============================================"
echo ""

$YOLO detect train model=yolo11x-BinaryAttentionV2.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV2/TinyPerson/yolo11x-V2.1

echo ""
echo "=== x-scale V2.1 COMPLETE ==="
echo ""
