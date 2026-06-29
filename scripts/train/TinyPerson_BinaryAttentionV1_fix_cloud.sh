#!/bin/bash
# ============================================================
# TinyPerson_BinaryAttentionV1_fix_cloud.sh
# Date: 2026-06-05
# Purpose: Fix l/x scales that returned near-zero mAP (0.227/0.001)
#          BinaryAttentionV1: float32+adaptive clip+smooth STE
# Machine: Cloud-2 (3090-2, port 10951)
# Env: no6-ai (torch 2.11.0+cu128)
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=TinyPerson.yaml

# === 1. l-scale (batch=8) — 原版 mAP50=0.227 ===
echo "========== 1. TinyPerson yolo11l-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11l-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttentionV1/TinyPerson/yolo11l
echo "=== TinyPerson yolo11l-BinaryAttentionV1 COMPLETE ==="

# === 2. x-scale (batch=4) — 原版 mAP50=0.001 ===
echo "========== 2. TinyPerson yolo11x-BinaryAttentionV1 =========="
$YOLO detect train model=yolo11x-BinaryAttentionV1.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=0 \
  name=BinaryAttentionV1/TinyPerson/yolo11x
echo "=== TinyPerson yolo11x-BinaryAttentionV1 COMPLETE ==="

echo ""
echo "============================================"
echo "TINYPERSON BINARYATTENTIONV1 FIX RUNS COMPLETE!"
echo "============================================"
