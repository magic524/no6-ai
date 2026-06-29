#!/bin/bash
# ============================================================
# TinyPerson_BinaryAttention_cloud.sh
# Date: 2026-06-04
# Purpose: Train YOLO11 BinaryAttention (1-bit) on TinyPerson, all 5 scales
# Machine: Cloud-2 (3090-2, port 10951)
# Env: no6-ai (torch 2.11.0+cu128)
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=TinyPerson.yaml

# === 1. n-scale (batch=16) ===
echo "========== 1. TinyPerson yolo11n-BinaryAttention =========="
$YOLO detect train model=yolo11n-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=0 \
  name=BinaryAttention/TinyPerson/yolo11n
echo "=== TinyPerson yolo11n-BinaryAttention COMPLETE ==="

# === 2. s-scale (batch=16) ===
echo "========== 2. TinyPerson yolo11s-BinaryAttention =========="
$YOLO detect train model=yolo11s-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=0 \
  name=BinaryAttention/TinyPerson/yolo11s
echo "=== TinyPerson yolo11s-BinaryAttention COMPLETE ==="

# === 3. m-scale (batch=8) ===
echo "========== 3. TinyPerson yolo11m-BinaryAttention =========="
$YOLO detect train model=yolo11m-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttention/TinyPerson/yolo11m
echo "=== TinyPerson yolo11m-BinaryAttention COMPLETE ==="

# === 4. l-scale (batch=8) ===
echo "========== 4. TinyPerson yolo11l-BinaryAttention =========="
$YOLO detect train model=yolo11l-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=0 \
  name=BinaryAttention/TinyPerson/yolo11l
echo "=== TinyPerson yolo11l-BinaryAttention COMPLETE ==="

# === 5. x-scale (batch=4) ===
echo "========== 5. TinyPerson yolo11x-BinaryAttention =========="
$YOLO detect train model=yolo11x-BinaryAttention.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=0 \
  name=BinaryAttention/TinyPerson/yolo11x
echo "=== TinyPerson yolo11x-BinaryAttention COMPLETE ==="

echo ""
echo "============================================"
echo "ALL TINYPERSON BINARYATTENTION RUNS COMPLETED!"
echo "============================================"
