#!/bin/bash
# ============================================================
# TinyPerson_BinaryAttentionV2_cloud.sh
# Date: 2026-06-15
# Purpose: Train BinaryAttentionV2 on TinyPerson for n/s/m/l/x
# Machine: Cloud 2 (AutoDL 3090, port 10951)
# Env: no6-ai
#
# V2 vs V1:
#   1. Ternary Q/K {+1,0,-1} + 自适应阈值 → 保留幅度信息，过滤噪声
#   2. 混合设计: Attention + DepthwiseConv 并行融合 α
# ============================================================
set -euo pipefail

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=TinyPerson.yaml
DEVICE=0

echo "============================================"
echo "BinaryAttentionV2 TinyPerson 全尺度训练"
echo "============================================"
echo ""

# === 1. n-scale (batch=16) ===
echo "========== 1/5: n-scale =========="
$YOLO detect train model=yolo11n-BinaryAttentionV2.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV2/TinyPerson/yolo11n
echo "=== n COMPLETE ==="
echo ""

# === 2. s-scale (batch=16) ===
echo "========== 2/5: s-scale =========="
$YOLO detect train model=yolo11s-BinaryAttentionV2.yaml data=$DATA \
  epochs=200 batch=16 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV2/TinyPerson/yolo11s
echo "=== s COMPLETE ==="
echo ""

# === 3. m-scale (batch=8) ===
echo "========== 3/5: m-scale =========="
$YOLO detect train model=yolo11m-BinaryAttentionV2.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV2/TinyPerson/yolo11m
echo "=== m COMPLETE ==="
echo ""

# === 4. l-scale (batch=8) ===
echo "========== 4/5: l-scale =========="
$YOLO detect train model=yolo11l-BinaryAttentionV2.yaml data=$DATA \
  epochs=200 batch=8 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV2/TinyPerson/yolo11l
echo "=== l COMPLETE ==="
echo ""

# === 5. x-scale (batch=4) ===
echo "========== 5/5: x-scale =========="
$YOLO detect train model=yolo11x-BinaryAttentionV2.yaml data=$DATA \
  epochs=200 batch=4 imgsz=640 device=$DEVICE \
  name=BinaryAttentionV2/TinyPerson/yolo11x
echo "=== x COMPLETE ==="
echo ""

echo ""
echo "============================================"
echo "BinaryAttentionV2 TINYPERSON ALL SCALES DONE!"
echo "============================================"
