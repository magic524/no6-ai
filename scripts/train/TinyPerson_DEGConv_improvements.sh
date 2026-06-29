#!/bin/bash
# ============================================================
# TinyPerson_DEGConv_improvements.sh
# Date: 2026-06-05
# Purpose: Test 3 DEGConv small-object improvement variants
#          AP (Adaptive Patch), DE (Detail-Enhanced), MH (Multi-Head)
# Machine: GPU1 (RTX 2080 Ti, device=1)
# Env: no6-ai (torch 2.11.0+cu128)
# ============================================================
set -euo pipefail

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATA=TinyPerson.yaml
DEVICE=1

echo "========== DEGConv Improvement Experiments on TinyPerson n-scale =========="
echo "Baseline: DEGConv V3 Full n-scale TinyPerson mAP50=0.315"
echo ""
sleep 2

# === 1. AP-DEGConv (Adaptive Patch) ===
echo "========== 1. TinyPerson yolo11n-DEGConv_AP =========="
$YOLO detect train data=$DATA model=yolo11n-DEGConv_AP.yaml epochs=200 batch=16 \
  imgsz=640 device=$DEVICE seed=0 \
  name=DEGConv_improvement/TinyPerson/yolo11n-AP
echo "=== yolo11n-AP COMPLETE ==="
echo ""

# === 2. DE-DEGConv (Detail-Enhanced) ===
echo "========== 2. TinyPerson yolo11n-DEGConv_DE =========="
$YOLO detect train data=$DATA model=yolo11n-DEGConv_DE.yaml epochs=200 batch=16 \
  imgsz=640 device=$DEVICE seed=0 \
  name=DEGConv_improvement/TinyPerson/yolo11n-DE
echo "=== yolo11n-DE COMPLETE ==="
echo ""

# === 3. MH-DEGConv (Multi-Head) ===
echo "========== 3. TinyPerson yolo11n-DEGConv_MH =========="
$YOLO detect train data=$DATA model=yolo11n-DEGConv_MH.yaml epochs=200 batch=16 \
  imgsz=640 device=$DEVICE seed=0 \
  name=DEGConv_improvement/TinyPerson/yolo11n-MH
echo "=== yolo11n-MH COMPLETE ==="
echo ""

echo ""
echo "============================================"
echo "ALL 3 DEGConv IMPROVEMENT EXPERIMENTS DONE!"
echo "============================================"
