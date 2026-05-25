#!/bin/bash
# ============================================================
# 0522_visdrone_v3full_scales.sh
# Date: 2026-05-22
# Purpose: VisDrone V3 Full (DEGConv backbone+neck) for s/m/l/x scales
# GPU: 1 (RTX 2080 Ti, 22.5GB)
# Env: no6-ai (torch 2.11.0+cu126)
# Note: n-scale already done (mAP50=0.3287)
#       Batch sizes match standard baseline settings per scale
# ============================================================
set -euo pipefail

cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATASET=VisDrone
GPU=1
EPOCHS=200
IMGSZ=640

echo "=========================================="
echo "[$(date)] VisDrone V3 Full Scale Experiment Start"
echo "GPU: $GPU | Epochs: $EPOCHS"
echo "=========================================="

# === s-scale (batch=16) ===
echo ""
echo "--- [$(date)] yolo11s-V3_full start (batch=16) ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11s-DEGConv-full.yaml \
  epochs=$EPOCHS batch=16 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11s-V3_full
echo "--- [$(date)] yolo11s-V3_full DONE ✅ ---"

# === m-scale (batch=8) ===
echo ""
echo "--- [$(date)] yolo11m-V3_full start (batch=8) ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11m-DEGConv-full.yaml \
  epochs=$EPOCHS batch=8 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11m-V3_full
echo "--- [$(date)] yolo11m-V3_full DONE ✅ ---"

# === l-scale (batch=8) ===
echo ""
echo "--- [$(date)] yolo11l-V3_full start (batch=8) ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11l-DEGConv-full.yaml \
  epochs=$EPOCHS batch=8 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11l-V3_full
echo "--- [$(date)] yolo11l-V3_full DONE ✅ ---"

# === x-scale (batch=4) ===
echo ""
echo "--- [$(date)] yolo11x-V3_full start (batch=4) ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11x-DEGConv-full.yaml \
  epochs=$EPOCHS batch=4 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11x-V3_full
echo "--- [$(date)] yolo11x-V3_full DONE ✅ ---"

echo ""
echo "=========================================="
echo "[$(date)] ALL VisDrone V3 Full scales complete! ✅"
echo "=========================================="
