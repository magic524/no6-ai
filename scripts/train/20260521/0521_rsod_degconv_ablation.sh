#!/bin/bash
# ============================================================
# 0521_rsod_degconv_ablation.sh
# Date: 2026-05-21
# Purpose: RSOD DEGConv ablation (V1/V2/V3) on no6-ai env
# GPU: 0 (RTX 3080)
# Env: no6-ai (torch 2.11.0+cu126)
# Note: Run AFTER baseline verified. Only 3 variants (no V4/V5).
# ============================================================
set -euo pipefail

cd /home/magic524/projects/no6-ai

# === Configuration ===
YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATASET=RSOD
GPU=0
EPOCHS=200
BATCH=16
IMGSZ=640
MODEL_DIR=ultralytics/cfg/models/11/DEGConv

echo "=========================================="
echo "[$(date)] RSOD DEGConv Ablation Start"
echo "GPU: $GPU | Epochs: $EPOCHS | Batch: $BATCH"
echo "=========================================="

# === V1: Backbone DEGConv ===
echo ""
echo "--- [$(date)] V1 Backbone start ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=${MODEL_DIR}/yolo11-DEGConv.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V1_backbone
echo "--- [$(date)] V1 Backbone DONE ✅ ---"

# === V2: Neck DEGConv ===
echo ""
echo "--- [$(date)] V2 Neck start ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=${MODEL_DIR}/yolo11-DEGConv-neck.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V2_neck
echo "--- [$(date)] V2 Neck DONE ✅ ---"

# === V3: Full DEGConv (Backbone + Neck) ===
echo ""
echo "--- [$(date)] V3 Full start ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=${MODEL_DIR}/yolo11-DEGConv-full.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V3_full
echo "--- [$(date)] V3 Full DONE ✅ ---"

echo ""
echo "=========================================="
echo "[$(date)] ALL RSOD DEGConv experiments complete! ✅"
echo "=========================================="
