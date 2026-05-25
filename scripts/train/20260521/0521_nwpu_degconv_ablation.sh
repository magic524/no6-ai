#!/bin/bash
# ============================================================
# 0521_nwpu_degconv_ablation.sh
# Date: 2026-05-21
# Purpose: NWPU_VHR-10 baseline + DEGConv ablation (V1/V2/V3)
# GPU: 0 (RTX 3080)
# Env: no6-ai (torch 2.11.0+cu126)
# ============================================================
set -euo pipefail

cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATASET=NWPU_VHR-10
GPU=0
EPOCHS=200
BATCH=16
IMGSZ=640
# Model paths use short names — YOLO auto-resolves from cfg tree

echo "=========================================="
echo "[$(date)] Starting $DATASET experiments"
echo "GPU: $GPU | Epochs: $EPOCHS | Batch: $BATCH"
echo "=========================================="

# === Baseline ===
echo ""
echo "--- [$(date)] Baseline start ---"
$YOLO detect train \
  data=${DATASET}.yaml model=yolo11n.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-baseline
echo "--- [$(date)] Baseline DONE ---"

# === V1 Backbone ===
echo ""
echo "--- [$(date)] V1 Backbone start ---"
$YOLO detect train \
  data=${DATASET}.yaml model=yolo11n-DEGConv.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V1_backbone
echo "--- [$(date)] V1 Backbone DONE ---"

# === V2 Neck ===
echo ""
echo "--- [$(date)] V2 Neck start ---"
$YOLO detect train \
  data=${DATASET}.yaml model=yolo11n-DEGConv-neck.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V2_neck
echo "--- [$(date)] V2 Neck DONE ---"

# === V3 Full ===
echo ""
echo "--- [$(date)] V3 Full start ---"
$YOLO detect train \
  data=${DATASET}.yaml model=yolo11n-DEGConv-full.yaml \
  epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V3_full
echo "--- [$(date)] V3 Full DONE ---"

echo ""
echo "=========================================="
echo "[$(date)] ALL $DATASET experiments complete! ✅"
echo "=========================================="
