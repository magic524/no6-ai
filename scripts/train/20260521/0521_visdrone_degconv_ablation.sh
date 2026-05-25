#!/bin/bash
# ============================================================
# 0521_visdrone_degconv_ablation.sh
# Date: 2026-05-21
# Purpose: VisDrone DEGConv ablation (V1/V2/V3) on no6-ai env
# GPU: 1 (RTX 2080 Ti, 22.5GB)
# Env: no6-ai (torch 2.11.0+cu126)
# Note: Baseline uses existing Baseline_Model_Experiment (mAP50≈0.3091, confirmed OK)
#       V3 Full uses batch=8 to avoid OOM on GPU1 (RTX 2080 Ti has 11GB effective)
# ============================================================
set -euo pipefail

cd /home/magic524/projects/no6-ai

# === Configuration ===
YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATASET=VisDrone
GPU=1
EPOCHS=200
IMGSZ=640
MODEL_DIR=ultralytics/cfg/models/11/DEGConv

echo "=========================================="
echo "[$(date)] VisDrone DEGConv Ablation Start"
echo "GPU: $GPU | Epochs: $EPOCHS"
echo "=========================================="

# === V1: Backbone DEGConv (batch=16) ===
echo ""
echo "--- [$(date)] V1 Backbone start ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=${MODEL_DIR}/yolo11-DEGConv.yaml \
  epochs=$EPOCHS batch=16 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V1_backbone
echo "--- [$(date)] V1 Backbone DONE ✅ ---"

# === V2: Neck DEGConv (batch=16) ===
echo ""
echo "--- [$(date)] V2 Neck start ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=${MODEL_DIR}/yolo11-DEGConv-neck.yaml \
  epochs=$EPOCHS batch=16 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V2_neck
echo "--- [$(date)] V2 Neck DONE ✅ ---"

# === V3: Full DEGConv (batch=8 to avoid OOM on 2080 Ti) ===
echo ""
echo "--- [$(date)] V3 Full start (batch=8) ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=${MODEL_DIR}/yolo11-DEGConv-full.yaml \
  epochs=$EPOCHS batch=8 imgsz=$IMGSZ device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-V3_full
echo "--- [$(date)] V3 Full DONE ✅ ---"

echo ""
echo "=========================================="
echo "[$(date)] ALL VisDrone DEGConv experiments complete! ✅"
echo "=========================================="
