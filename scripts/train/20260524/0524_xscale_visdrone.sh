#!/bin/bash
# ============================================================
# 0524_xscale_visdrone.sh
# Date: 2026-05-24
# Purpose: Train VisDrone V3 Full x-scale (200 epochs, fresh)
# GPU: 1 (RTX 2080 Ti, 22.5GB)
# Env: no6-ai (torch 2.11.0+cu126)
# ============================================================
set -euo pipefail

cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATASET=VisDrone
GPU=1
EPOCHS=200
IMGSZ=640

echo "=========================================="
echo "[$(date)] VisDrone V3 Full x-scale start"
echo "GPU: $GPU | Epochs: $EPOCHS | Batch: 4"
echo "=========================================="

$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11x-DEGConv-full.yaml \
  epochs=$EPOCHS \
  batch=4 \
  imgsz=$IMGSZ \
  device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11x-V3_full

echo ""
echo "[$(date)] yolo11x-V3_full DONE ✅"
