#!/bin/bash
# ============================================================
# 0524_resume_lx_visdrone.sh
# Date: 2026-05-24
# Purpose:
#   - Resume yolo11l-V3_full (169→200, data=VisDrone.yaml)
#   - Train yolo11x-V3_full (0→200, batch=4)
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
echo "[$(date)] Resume LX VisDrone V3 Full Experiments"
echo "GPU: $GPU"
echo "=========================================="

# === Resume l-scale (169→200, resume from last.pt) ===
echo ""
echo "--- [$(date)] yolo11l-V3_full resume (169→200) ---"
$YOLO detect train resume \
  model=runs/detect/DEGConv_ablation/${DATASET}/yolo11l-V3_full/weights/last.pt \
  data=${DATASET}.yaml
echo "--- [$(date)] yolo11l-V3_full resume DONE ✅ ---"

# === Fresh x-scale (0→200, batch=4) ===
echo ""
echo "--- [$(date)] yolo11x-V3_full start (batch=4) ---"
$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11x-DEGConv-full.yaml \
  epochs=$EPOCHS \
  batch=4 \
  imgsz=$IMGSZ \
  device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11x-V3_full
echo "--- [$(date)] yolo11x-V3_full DONE ✅ ---"

echo ""
echo "=========================================="
echo "[$(date)] LX experiments complete! ✅"
echo "=========================================="
