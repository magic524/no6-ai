#!/bin/bash
# ============================================================
# 0524_resume_visdrone_v3full_scales.sh
# Date: 2026-05-24
# Purpose: Resume interrupted VisDrone V3 Full s/m/l/x scale training
#          - yolo11m-V3_full: 197/200 → 200/200 (resume)
#          - yolo11l-V3_full: 169/200 → 200/200 (resume)
#          - yolo11x-V3_full:   0/200 → 200/200 (fresh train)
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
echo "[$(date)] Resume VisDrone V3 Full Scale Experiments"
echo "GPU: $GPU"
echo "=========================================="

# === Resume m-scale (197→200, resume from last.pt) ===
echo ""
echo "--- [$(date)] yolo11m-V3_full resume (197→200) ---"
$YOLO detect train resume \
  model=runs/detect/DEGConv_ablation/${DATASET}/yolo11m-V3_full/weights/last.pt
echo "--- [$(date)] yolo11m-V3_full resume DONE ✅ ---"

# === Resume l-scale (169→200, resume from last.pt) ===
echo ""
echo "--- [$(date)] yolo11l-V3_full resume (169→200) ---"
$YOLO detect train resume \
  model=runs/detect/DEGConv_ablation/${DATASET}/yolo11l-V3_full/weights/last.pt
echo "--- [$(date)] yolo11l-V3_full resume DONE ✅ ---"

# === Fresh x-scale (0→200, batch=4) ===
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
