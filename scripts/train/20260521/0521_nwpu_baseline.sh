#!/bin/bash
# ============================================================
# 0521_nwpu_baseline.sh
# Date: 2026-05-21
# Purpose: NWPU_VHR-10 yolo11n baseline verification on no6-ai
# GPU: 0 (RTX 3080) — concurrent with VisDrone on GPU1
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

echo "[$(date)] Starting $DATASET yolo11n baseline on GPU$GPU..."

$YOLO detect train \
  data=${DATASET}.yaml \
  model=yolo11n.yaml \
  epochs=$EPOCHS \
  batch=$BATCH \
  imgsz=$IMGSZ \
  device=$GPU \
  name=DEGConv_ablation/${DATASET}/yolo11n-baseline

echo "[$(date)] $DATASET baseline DONE ✅"
