#!/bin/bash
# ============================================================
# 0521_rsod_baseline.sh
# Date: 2026-05-21
# Purpose: RSOD yolo11n baseline verification on no6-ai env
# GPU: 0 (RTX 3080)
# Env: no6-ai (torch 2.11.0+cu126)
# Previous experiments: ALL DEGConv data from ultralytics-no5 is INVALID
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

# === RSOD yolo11n Baseline (verify reproducibility with old Baseline_Model_Experiment) ===
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
echo "Compare with: runs/detect/Baseline_Model_Experiment/$DATASET/yolo11n/results.csv"
