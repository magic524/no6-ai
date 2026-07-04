#!/bin/bash
# C2PSA_AFFN — 5-epoch smoke test (RSOD, n-scale)
# Verifies module runs without errors before full training

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=RSOD.yaml
DEVICE=0

echo "[SMOKE TEST] C2PSA_AFFN — yolo11n, RSOD, 5 epochs, batch 16"
$YOLO detect train data=$DATA model=yolo11n-C2PSA_AFFN.yaml epochs=5 batch=16 device=$DEVICE
echo "Smoke test DONE at $(date)"
