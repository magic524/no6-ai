#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

for model in yolo26l yolo26m yolo26s yolo26n; do
  case $model in
    yolo26l) BATCH=8 ;;
    yolo26m) BATCH=8 ;;
    yolo26s) BATCH=16 ;;
    yolo26n) BATCH=16 ;;
  esac
  echo "=========================================="
  echo "=== START: $model (batch=$BATCH, device=0) ==="
  echo "=========================================="
  conda run -n no6-ai yolo detect train data=RSOD.yaml model=ultralytics/cfg/models/26/$model.yaml batch=$BATCH device=0 name="Baseline_Model_Experiment/RSOD/$model"
  echo "=== TRAINING COMPLETE: $model ==="
done

echo ""
echo "=== ALL RSOD RUNS COMPLETED! ==="
