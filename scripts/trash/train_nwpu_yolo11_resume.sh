#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo

echo "=========================================="
echo "=== RESUME: yolo11x (device=0) ==="
echo "=========================================="
$YOLO detect train resume model=runs/detect/Baseline_Model_Experiment/NWPU_VHR-10/yolo11x-2/weights/last.pt
echo "=== yolo11x COMPLETE ==="

for model in yolo11l yolo11m yolo11s yolo11n; do
  case $model in
    yolo11l | yolo11m) BATCH=8 ;;
    yolo11s | yolo11n) BATCH=16 ;;
  esac
  echo "=========================================="
  echo "=== START: $model (batch=$BATCH, device=0) ==="
  echo "=========================================="
  $YOLO detect train data=NWPU_VHR-10.yaml model=${model}.yaml batch=$BATCH device=0 name="Baseline_Model_Experiment/NWPU_VHR-10/$model"
  echo "=== $model COMPLETE ==="
done

echo ""
echo "=== ALL NWPU YOLO11 RUNS COMPLETED! ==="
