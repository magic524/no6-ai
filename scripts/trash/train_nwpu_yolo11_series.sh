#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo

for model in yolo11x yolo11l yolo11m yolo11s yolo11n; do
    case $model in
        yolo11x) BATCH=4 ;;
        yolo11l|yolo11m) BATCH=8 ;;
        yolo11s|yolo11n) BATCH=16 ;;
    esac
    echo "=========================================="
    echo "=== START: $model (batch=$BATCH, device=0) ==="
    echo "=========================================="
    $YOLO detect train data=NWPU_VHR-10.yaml model=${model}.yaml batch=$BATCH device=0 name="Baseline_Model_Experiment/NWPU_VHR-10/$model"
    echo "=== TRAINING COMPLETE: $model ==="
done

echo ""
echo "=== ALL NWPU_VHR-10 YOLO11 RUNS COMPLETED! ==="
