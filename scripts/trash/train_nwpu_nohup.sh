#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
LOG=/home/magic524/projects/no6-ai/logs/nwpu_yolo11_resume.log

{
  echo "=== $(date) NWPU yolo11l resume ==="
  $YOLO detect train resume model=runs/detect/Baseline_Model_Experiment/NWPU_VHR-10/yolo11l/weights/last.pt
  echo "=== $(date) yolo11l COMPLETE ==="

  for model in yolo11m yolo11s yolo11n; do
    case $model in
      yolo11m) BATCH=8 ;;
      yolo11s | yolo11n) BATCH=16 ;;
    esac
    echo "=== $(date) $model start ==="
    $YOLO detect train data=NWPU_VHR-10.yaml model=${model}.yaml batch=$BATCH device=0 name="Baseline_Model_Experiment/NWPU_VHR-10/$model"
    echo "=== $(date) $model COMPLETE ==="
  done

  echo "=== ALL NWPU YOLO11 RUNS COMPLETED! ==="
} >> "$LOG" 2>&1
