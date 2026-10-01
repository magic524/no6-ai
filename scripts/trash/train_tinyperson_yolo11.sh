#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
LOG=/home/magic524/projects/no6-ai/logs/tinyperson_yolo11_train.log

{
  echo "=== $(date) TinyPerson YOLO11 Series Start ==="

  for MODEL in yolo11n yolo11s yolo11m yolo11l yolo11x; do
    DIR=runs/detect/Baseline_Model_Experiment/TinyPerson/$MODEL

    # Set batch size
    case $MODEL in
      yolo11n | yolo11s) BATCH=16 ;;
      yolo11m | yolo11l) BATCH=8 ;;
      yolo11x) BATCH=4 ;;
    esac

    if [ -f "$DIR/weights/best.pt" ]; then
      echo "=== $(date) $MODEL already completed, skipping ==="
      continue
    fi

    if [ -f "$DIR/weights/last.pt" ]; then
      echo "=== $(date) $MODEL resuming from last.pt ==="
      $YOLO detect train resume=True model=$DIR/weights/last.pt device=0
    else
      echo "=== $(date) $MODEL fresh start (batch=$BATCH) ==="
      $YOLO detect train \
        data=TinyPerson.yaml \
        model=$MODEL.yaml \
        batch=$BATCH \
        device=0 \
        name=Baseline_Model_Experiment/TinyPerson/$MODEL
    fi
    echo "=== $(date) $MODEL COMPLETE ==="
  done

  echo "=== $(date) ALL TinyPerson YOLO11 RUNS COMPLETED! ==="
} >> "$LOG" 2>&1
