#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
LOG=/home/magic524/projects/no6-ai/logs/visdrone_yolo11_train.log

{
  echo "=== $(date) VisDrone YOLO11 Series Start ==="

  for MODEL in yolo11n yolo11s yolo11m yolo11l yolo11x; do
    DIR=runs/detect/Baseline_Model_Experiment/VisDrone/$MODEL

    # Set batch size
    case $MODEL in
      yolo11n | yolo11s) BATCH=16 ;;
      yolo11m | yolo11l) BATCH=8 ;;
      yolo11x) BATCH=4 ;;
    esac

    # If best.pt exists → already complete, skip
    if [ -f "$DIR/weights/best.pt" ]; then
      echo "=== $(date) $MODEL already completed, skipping ==="
      continue
    fi

    # If last.pt exists but no best.pt → partial training, resume
    if [ -f "$DIR/weights/last.pt" ]; then
      echo "=== $(date) $MODEL resuming from last.pt ==="
      $YOLO detect train \
        resume=True \
        model=$DIR/weights/last.pt \
        device=1
    else
      # Fresh training
      echo "=== $(date) $MODEL fresh start (batch=$BATCH) ==="
      $YOLO detect train \
        data=VisDrone.yaml \
        model=$MODEL.yaml \
        batch=$BATCH \
        device=1 \
        name=Baseline_Model_Experiment/VisDrone/$MODEL
    fi
    echo "=== $(date) $MODEL COMPLETE ==="
  done

  echo "=== $(date) ALL VISDRONE YOLO11 RUNS COMPLETED! ==="
} >> "$LOG" 2>&1
