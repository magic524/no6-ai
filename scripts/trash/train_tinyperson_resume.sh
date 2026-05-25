#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

echo "=========================================="
echo "=== RESUME: yolo26l (epoch 109/200) ==="
echo "=========================================="
conda run -n no6-ai yolo detect train model=runs/detect/Baseline_Model_Experiment/TinyPerson/yolo26l/weights/last.pt resume=True
echo "=== TRAINING COMPLETE: yolo26l ==="

echo "=========================================="
echo "=== START: yolo26x (batch=4, device=1) ==="
echo "=========================================="
conda run -n no6-ai yolo detect train data=TinyPerson.yaml model=ultralytics/cfg/models/26/yolo26x.yaml batch=4 device=1 name='Baseline_Model_Experiment/TinyPerson/yolo26x'
echo "=== TRAINING COMPLETE: yolo26x ==="

echo ""
echo "=========================================="
echo "  ALL TRAINING RUNS COMPLETED!  "
echo "=========================================="
