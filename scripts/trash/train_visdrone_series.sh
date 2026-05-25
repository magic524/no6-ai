#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

echo "=========================================="
echo "=== START: yolo26s (batch=16, device=1) ==="
echo "=========================================="
conda run -n no6-ai yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26s.yaml batch=16 device=1 name='Baseline_Model_Experiment/VisDrone/yolo26s'
echo "=== TRAINING COMPLETE: yolo26s ==="

echo "=========================================="
echo "=== START: yolo26m (batch=8, device=1) ==="
echo "=========================================="
conda run -n no6-ai yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26m.yaml batch=8 device=1 name='Baseline_Model_Experiment/VisDrone/yolo26m'
echo "=== TRAINING COMPLETE: yolo26m ==="

echo "=========================================="
echo "=== START: yolo26l (batch=8, device=1) ==="
echo "=========================================="
conda run -n no6-ai yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26l.yaml batch=8 device=1 name='Baseline_Model_Experiment/VisDrone/yolo26l'
echo "=== TRAINING COMPLETE: yolo26l ==="

echo "=========================================="
echo "=== START: yolo26x (batch=4, device=1) ==="
echo "=========================================="
conda run -n no6-ai yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26x.yaml batch=4 device=1 name='Baseline_Model_Experiment/VisDrone/yolo26x'
echo "=== TRAINING COMPLETE: yolo26x ==="

echo ""
echo "=========================================="
echo "  ALL 4 TRAINING RUNS COMPLETED!  "
echo "=========================================="
