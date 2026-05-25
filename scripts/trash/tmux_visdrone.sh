#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

echo "=== VISDRONE: Resume yolo26m (epoch 148) ==="
conda run -n no6-ai yolo detect train model=runs/detect/Baseline_Model_Experiment/VisDrone/yolo26m/weights/last.pt resume=True
echo "=== VISDRONE: yolo26m done ==="

echo "=== VISDRONE: yolo26l (batch=8, device=1) ==="
conda run -n no6-ai yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26l.yaml batch=8 device=1 name='Baseline_Model_Experiment/VisDrone/yolo26l'
echo "=== VISDRONE: yolo26l done ==="

echo "=== VISDRONE: yolo26x (batch=4, device=1) ==="
conda run -n no6-ai yolo detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26x.yaml batch=4 device=1 name='Baseline_Model_Experiment/VisDrone/yolo26x'
echo "=== VISDRONE: yolo26x done ==="

echo "=== ALL VISDRONE COMPLETE ==="
