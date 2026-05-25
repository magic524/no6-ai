#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo

echo "=== $(date) VisDrone yolo26l resume ==="
$YOLO detect train resume model=runs/detect/Baseline_Model_Experiment/VisDrone/yolo26l/weights/last.pt
echo "=== $(date) yolo26l COMPLETE ==="

echo "=== $(date) VisDrone yolo26x start ==="
$YOLO detect train data=VisDrone.yaml model=ultralytics/cfg/models/26/yolo26x.yaml batch=4 device=1 name="Baseline_Model_Experiment/VisDrone/yolo26x"
echo "=== $(date) yolo26x COMPLETE ==="
