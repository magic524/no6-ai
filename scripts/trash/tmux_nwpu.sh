#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai

echo "=== NWPU: Resume yolo26l (epoch 5) ==="
conda run -n no6-ai yolo detect train model=runs/detect/Baseline_Model_Experiment/NWPU_VHR-10/yolo26l/weights/last.pt resume=True
echo "=== NWPU: yolo26l done ==="

echo "=== NWPU: yolo26m (batch=8, device=0) ==="
conda run -n no6-ai yolo detect train data=NWPU_VHR-10.yaml model=ultralytics/cfg/models/26/yolo26m.yaml batch=8 device=0 name='Baseline_Model_Experiment/NWPU_VHR-10/yolo26m'
echo "=== NWPU: yolo26m done ==="

echo "=== NWPU: yolo26s (batch=16, device=0) ==="
conda run -n no6-ai yolo detect train data=NWPU_VHR-10.yaml model=ultralytics/cfg/models/26/yolo26s.yaml batch=16 device=0 name='Baseline_Model_Experiment/NWPU_VHR-10/yolo26s'
echo "=== NWPU: yolo26s done ==="

echo "=== NWPU: yolo26n (batch=16, device=0) ==="
conda run -n no6-ai yolo detect train data=NWPU_VHR-10.yaml model=ultralytics/cfg/models/26/yolo26n.yaml batch=16 device=0 name='Baseline_Model_Experiment/NWPU_VHR-10/yolo26n'
echo "=== NWPU: yolo26n done ==="

echo "=== ALL NWPU COMPLETE ==="
