#!/bin/bash
set -euo pipefail
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PY=/root/miniconda3/envs/no6-ai/bin/python
EVAL=/root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py
BASE=/root/autodl-tmp/no6-ai/runs/detect
PROJ=/root/autodl-tmp/no6-ai/runs/detect/val/bav1_nd
GT=/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json

echo "Starting m/l/x val+eval+params..."

for s in m l x; do
  best=$BASE/BinaryAttentionV1/NightDrone/yolo11$s/weights/best.pt

  echo "========== yolo11$s =========="

  # val
  $YOLO val model=$best data=NightDrone.yaml batch=32 device=0 imgsz=640 save_json=True \
    project=$PROJ name=yolo11$s 2>&1 | tee $PROJ/yolo11$s/val.log

  # eval_ap
  pred=$PROJ/yolo11$s/predictions.json
  $PY $EVAL --gt $GT --pred $pred 2>&1 | tee $PROJ/yolo11$s/eval_ap.log

  # params
  $PY -c "from ultralytics import YOLO; m=YOLO('$best'); print(f'{sum(p.numel() for p in m.model.parameters())/1e6:.2f}')" \
    2>&1 | tee $PROJ/yolo11$s/params.txt

  echo "yolo11$s DONE"
done

echo "ALL DONE"
