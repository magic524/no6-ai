#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai
DATA=RSOD.yaml
GT=/mnt/e/Datasets/Small_Objects_Dataset/RSOD/annotations/instances_val.json
BASE=runs/detect/Baseline_Model_Experiment/RSOD

for model in yolo11n yolo11s yolo11m yolo11l yolo11x; do
  echo "=== VAL: $model (GPU0) ==="
  /home/magic524/miniconda3/envs/no6-ai/bin/yolo val \
    model=$BASE/$model/weights/best.pt \
    data=$DATA device=0 save_json=True \
    name=val_rsod_yolo11_${model}

  VAL_DIR=runs/detect/val_rsod_yolo11_${model}
  echo "=== COCO EVAL: $model ==="
  /home/magic524/miniconda3/envs/no6-ai/bin/python eval_ap.py \
    --gt $GT \
    --pred $VAL_DIR/predictions.json \
    --output $VAL_DIR/predictions_fixed.json
done

echo "=== ALL RSOD YOLO11 EVAL DONE ==="
