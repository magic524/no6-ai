#!/bin/bash
set -e
cd /home/magic524/projects/no6-ai
DATA=RSOD.yaml
GT=/mnt/e/Datasets/Small_Objects_Dataset/RSOD/annotations/instances_val.json
BASE=runs/detect/Baseline_Model_Experiment/RSOD

for model in yolo26n yolo26s yolo26m yolo26l yolo26x; do
  echo "=== VAL: $model (GPU0) ==="
  conda run -n no6-ai yolo val \
    model=$BASE/$model/weights/best.pt \
    data=$DATA device=0 save_json=True \
    name=val_rsod_${model}

  VAL_DIR=runs/detect/val_rsod_${model}
  echo "=== COCO EVAL: $model ==="
  /home/magic524/miniconda3/envs/no6-ai/bin/python eval_ap.py \
    --gt $GT \
    --pred $VAL_DIR/predictions.json \
    --output $VAL_DIR/predictions_fixed.json
done

echo "=== ALL DONE ==="
