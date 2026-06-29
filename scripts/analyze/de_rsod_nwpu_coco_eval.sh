#!/bin/bash
# DEGConv_DE RSOD + NWPU_VHR-10 COCO eval
# Run on cloud1

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PYTHON=/root/miniconda3/envs/no6-ai/bin/python
RSOD_GT=/root/autodl-tmp/datasets/RSOD/annotations/instances_val.json
NWPU_GT=/root/autodl-tmp/datasets/NWPU_VHR-10/annotations/instances_val.json

# ========== RSOD COCO eval ==========
echo "========== RSOD COCO eval =========="

# RSOD n→s→m→l→x (predictions from training output dirs)
$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $RSOD_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11n/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11n/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $RSOD_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11s/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11s/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $RSOD_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11m/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11m/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $RSOD_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11l/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11l/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $RSOD_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11x/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/RSOD/yolo11x/coco_eval.json

# ========== NWPU_VHR-10 COCO eval ==========
echo "========== NWPU_VHR-10 COCO eval =========="

# NWPU n→s→m→l→x (predictions from training output dirs)
$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $NWPU_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11n/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11n/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $NWPU_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11s/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11s/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $NWPU_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11m/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11m/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $NWPU_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11l/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11l/coco_eval.json

$PYTHON /root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py \
  --gt $NWPU_GT \
  --pred /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11x/predictions.json \
  --output /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/NWPU_VHR-10/yolo11x/coco_eval.json

echo "========== ALL COCO EVAL DONE =========="
