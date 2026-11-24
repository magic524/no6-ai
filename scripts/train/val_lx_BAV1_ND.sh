#!/bin/bash
set -euo pipefail

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PY=/root/miniconda3/envs/no6-ai/bin/python
EVAL=/root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py
GT=/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json
BASE=/root/autodl-tmp/no6-ai/runs/detect

for s in l x; do
  best=$BASE/BinaryAttentionV1/NightDrone/yolo11$s/weights/best.pt
  # Clean prior output
  rm -rf $BASE/val/bav1_nd/yolo11$s*

  echo "========== yolo11$s =========="

  # val
  $YOLO val model=$best data=NightDrone.yaml batch=32 device=0 imgsz=640 save_json=True \
    project=$BASE/val/bav1_nd name=yolo11$s 2>&1 | tee /tmp/val_${s}.log

  # Find actual output dir
  out=$(find $BASE/val/bav1_nd -maxdepth 1 -type d -name "yolo11$s*" | sort | tail -1)
  cp /tmp/val_${s}.log $out/

  map50=$(grep -a '^[[:space:]]*all' /tmp/val_${s}.log | awk '{print $7}')
  map95=$(grep -a '^[[:space:]]*all' /tmp/val_${s}.log | awk '{print $8}')
  gflops=$(grep -oP '[\d.]+(?= GFLOPs)' /tmp/val_${s}.log | head -1)
  echo "mAP50=$map50  mAP50-95=$map95  GFLOPs=$gflops"

  # eval_ap
  $PY $EVAL --gt $GT --pred $out/predictions.json 2>&1 | tee $out/eval_ap.log

  ap_s=$(grep 'AP_s:' $out/eval_ap.log | awk '{print $2}')
  ap_m=$(grep 'AP_m:' $out/eval_ap.log | awk '{print $2}')
  ap_l=$(grep 'AP_l:' $out/eval_ap.log | awk '{print $2}')
  echo "AP_s=$ap_s  AP_m=$ap_m  AP_l=$ap_l"

  # params
  $PY -c "from ultralytics import YOLO; m=YOLO('$best'); print(f'{sum(p.numel() for p in m.model.parameters())/1e6:.2f}')" 2>&1 | tee $out/params.txt
  params=$(cat $out/params.txt)
  echo "Params=${params}M"

  # summary
  cat > $out/summary.txt << EOF
SCALE=$s
mAP50=$map50
mAP50-95=$map95
AP_s=$ap_s
AP_m=$ap_m
AP_l=$ap_l
Params(M)=$params
GFLOPs=$gflops
EOF
  cat $out/summary.txt
  echo "yolo11$s DONE"
done

echo "ALL DONE"
