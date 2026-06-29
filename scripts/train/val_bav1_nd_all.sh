#!/bin/bash
# Sequential val + eval_ap + Params for BAV1 NightDrone all 5 scales
# Key: do NOT pre-create output dirs, find them after yolo val creates them
set -euo pipefail

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PY=/root/miniconda3/envs/no6-ai/bin/python
EVAL=/root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py
DATA=NightDrone.yaml
GT=/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json
BASE=/root/autodl-tmp/no6-ai/runs/detect
PROJECT=$BASE/val/bav1_nd

# Start fresh
rm -rf $PROJECT

for s in n s m l x; do
    best="$BASE/BinaryAttentionV1/NightDrone/yolo11$s/weights/best.pt"
    
    echo ""
    echo "========== yolo11$s-BAV1 NightDrone =========="
    
    # Step 1: yolo val (let it create its own dir)
    echo "[1/4] yolo val ..."
    $YOLO val model=$best data=$DATA batch=32 device=0 imgsz=640 save_json=True \
        project=$PROJECT name=yolo11$s 2>&1 | tee /tmp/val_${s}.log
    
    # Find the actual output dir (yolo may add -N suffix if name exists)
    out=$(find $PROJECT -maxdepth 1 -type d -name "yolo11$s*" | sort | tail -1)
    echo "Output dir: $out"
    
    # Copy log
    cp /tmp/val_${s}.log $out/
    
    # Parse mAP from log
    map50=$(grep -a '^[[:space:]]*all' /tmp/val_${s}.log | awk '{print $7}')
    map95=$(grep -a '^[[:space:]]*all' /tmp/val_${s}.log | awk '{print $8}')
    
    # Parse GFLOPs from log
    gflops=$(grep -oP '[\d.]+(?= GFLOPs)' /tmp/val_${s}.log | head -1)
    
    echo "mAP50=$map50  mAP50-95=$map95  GFLOPs=$gflops"
    
    # Step 2: eval_ap.py
    echo "[2/4] eval_ap.py ..."
    pred=$(find $out -name 'predictions.json' | head -1)
    if [ -n "$pred" ]; then
        $PY $EVAL --gt $GT --pred $pred 2>&1 | tee $out/eval_ap.log
    else
        echo "WARNING: predictions.json not found"
    fi
    
    ap_s=$(grep 'AP_s:' $out/eval_ap.log 2>/dev/null | awk '{print $2}')
    ap_m=$(grep 'AP_m:' $out/eval_ap.log 2>/dev/null | awk '{print $2}')
    ap_l=$(grep 'AP_l:' $out/eval_ap.log 2>/dev/null | awk '{print $2}')
    echo "AP_s=$ap_s  AP_m=$ap_m  AP_l=$ap_l"
    
    # Step 3: Params
    echo "[3/4] Params ..."
    $PY -c "
from ultralytics import YOLO
m = YOLO('$best')
p = sum(p.numel() for p in m.model.parameters()) / 1e6
print(f'{p:.2f}')
" 2>&1 | tee $out/params.txt
    params=$(cat $out/params.txt | tail -1)
    echo "Params=${params}M"
    
    # Save summary
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
    echo "========== Summary =========="
    cat $out/summary.txt
done

echo ""
echo "========================================"
echo "ALL DONE!"
echo "========================================"
for f in $PROJECT/*/summary.txt; do
    echo "--- $f ---"
    cat $f
done
