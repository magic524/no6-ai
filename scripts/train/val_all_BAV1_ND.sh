#!/bin/bash
set -euo pipefail

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PY=/root/miniconda3/envs/no6-ai/bin/python
EVAL=/root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py
DATA=NightDrone.yaml
GT=/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json
BASE=/root/autodl-tmp/no6-ai/runs/detect
PROJECT=$BASE/val/BAV1_ND

# Clean slate
rm -rf $PROJECT

for s in n s m l x; do
    best="$BASE/BinaryAttentionV1/NightDrone/yolo11$s/weights/best.pt"
    
    echo ""
    echo "========== yolo11$s-BAV1 NightDrone =========="
    
    # 1. yolo val (creates its own dir under $PROJECT)
    echo "[1/4] yolo val ..."
    $YOLO val model=$best data=$DATA batch=32 device=0 imgsz=640 save_json=True \
        project=$PROJECT name=yolo11$s 2>&1 | tee /tmp/val_${s}.log
    
    # Find the actual output dir (may have -N suffix)
    out=$(find $PROJECT -maxdepth 1 -type d -name "yolo11$s*" | sort | tail -1)
    echo "Output dir: $out"
    
    # Copy log into output dir
    cp /tmp/val_${s}.log $out/val.log
    
    # parse mAP
    map50=$(grep -a '^[[:space:]]*all' /tmp/val_${s}.log | awk '{print $7}')
    map95=$(grep -a '^[[:space:]]*all' /tmp/val_${s}.log | awk '{print $8}')
    echo "mAP50=$map50 mAP50-95=$map95"
    
    # 2. eval_ap
    echo "[2/4] eval_ap.py ..."
    pred=$(find $out -name 'predictions.json' | head -1)
    if [ -n "$pred" ]; then
        $PY $EVAL --gt $GT --pred $pred 2>&1 | tee $out/eval_ap.log
    fi
    
    ap_s=$(grep 'AP_s:' $out/eval_ap.log 2>/dev/null | awk '{print $2}')
    ap_m=$(grep 'AP_m:' $out/eval_ap.log 2>/dev/null | awk '{print $2}')
    ap_l=$(grep 'AP_l:' $out/eval_ap.log 2>/dev/null | awk '{print $2}')
    echo "AP_s=$ap_s AP_m=$ap_m AP_l=$ap_l"
    
    # 3. Params
    echo "[3/4] Params/GFLOPs ..."
    $PY -c "
from ultralytics import YOLO
m = YOLO('$best')
p = sum(p.numel() for p in m.model.parameters())
print(f'Params: {p/1e6:.2f}M')
" 2>&1 | tee $out/params.log
    
    params=$(grep 'Params:' $out/params.log | awk '{print $2}' | sed 's/M//')
    gflops=$(grep -oP '[\d.]+(?= GFLOPs)' /tmp/val_${s}.log | head -1)
    echo "Params=${params}M GFLOPs=$gflops"
    
    # 4. Save summary
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
    echo "Summary:"
    cat $out/summary.txt
done

echo ""
echo "===== ALL DONE ====="
find $PROJECT -name 'summary.txt' -exec echo '{}:' \; -exec cat {} \;
