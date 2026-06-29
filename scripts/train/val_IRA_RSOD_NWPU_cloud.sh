#!/bin/bash
# IRA RSOD + NWPU_VHR-10 全尺度 val + eval_ap + Params/GFLOPs
# 在云1 (port 44908) 上运行

set -o pipefail

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PYTHON=/root/miniconda3/envs/no6-ai/bin/python
SCRIPTS=/root/autodl-tmp/no6-ai/scripts/analyze
ROOT=/root/autodl-tmp/no6-ai/runs/detect/IRA
device=0
BATCH=16

run_val() {
    local dataset=$1      # RSOD or NWPU_VHR-10
    local scale=$2        # n s m l x
    local best="${ROOT}/${dataset}/yolo11${scale}/weights/best.pt"
    local val_name="val_ira_${dataset}/yolo11${scale}"
    local VAL_DIR="${ROOT}/../val/${val_name}"
    local GT="/root/autodl-tmp/datasets/${dataset}/annotations/instances_val.json"
    local DATA="${dataset}.yaml"
    
    # Set batch based on scale
    local val_batch=$BATCH
    if [ "$scale" = "m" ] || [ "$scale" = "l" ]; then
        val_batch=8
    elif [ "$scale" = "x" ]; then
        val_batch=4
    fi
    
    echo ""
    echo "============================================"
    echo "Val: yolo11${scale} IRA on ${dataset}"
    echo "============================================"
    
    # Don't pre-create dir — let yolo create it to avoid auto-rename (-2)
    rm -rf ${ROOT}/../val/val_ira_${dataset}/yolo11${scale}*
    
    # Step 1: yolo val (use temp log to avoid yolo seeing pre-existing dir)
    local TMPLOG=$(mktemp)
    echo "[1/4] yolo val save_json ..."
    ${YOLO} val model=${best} data=${DATA} batch=${val_batch} device=${device} imgsz=640 save_json=True \
        project=${ROOT}/../val name=${val_name} 2>&1 | tee ${TMPLOG}
    
    # Find the actual val output dir (yolo creates it, may be yolo11n or yolo11n-2)
    local VAL_DIR_ACTUAL=$(find ${ROOT}/../val/val_ira_${dataset} -maxdepth 1 -name "yolo11${scale}*" -type d 2>/dev/null | head -1)
    if [ -z "$VAL_DIR_ACTUAL" ]; then
        echo "ERROR: Cannot find val output dir for yolo11${scale}"
        return 1
    fi
    VAL_DIR="$VAL_DIR_ACTUAL"
    # Copy val_stdout log into VAL_DIR for reference
    cp ${TMPLOG} ${VAL_DIR}/val_stdout.log
    rm ${TMPLOG}
    
    # Extract mAP (columns: all images instances P R mAP50 mAP50-95)
    mAP50=$(grep '^\s*all' ${VAL_DIR}/val_stdout.log | awk '{print $6}')
    mAP50_95=$(grep '^\s*all' ${VAL_DIR}/val_stdout.log | awk '{print $7}')
    echo "mAP50=${mAP50}, mAP50-95=${mAP50_95}"
    
    # Step 2: eval_ap.py
    echo "[2/4] eval_ap.py ..."
    PRED_JSON=$(find ${VAL_DIR} -name 'predictions.json' 2>/dev/null | head -1)
    AP_S="0"; AP_M="0"; AP_L="0"
    if [ -n "${PRED_JSON}" ]; then
        ${PYTHON} ${SCRIPTS}/eval_ap.py --gt ${GT} --pred ${PRED_JSON} 2>&1 | tee ${VAL_DIR}/eval_ap_result.log
        AP_S=$(grep -w 'AP_s' ${VAL_DIR}/eval_ap_result.log | awk '{print $2}')
        AP_M=$(grep -w 'AP_m' ${VAL_DIR}/eval_ap_result.log | awk '{print $2}')
        AP_L=$(grep -w 'AP_l' ${VAL_DIR}/eval_ap_result.log | awk '{print $2}')
        echo "AP_s=${AP_S} AP_m=${AP_M} AP_l=${AP_L}"
    else
        echo "WARNING: predictions.json not found"
    fi
    
    # Step 3: Params
    echo "[3/4] Params ..."
    ${PYTHON} -c "
from ultralytics import YOLO
model = YOLO('${best}')
params = sum(p.numel() for p in model.model.parameters())
print(f'Params: {params}')
" 2>&1 | tee ${VAL_DIR}/val_params.log
    
    GFLOPS=$(grep 'GFLOPs' ${VAL_DIR}/val_stdout.log | grep -oP '[\d.]+(?= GFLOPs)' || echo "?")
    PARAMS=$(grep 'Params:' ${VAL_DIR}/val_params.log | awk '{print $2}' || echo "0")
    PARAMS_M=$(${PYTHON} -c "
params = ${PARAMS}
print(f'{params/1e6:.2f}')" 2>/dev/null || echo "?")
    
    # Write summary
    cat > ${VAL_DIR}/summary.txt << EOF
DATASET=${dataset}
SCALE=${scale}
mAP50=${mAP50}
mAP50-95=${mAP50_95}
AP_s=${AP_S}
AP_m=${AP_M}
AP_l=${AP_L}
Params(M)=${PARAMS_M}
GFLOPs=${GFLOPS}
EOF
    echo "Summary saved to ${VAL_DIR}/summary.txt"
    cat ${VAL_DIR}/summary.txt
    echo ""
}

echo "============================================"
echo "Starting IRA val: RSOD + NWPU_VHR-10"
echo "============================================"

# RSOD: n s m l x
for s in n s m l x; do
    run_val "RSOD" "$s"
done

# NWPU_VHR-10: n s m l x
for s in n s m l x; do
    run_val "NWPU_VHR-10" "$s"
done

echo ""
echo "============================================"
echo "ALL VALIDATION COMPLETE!"
echo "============================================"

# Print summary of all results
echo ""
echo "=== RESULTS SUMMARY ==="
for dataset in RSOD NWPU_VHR-10; do
    echo "--- ${dataset} ---"
    for s in n s m l x; do
        SUM="${ROOT}/../val/val_ira_${dataset}/yolo11${s}/summary.txt"
        if [ -f "$SUM" ]; then
            echo "yolo11${s}: $(grep mAP50 $SUM | head -1) | $(grep mAP50-95 $SUM | head -1) | $(grep AP_s $SUM | head -1) | $(grep AP_m $SUM | head -1) | $(grep AP_l $SUM | head -1)"
        fi
    done
done

echo ""
echo "============================================"
echo "FPS MEASUREMENT (GPU0, batch=1, imgsz=640)"
echo "============================================"
for dataset in RSOD NWPU_VHR-10; do
    echo "--- ${dataset} ---"
    for s in n s m l x; do
        best="${ROOT}/${dataset}/yolo11${s}/weights/best.pt"
        if [ -f "$best" ]; then
            fps=$(${PYTHON} -c "
import torch, time
from ultralytics import YOLO
m = YOLO('${best}')
dummy = torch.randn(1, 3, 640, 640).cuda()
for _ in range(30):
    m.predict(dummy, verbose=False)
torch.cuda.synchronize()
start = time.time()
for _ in range(200):
    m.predict(dummy, verbose=False)
torch.cuda.synchronize()
fps = 200 / (time.time() - start)
print(f'{fps:.1f}')" 2>/dev/null || echo "?")
            echo "yolo11${s}: ${fps} FPS"
        fi
    done
done

echo ""
echo "============================================"
echo "ALL DONE!"
echo "============================================"
