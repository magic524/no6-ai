#!/bin/bash
# val + eval_ap + Params for BinaryAttentionV1 NightDrone all 5 scales
set -euo pipefail

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PYTHON=/root/miniconda3/envs/no6-ai/bin/python
SCRIPTS=/root/autodl-tmp/no6-ai/scripts/analyze
DATA=NightDrone.yaml
device=0
BATCH=32
ROOT=/root/autodl-tmp/no6-ai/runs/detect
GT=/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json

exp_outputs=()

for scale in n s m l x; do
  best="${ROOT}/BinaryAttentionV1/NightDrone/yolo11${scale}/weights/best.pt"
  val_name="val_bav1_nd/yolo11${scale}"
  VAL_DIR="${ROOT}/val/${val_name}"

  echo ""
  echo "=========================================="
  echo "Validating: yolo11${scale}-BinaryAttentionV1 NightDrone"
  echo "=========================================="

  # Clean prior val output
  rm -rf ${VAL_DIR}

  # Step 1: yolo val (capture mAP from stdout)
  echo "[Step 1/4] yolo val save_json ..."
  ${YOLO} val model=${best} data=${DATA} batch=${BATCH} device=${device} imgsz=640 save_json=True \
    project=${ROOT}/val name=${val_name} 2>&1 | tee ${VAL_DIR}/val_stdout.log

  # Extract mAP from the 'all' line
  mAP50=$(grep '^\s*all' ${VAL_DIR}/val_stdout.log | awk '{print $7}')
  mAP50_95=$(grep '^\s*all' ${VAL_DIR}/val_stdout.log | awk '{print $8}')
  echo "Extracted mAP50=${mAP50}, mAP50-95=${mAP50_95}"

  # Find predictions.json
  PRED_JSON=$(find ${VAL_DIR} -name 'predictions.json' 2> /dev/null | head -1)
  if [ -z "${PRED_JSON}" ]; then
    echo "ERROR: predictions.json not found in ${VAL_DIR}, skipping eval_ap"
    AP_S="0"
    AP_M="0"
    AP_L="0"
  else
    echo "Found predictions.json at: ${PRED_JSON}"

    # Step 2: eval_ap.py
    echo "[Step 2/4] eval_ap.py ..."
    ${PYTHON} ${SCRIPTS}/eval_ap.py --gt ${GT} --pred ${PRED_JSON} 2>&1 | tee ${VAL_DIR}/eval_ap_result.log

    AP_S=$(grep 'AP_s:' ${VAL_DIR}/eval_ap_result.log | awk '{print $2}')
    AP_M=$(grep 'AP_m:' ${VAL_DIR}/eval_ap_result.log | awk '{print $2}')
    AP_L=$(grep 'AP_l:' ${VAL_DIR}/eval_ap_result.log | awk '{print $2}')
  fi

  # Step 3: model Params/GFLOPs
  echo "[Step 3/4] Params/GFLOPs ..."
  ${PYTHON} -c "
from ultralytics import YOLO
model = YOLO('${best}')
params = sum(p.numel() for p in model.model.parameters())
print(f'Params: {params}')
" 2>&1 | tee ${VAL_DIR}/val_params.log

  # Get GFLOPs from the yolo summary line
  GFLOPS=$(grep 'GFLOPs' ${VAL_DIR}/val_stdout.log | grep -oP '[\d.]+(?= GFLOPs)')
  PARAMS_M=$(grep 'Params:' ${VAL_DIR}/val_params.log | awk '{print $2}')
  # Convert params to millions
  PARAMS_M=$(python3 -c "print(f'{${PARAMS_M}/1e6:.2f}')")

  # Save summary
  cat > ${VAL_DIR}/summary.txt << EOF
SCALE=${scale}
mAP50=${mAP50}
mAP50-95=${mAP50_95}
AP_s=${AP_S}
AP_m=${AP_M}
AP_l=${AP_L}
Params(M)=${PARAMS_M}
GFLOPs=${GFLOPS}
EOF

  echo "Summary:"
  cat ${VAL_DIR}/summary.txt

  exp_outputs+=("${scale}:done")
done

echo ""
echo "=========================================="
echo "ALL VALIDATION COMPLETE!"
echo "=========================================="
for out in "${exp_outputs[@]}"; do
  echo "  ${out}"
done
