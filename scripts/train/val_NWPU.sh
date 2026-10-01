#!/bin/bash
# NWPU_VHR-10 验证脚本：val + AP_s/m/l + Params/GFLOPs
set -e

cd /home/magic524/projects/no6-ai
YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
PY=/home/magic524/miniconda3/envs/no6-ai/bin/python
SCRIPTS=scripts/analyze
DATA=NWPU_VHR-10.yaml
DEVICE=0
BATCH=16

RESULTS_FILE=/tmp/nwpu_val_results.txt
> $RESULTS_FILE

validate() {
  local model=$1   # yolo11n, yolo11s, etc.
  local module=$2  # DEGConv or FAAFusion
  local variant=$3 # V3_full or FAAFusion
  local exp_name="${model}-${variant}-NWPU_VHR-10"

  echo "========== $exp_name ==========" | tee -a $RESULTS_FILE

  if [ "$module" = "DEGConv" ]; then
    BEST="runs/detect/DEGConv/NWPU_VHR-10/${model}-${variant}/weights/best.pt"
  else
    BEST="runs/detect/FAAFusion/NWPU_VHR-10/${model}-FAAFusion/weights/best.pt"
  fi

  if [ ! -f "$BEST" ]; then
    echo "SKIP: $BEST not found" | tee -a $RESULTS_FILE
    return
  fi

  OUT_DIR="runs/detect/Val/${exp_name}"

  # Step 1: yolo val
  echo "[1/4] yolo val ..." | tee -a $RESULTS_FILE
  $YOLO val model=$BEST data=$DATA batch=$BATCH device=$DEVICE imgsz=640 save_json=True \
    name=Val/${exp_name} 2>&1 | tee $OUT_DIR/val.log

  # Parse mAP50/mAP50-95/FPS from val log
  mAP50=$(grep -oP 'all.*mAP50.*?(\d+\.\d+)' $OUT_DIR/val.log | grep -oP '\d+\.\d+' | tail -1)
  mAP50_95=$(grep -oP 'all.*mAP50-95.*?(\d+\.\d+)' $OUT_DIR/val.log | grep -oP '\d+\.\d+' | tail -1)
  FPS=$(grep 'Speed' $OUT_DIR/val.log | grep -oP '\d+\.\d+ms' | tail -1 | sed 's/ms//')
  echo "mAP50=$mAP50 mAP50-95=$mAP50_95 FPS=${FPS}ms" | tee -a $RESULTS_FILE

  # Step 2: eval_ap.py
  echo "[2/4] eval_ap.py ..." | tee -a $RESULTS_FILE
  PRED_JSON=$(find $OUT_DIR -name 'predictions.json' | head -1)
  if [ -n "$PRED_JSON" ]; then
    $PY $SCRIPTS/eval_ap.py --prediction_path $PRED_JSON \
      --data_path ultralytics/cfg/datasets/NWPU_VHR-10.yaml \
      --mode yolo 2>&1 | tee $OUT_DIR/eval_ap.log

    AP_S=$(grep 'AP_s:' $OUT_DIR/eval_ap.log | awk '{print $2}')
    AP_M=$(grep 'AP_m:' $OUT_DIR/eval_ap.log | awk '{print $2}')
    AP_L=$(grep 'AP_l:' $OUT_DIR/eval_ap.log | awk '{print $2}')
    echo "AP_s=$AP_S AP_m=$AP_M AP_l=$AP_L" | tee -a $RESULTS_FILE
  else
    echo "WARNING: predictions.json not found" | tee -a $RESULTS_FILE
    AP_S="?"
    AP_M="?"
    AP_L="?"
  fi

  # Step 3: Params/GFLOPs
  echo "[3/4] Params/GFLOPs ..." | tee -a $RESULTS_FILE
  $PY -c "
from ultralytics import YOLO
m = YOLO('$BEST')
info = m.info()
print(f'Params(M): {info[0]:.2f}')
print(f'GFLOPs: {info[1]:.2f}')
" 2>&1 | tee -a $RESULTS_FILE

  PARAMS=$(grep 'Params(M):' $OUT_DIR/val.log | grep -oP '\d+\.\d+' | head -1)
  GFLOPS=$(grep 'GFLOPs:' $OUT_DIR/val.log | grep -oP '\d+\.\d+' | head -1)

  # Step 4: Notion (placeholder — will batch write later)
  echo "exp_name=$exp_name model=${model}+${variant} scale=${model#yolo11} dataset=NWPU_VHR-10 mAP50=$mAP50 mAP50-95=$mAP50_95 AP_s=$AP_S AP_m=$AP_M AP_l=$AP_L FPS=$FPS Params=$PARAMS GFLOPs=$GFLOPS" >> /tmp/nwpu_notion_entries.txt
  echo "[4/4] Done" | tee -a $RESULTS_FILE
  echo "" | tee -a $RESULTS_FILE
}

mkdir -p runs/detect/Val

# DEGConv V3 — s/m/l/x (n 已有)
for s in s m l x; do
  validate "yolo11${s}" "DEGConv" "V3_full"
done

# FAAFusion — n/s/m/l/x
for s in n s m l x; do
  validate "yolo11${s}" "FAAFusion" "FAAFusion"
done

echo ""
echo "========== 全部验证完成 =========="
cat /tmp/nwpu_notion_entries.txt
echo ""
echo "结果摘要已写入 /tmp/nwpu_notion_entries.txt"
