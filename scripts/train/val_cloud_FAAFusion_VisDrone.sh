#!/bin/bash
# 云端验证 FAAFusion VisDrone 结果
# 4步：val → eval_ap → Params → Notion
set -e

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
PYTHON=/root/miniconda3/envs/no6-ai/bin/python
SCRIPTS=/root/autodl-tmp/no6-ai/scripts/analyze
DATA=VisDrone.yaml
DEVICE=0
BATCH=16
ROOT=/root/autodl-tmp

# 各实验路径：n 单独目录，s/m/l/x 在 VisDrone/ 下
EXPS=(
  "n:$ROOT/no6-ai/runs/detect/FAAFusion/yolo11n-FAAFusion_VisDrone"
  "s:$ROOT/no6-ai/runs/detect/FAAFusion/VisDrone/yolo11s"
  "m:$ROOT/no6-ai/runs/detect/FAAFusion/VisDrone/yolo11m"
  "l:$ROOT/no6-ai/runs/detect/FAAFusion/VisDrone/yolo11l"
  "x:$ROOT/no6-ai/runs/detect/FAAFusion/VisDrone/yolo11x"
)

for exp in "${EXPS[@]}"; do
  scale="${exp%%:*}"
  path="${exp##*:}"
  name="FAAFusion/VisDrone/yolo11${scale}"
  best="$path/weights/best.pt"

  echo ""
  echo "=========================================="
  echo "验证: yolo11${scale}-FAAFusion VisDrone"
  echo "=========================================="

  # Step 1: yolo val
  echo "[Step 1/4] yolo val ..."
  $YOLO val model=$best data=$DATA batch=$BATCH device=$DEVICE imgsz=640 save_json=True \
    name=$name

  # Step 2: eval_ap.py (需要 predictions.json)
  echo "[Step 2/4] eval_ap.py ..."
  # predictions.json 应该在 val/ 下
  VAL_DIR="$ROOT/no6-ai/runs/detect/$name"
  PRED_JSON=$(find $VAL_DIR -name 'predictions.json' 2> /dev/null | head -1)
  if [ -n "$PRED_JSON" ]; then
    $PYTHON $SCRIPTS/eval_ap.py --prediction_path $PRED_JSON \
      --data_path $ROOT/no6-ai/ultralytics/cfg/datasets/VisDrone.yaml \
      --mode yolo 2>&1 | tee $VAL_DIR/eval_ap_result.log
  else
    echo "WARNING: predictions.json not found in $VAL_DIR"
  fi

  # Step 3: model Params/GFLOPs
  echo "[Step 3/4] Params/GFLOPs ..."
  $PYTHON -c "
import torch
from ultralytics import YOLO
model = YOLO('$best')
info = model.info()
print(f'Params(M): {info[0]:.2f}')
print(f'GFLOPs: {info[1]:.2f}')
" 2>&1 | tee -a $VAL_DIR/val_results.log

  # Step 4: Notion 写入
  echo "[Step 4/4] Notion 写入 ..."
  # 从 results.csv 提取 mAP
  mAP50=$(tail -1 $VAL_DIR/results.csv | cut -d',' -f8)
  mAP50_95=$(tail -1 $VAL_DIR/results.csv | cut -d',' -f9)
  # 从 eval_ap_result.log 提取 AP_s/m/l
  AP_S=$(grep 'AP_s:' $VAL_DIR/eval_ap_result.log 2> /dev/null | awk '{print \$2}')
  AP_M=$(grep 'AP_m:' $VAL_DIR/eval_ap_result.log 2> /dev/null | awk '{print \$2}')
  AP_L=$(grep 'AP_l:' $VAL_DIR/eval_ap_result.log 2> /dev/null | awk '{print \$2}')
  FPS=$(grep 'Speed' $VAL_DIR/val_results.log 2> /dev/null | head -1)

  $PYTHON $SCRIPTS/notion_write.py \
    --name "yolo11${scale}-FAAFusion-VisDrone" \
    --dataset "VisDrone" \
    --model "yolo11${scale}+FAAFusion" \
    --scale "${scale}" \
    --mAP50 "$mAP50" \
    --mAP50-95 "$mAP50_95" \
    --AP_s "$AP_S" \
    --AP_m "$AP_M" \
    --AP_l "$AP_L" \
    --FPS "?" \
    --Params "?" \
    --GFLOPs "?"

  echo "yolo11${scale} 完成"
done

echo ""
echo "=========================================="
echo "全部验证完成！"
echo "=========================================="
