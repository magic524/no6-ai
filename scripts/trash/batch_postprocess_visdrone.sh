#!/bin/bash
# batch_postprocess_visdrone.sh — VisDrone YOLO11 全系列后处理
set -e
CDIR="/home/magic524/projects/no6-ai"
YOLO="/home/magic524/miniconda3/envs/no6-ai/bin/yolo"
PY="/home/magic524/miniconda3/envs/no6-ai/bin/python"
EVAL="$CDIR/eval_ap.py"
NOTION="$CDIR/notion_write.py"
GT="/mnt/e/Datasets/Small_Objects_Dataset/VisDrone/annotations/instances_val.json"
cd "$CDIR"

# 模型参数
declare -A PARAMS GFLOPS SCALES
SCALES=([yolo11n]=n [yolo11s]=s [yolo11m]=m [yolo11l]=l [yolo11x]=x)
PARAMS=([yolo11n]=2.6 [yolo11s]=9.4 [yolo11m]=20.0 [yolo11l]=25.3 [yolo11x]=56.8)
GFLOPS=([yolo11n]=6.3 [yolo11s]=21.3 [yolo11m]=67.7 [yolo11l]=86.6 [yolo11x]=194.5)

for MODEL in yolo11n yolo11s yolo11m yolo11l yolo11x; do
  WT="runs/detect/Baseline_Model_Experiment/VisDrone/$MODEL/weights/best.pt"
  if [ ! -f "$WT" ]; then
    echo "❌ $MODEL 权重不存在，跳过"
    continue
  fi

  EXPERIMENT="${MODEL}-VisDrone"
  VAL_DIR="val/VisDrone/${MODEL}"
  LOG="/tmp/val_${MODEL}_VisDrone.log"
  SCALE="${SCALES[$MODEL]}"
  P="${PARAMS[$MODEL]}"
  G="${GFLOPS[$MODEL]}"

  echo ""
  echo "=========================================="
  echo "▶ $EXPERIMENT  scale=$SCALE  params=${P}M  gflops=${G}G"
  echo "=========================================="

  # Step 1: YOLO val
  echo "[1/4] YOLO val..."
  $YOLO detect val data=VisDrone.yaml model="$WT" device=0 name="$VAL_DIR" 2>&1 | tee "$LOG"

  MAP50=$(awk '/^[[:space:]]*all/{print $(NF-1)}' "$LOG")
  MAP50_95=$(awk '/^[[:space:]]*all/{print $NF}' "$LOG")

  SPEED=$(grep "Speed:" "$LOG" | tail -1)
  PRE=$(echo "$SPEED" | grep -oP '[\d.]+(?=ms preprocess)')
  INF=$(echo "$SPEED" | grep -oP '[\d.]+(?=ms inference)')
  POST=$(echo "$SPEED" | grep -oP '[\d.]+(?=ms postprocess)')
  TOTAL=$(echo "$PRE + $INF + $POST" | bc 2> /dev/null || echo "10")
  FPS=$(echo "scale=1; 1000 / $TOTAL" | bc 2> /dev/null || echo "0")
  echo "  mAP50=$MAP50  mAP50:95=$MAP50_95  FPS=$FPS"

  # Step 2: eval_ap.py
  echo "[2/4] eval_ap.py..."
  PRED_JSON="runs/detect/$VAL_DIR/predictions.json"
  AP_S=0
  AP_M=0
  AP_L=0
  if [ -f "$PRED_JSON" ]; then
    eval_log="/tmp/eval_${MODEL}_VisDrone.log"
    $PY "$EVAL" --gt "$GT" --pred "$PRED_JSON" 2>&1 | tee "$eval_log"
    AP_S=$(grep -oP 'AP_s\s+[\d.]+' "$eval_log" | awk '{print $2}')
    AP_M=$(grep -oP 'AP_m\s+[\d.]+' "$eval_log" | awk '{print $2}')
    AP_L=$(grep -oP 'AP_l\s+[\d.]+' "$eval_log" | awk '{print $2}')
    echo "  AP_s=$AP_S  AP_m=$AP_M  AP_l=$AP_L"
  fi

  # Step 3: Notion
  echo "[3/4] Notion 写入..."
  $PY "$NOTION" \
    --name "$EXPERIMENT" \
    --model "YOLO11" \
    --scale "$SCALE" \
    --dataset "VisDrone" \
    --map50 "$MAP50" \
    --map50-95 "$MAP50_95" \
    --ap-s "$AP_S" \
    --ap-m "$AP_M" \
    --ap-l "$AP_L" \
    --fps "$FPS" \
    --params "$P" \
    --gflops "$G" 2>&1

  echo "✅ $EXPERIMENT 完成"
done

echo ""
echo "=========================================="
echo "🎉 VisDrone YOLO11 全系列后处理完成！"
echo "=========================================="
