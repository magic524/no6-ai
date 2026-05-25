#!/bin/bash
# post_process.sh — 训练完成后自动跑 val + eval_ap + FPS + Notion 录入
# 用法: ./post_process.sh --model yolo11n --dataset VisDrone --weight path/to/best.pt

set -e

CDIR="/home/magic524/projects/no6-ai"
YOLO="/home/magic524/miniconda3/envs/no6-ai/bin/yolo"
PY="/home/magic524/miniconda3/envs/no6-ai/bin/python"
EVAL_AP="$CDIR/eval_ap.py"
NOTION_WRITE="$CDIR/notion_write.py"
DEVICE=0  # val/eval 统一用 GPU0

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --weight) WEIGHT="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --params) PARAMS="$2"; shift 2 ;;
    --gflops) GFLOPS="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
  echo "用法: $0 --model yolo11n --dataset VisDrone --weight runs/.../best.pt [--scale n] [--params 2.6] [--gflops 6.3]"
  exit 1
fi

# ── 数据集 → yaml 映射 ──
declare -A YAMLS
YAMLS[VisDrone]="VisDrone.yaml"
YAMLS[TinyPerson]="TinyPerson.yaml"
YAMLS[RSOD]="RSOD.yaml"
YAMLS[NWPU_VHR-10]="NWPU_VHR-10.yaml"
YAML="${YAMLS[$DATASET]}"

# ── GT 文件路径 ──
declare -A GTS
GTS[VisDrone]="/mnt/e/Datasets/VisDrone/annotations/instances_val.json"
GTS[TinyPerson]="/mnt/e/Datasets/Small_Objects_Dataset/TinyPerson/annotations/instances_val.json"
GTS[RSOD]="/mnt/e/Datasets/RSOD/annotations/instances_val.json"
GTS[NWPU_VHR-10]="/mnt/e/Datasets/NWPU_VHR-10/annotations/instances_val.json"
GT="${GTS[$DATASET]}"

if [[ -z "$YAML" || -z "$GT" ]]; then
  echo "数据集 $DATASET 未在映射表中"
  exit 1
fi

if [[ ! -f "$WEIGHT" ]]; then
  echo "权重文件不存在: $WEIGHT"
  exit 1
fi

cd "$CDIR"
EXPERIMENT_NAME="${MODEL}-${DATASET}"

echo "=========================================="
echo "后处理开始: $EXPERIMENT_NAME"
echo "权重: $WEIGHT"
echo "=========================================="

# ── Step 1: YOLO val ──
echo "[1/4] YOLO val..."
VAL_DIR="val/${DATASET}/${MODEL}"
$YOLO detect val data="$YAML" model="$WEIGHT" device=$DEVICE name="$VAL_DIR" 2>&1 | tee /tmp/val_${MODEL}_${DATASET}.log

# 从 val 输出中提取 mAP
MAP50=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+' /tmp/val_${MODEL}_${DATASET}.log | tail -1 | awk '{print $4}')
MAP50_95=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+' /tmp/val_${MODEL}_${DATASET}.log | tail -1 | awk '{print $5}')

# 从 val 输出中提取速度
SPEED_LINE=$(grep "Speed:" /tmp/val_${MODEL}_${DATASET}.log | tail -1)
PRE=$(echo "$SPEED_LINE" | grep -oP '[\d.]+(?=ms preprocess)')
INF=$(echo "$SPEED_LINE" | grep -oP '[\d.]+(?=ms inference)')
POST=$(echo "$SPEED_LINE" | grep -oP '[\d.]+(?=ms postprocess)')
TOTAL=$(echo "$PRE + $INF + $POST" | bc)
FPS=$(echo "scale=1; 1000 / $TOTAL" | bc)

echo "  mAP50=$MAP50  mAP50:95=$MAP50_95  FPS=$FPS"

# ── Step 2: eval_ap.py (AP_s/m/l) ──
echo "[2/4] eval_ap.py..."
PRED_JSON="$CDIR/runs/detect/$VAL_DIR/predictions.json"
AP_S="0"
AP_M="0"
AP_L="0"
if [[ -f "$PRED_JSON" ]]; then
  $PY "$EVAL_AP" --gt "$GT" --pred "$PRED_JSON" 2>&1 | tee /tmp/eval_${MODEL}_${DATASET}.log
  AP_S=$(grep -oP 'AP_s\s+[0-9.]+' /tmp/eval_${MODEL}_${DATASET}.log | awk '{print $2}')
  AP_M=$(grep -oP 'AP_m\s+[0-9.]+' /tmp/eval_${MODEL}_${DATASET}.log | awk '{print $2}')
  AP_L=$(grep -oP 'AP_l\s+[0-9.]+' /tmp/eval_${MODEL}_${DATASET}.log | awk '{print $2}')
  echo "  AP_s=$AP_S  AP_m=$AP_M  AP_l=$AP_L"
else
  echo "  predictions.json 不存在，跳过 eval_ap"
fi

# ── Step 3: 模型信息（如果没提供则自动推断）──
echo "[3/4] 模型信息..."
if [[ -z "$SCALE" ]]; then
  SCALE=$(echo "$MODEL" | grep -oP '[nmslx]$')
fi
if [[ -z "$PARAMS" || -z "$GFLOPS" ]]; then
  # 从模型文件推断
  PARAMS=0
  GFLOPS=0
fi
echo "  scale=$SCALE  params=$PARAMS  gflops=$GFLOPS"

# ── Step 4: Notion 写入 ──
echo "[4/4] Notion 写入..."
$PY "$NOTION_WRITE" \
  --name "$EXPERIMENT_NAME" \
  --model "YOLO11" \
  --scale "$SCALE" \
  --dataset "$DATASET" \
  --map50 "$MAP50" \
  --map50-95 "$MAP50_95" \
  --ap-s "$AP_S" \
  --ap-m "$AP_M" \
  --ap-l "$AP_L" \
  --fps "$FPS" \
  --params "$PARAMS" \
  --gflops "$GFLOPS"

echo "=========================================="
echo "完成: $EXPERIMENT_NAME"
echo "=========================================="
