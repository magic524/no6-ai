#!/bin/bash
# Smoke Test — DEGConv_DE_DyFusFuse — RSOD yolo11n 5-epoch
# Cloud2 (RTX 3090, device=0)
# 验证：loss 正常下降、无 NaN、mAP 正常上升

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=RSOD.yaml
DEVICE=0
MODULE=DEGConv_DE_DyFusFuse

echo "[Smoke Test] $MODULE — $DATA yolo11n — 5 epoch"
$YOLO detect train data=$DATA model=yolo11n-$MODULE.yaml batch=16 device=$DEVICE name=$MODULE/RSOD/yolo11n-$MODULE epochs=5

echo ""
echo "=== 结果检查 ==="
csv=runs/detect/$MODULE/RSOD/yolo11n-$MODULE/results.csv
if [ -f "$csv" ]; then
  echo "✅ results.csv 存在"
  echo "--- 最终 epoch ---"
  tail -1 "$csv"
  echo "--- Loss 趋势 (epoch box cls dfl) ---"
  awk -F',' 'NR>1 {print $1, $3, $4, $5}' "$csv"
  echo "--- mAP ---"
  awk -F',' 'NR>1 {print "epoch="$1, "mAP50="$8, "mAP50-95="$9}' "$csv"
else
  echo "❌ results.csv 不存在 — 训练可能失败"
fi
echo ""
echo "Smoke Test 完成！$(date)"
