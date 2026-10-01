#!/bin/bash
# TGADetect — RSOD Smoke Test (5-epoch, n-scale)
# 验证 TGADect 能否正常训练：loss 应稳步下降，无 NaN
# Cloud1 RTX 3090, device=0

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=RSOD.yaml
DEVICE=0

echo "[Smoke Test] TGADetect — RSOD yolo11n — 5 epoch"
$YOLO detect train data=$DATA model=yolo11n-TGADetect.yaml batch=16 device=$DEVICE name=TGADetect/RSOD/yolo11n-TGADetect epochs=5

echo ""
echo "=== 结果检查 ==="
csv=runs/detect/TGADetect/RSOD/yolo11n-TGADetect/results.csv
if [ -f "$csv" ]; then
  echo "✅ results.csv 存在"
  echo "--- 最终 epoch ---"
  tail -1 "$csv"
  echo "--- Loss 趋势 ---"
  awk -F',' 'NR>1 {print $1, $3, $4, $5}' "$csv" | head -5
  echo "--- mAP ---"
  awk -F',' 'NR>1 {print "epoch="$1, "mAP50="$8, "mAP50-95="$9}' "$csv"
else
  echo "❌ results.csv 不存在 — 训练可能失败"
fi
echo ""
echo "Smoke Test 完成！$(date)"
