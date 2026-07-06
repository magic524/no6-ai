#!/bin/bash
# C2PSA_AFFN — TinyPerson (Cloud1, RTX 3090)
# Serial: 5 scales × 1 dataset = 5 runs

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DEVICE=0

echo "============================================"
echo "C2PSA_AFFN — TinyPerson (Cloud1) — $(date)"
echo "============================================"

echo "[1/5] yolo11n — TinyPerson — batch 16"
$YOLO detect train data=TinyPerson.yaml model=yolo11n-C2PSA_AFFN.yaml batch=16 device=$DEVICE
echo "yolo11n DONE at $(date)"

echo "[2/5] yolo11s — TinyPerson — batch 16"
$YOLO detect train data=TinyPerson.yaml model=yolo11s-C2PSA_AFFN.yaml batch=16 device=$DEVICE
echo "yolo11s DONE at $(date)"

echo "[3/5] yolo11m — TinyPerson — batch 8"
$YOLO detect train data=TinyPerson.yaml model=yolo11m-C2PSA_AFFN.yaml batch=8 device=$DEVICE
echo "yolo11m DONE at $(date)"

echo "[4/5] yolo11l — TinyPerson — batch 8"
$YOLO detect train data=TinyPerson.yaml model=yolo11l-C2PSA_AFFN.yaml batch=8 device=$DEVICE
echo "yolo11l DONE at $(date)"

echo "[5/5] yolo11x — TinyPerson — batch 4"
$YOLO detect train data=TinyPerson.yaml model=yolo11x-C2PSA_AFFN.yaml batch=4 device=$DEVICE
echo "yolo11x DONE at $(date)"

echo "全部完成！$(date)"