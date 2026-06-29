#!/bin/bash
# TinyPerson IRA (C3k2_IRA) — n/s/m/l/x 串行
# Cloud1 RTX 3090 (24GB), device=0
# epochs=200, imgsz=640, data=TinyPerson.yaml

set -e
cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=TinyPerson.yaml
DEVICE=0
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "TinyPerson IRA (C3k2_IRA) — 串行训练"
echo "Start: $(date)"
echo "=========================================="

# n — batch 16
echo "[1/5] yolo11n-C3k2_IRA — batch 16"
$YOLO train model=yolo11n-C3k2_IRA.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
    name=IRA/TinyPerson/yolo11n
echo "yolo11n DONE at $(date)"
echo ""

# s — batch 16
echo "[2/5] yolo11s-C3k2_IRA — batch 16"
$YOLO train model=yolo11s-C3k2_IRA.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
    name=IRA/TinyPerson/yolo11s
echo "yolo11s DONE at $(date)"
echo ""

# m — batch 8
echo "[3/5] yolo11m-C3k2_IRA — batch 8"
$YOLO train model=yolo11m-C3k2_IRA.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
    name=IRA/TinyPerson/yolo11m
echo "yolo11m DONE at $(date)"
echo ""

# l — batch 8
echo "[4/5] yolo11l-C3k2_IRA — batch 8"
$YOLO train model=yolo11l-C3k2_IRA.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
    name=IRA/TinyPerson/yolo11l
echo "yolo11l DONE at $(date)"
echo ""

# x — batch 4
echo "[5/5] yolo11x-C3k2_IRA — batch 4"
$YOLO train model=yolo11x-C3k2_IRA.yaml data=$DATA epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
    name=IRA/TinyPerson/yolo11x
echo "yolo11x DONE at $(date)"
echo ""

echo "=========================================="
echo "TinyPerson IRA (C3k2_IRA) 全部完成！"
echo "End: $(date)"
echo "=========================================="
