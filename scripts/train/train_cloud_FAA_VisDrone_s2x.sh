#!/bin/bash
# 云端 FAAFusion VisDrone s/m/l/x
# GPU0 (RTX 3090, 24GB)
# model=yolo11{s/m/l/x}-FAAFusion.yaml ← 正确格式！不用路径！
set -e

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "FAAFusion VisDrone s/m/l/x"
echo "Start: $(date)"
echo "=========================================="

# s — batch 16
echo "[1/4] yolo11s-FAAFusion — batch 16"
$YOLO train model=yolo11s-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
    name=FAAFusion/VisDrone/yolo11s
echo "yolo11s-FAAFusion DONE at $(date)"
echo ""

# m — batch 8
echo "[2/4] yolo11m-FAAFusion — batch 8"
$YOLO train model=yolo11m-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
    name=FAAFusion/VisDrone/yolo11m
echo "yolo11m-FAAFusion DONE at $(date)"
echo ""

# l — batch 8
echo "[3/4] yolo11l-FAAFusion — batch 8"
$YOLO train model=yolo11l-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
    name=FAAFusion/VisDrone/yolo11l
echo "yolo11l-FAAFusion DONE at $(date)"
echo ""

# x — batch 4
echo "[4/4] yolo11x-FAAFusion — batch 4"
$YOLO train model=yolo11x-FAAFusion.yaml data=$DATA epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
    name=FAAFusion/VisDrone/yolo11x
echo "yolo11x-FAAFusion DONE at $(date)"
echo ""

echo "=========================================="
echo "FAAFusion VisDrone 全部完成！"
echo "End: $(date)"
echo "=========================================="
