#!/bin/bash
# NightDrone YOLO11 Baseline — n/s/m/l/x 串行
# GPU: device=1 (RTX 2080 Ti, 22.5GB, 4764 train images > 1000)
# epochs=200, imgsz=640, data=NightDrone.yaml

set -e
cd /home/magic524/projects/no6-ai

YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo
DATA=NightDrone.yaml
DEVICE=1
EPOCHS=200
IMSZ=640

echo "=========================================="
echo "NightDrone YOLO11 Baseline — 串行训练"
echo "Start: $(date)"
echo "=========================================="

# n — batch 16
echo "[1/5] yolo11n — batch 16"
$YOLO train model=yolo11n.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
    name=Baseline_Model_Experiment/NightDrone/yolo11n
echo "yolo11n DONE at $(date)"
echo ""

# s — batch 16
echo "[2/5] yolo11s — batch 16"
$YOLO train model=yolo11s.yaml data=$DATA epochs=$EPOCHS batch=16 device=$DEVICE imgsz=$IMSZ \
    name=Baseline_Model_Experiment/NightDrone/yolo11s
echo "yolo11s DONE at $(date)"
echo ""

# m — batch 8
echo "[3/5] yolo11m — batch 8"
$YOLO train model=yolo11m.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
    name=Baseline_Model_Experiment/NightDrone/yolo11m
echo "yolo11m DONE at $(date)"
echo ""

# l — batch 8
echo "[4/5] yolo11l — batch 8"
$YOLO train model=yolo11l.yaml data=$DATA epochs=$EPOCHS batch=8 device=$DEVICE imgsz=$IMSZ \
    name=Baseline_Model_Experiment/NightDrone/yolo11l
echo "yolo11l DONE at $(date)"
echo ""

# x — batch 4
echo "[5/5] yolo11x — batch 4"
$YOLO train model=yolo11x.yaml data=$DATA epochs=$EPOCHS batch=4 device=$DEVICE imgsz=$IMSZ \
    name=Baseline_Model_Experiment/NightDrone/yolo11x
echo "yolo11x DONE at $(date)"
echo ""

echo "=========================================="
echo "NightDrone YOLO11 Baseline 全部完成！"
echo "End: $(date)"
echo "=========================================="
