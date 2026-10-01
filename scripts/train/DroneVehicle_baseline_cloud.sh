#!/bin/bash
# DroneVehicle YOLO11 Baseline — n/s/m/l/x 串行
# Cloud1 RTX 3090 (24GB), device=0
# epochs=200, patience=100, imgsz=640, data=DroneVehicle.yaml
# 18K train images — 早停 patience 避免无效等待

set -e
cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=DroneVehicle.yaml
DEVICE=0
EPOCHS=200
PATIENCE=100
IMSZ=640

echo "=========================================="
echo "DroneVehicle YOLO11 Baseline — 串行训练"
echo "Start: $(date)"
echo "GPU: RTX 3090 24GB"
echo "=========================================="

# n — batch 16
echo "[1/5] yolo11n — batch 16"
$YOLO train model=yolo11n.yaml data=$DATA epochs=$EPOCHS patience=$PATIENCE batch=16 device=$DEVICE imgsz=$IMSZ \
  name=Baseline_Model_Experiment/DroneVehicle/yolo11n
echo "yolo11n DONE at $(date)"
echo ""

# s — batch 16
echo "[2/5] yolo11s — batch 16"
$YOLO train model=yolo11s.yaml data=$DATA epochs=$EPOCHS patience=$PATIENCE batch=16 device=$DEVICE imgsz=$IMSZ \
  name=Baseline_Model_Experiment/DroneVehicle/yolo11s
echo "yolo11s DONE at $(date)"
echo ""

# m — batch 8
echo "[3/5] yolo11m — batch 8"
$YOLO train model=yolo11m.yaml data=$DATA epochs=$EPOCHS patience=$PATIENCE batch=8 device=$DEVICE imgsz=$IMSZ \
  name=Baseline_Model_Experiment/DroneVehicle/yolo11m
echo "yolo11m DONE at $(date)"
echo ""

# l — batch 8
echo "[4/5] yolo11l — batch 8"
$YOLO train model=yolo11l.yaml data=$DATA epochs=$EPOCHS patience=$PATIENCE batch=8 device=$DEVICE imgsz=$IMSZ \
  name=Baseline_Model_Experiment/DroneVehicle/yolo11l
echo "yolo11l DONE at $(date)"
echo ""

# x — batch 4
echo "[5/5] yolo11x — batch 4"
$YOLO train model=yolo11x.yaml data=$DATA epochs=$EPOCHS patience=$PATIENCE batch=4 device=$DEVICE imgsz=$IMSZ \
  name=Baseline_Model_Experiment/DroneVehicle/yolo11x
echo "yolo11x DONE at $(date)"
echo ""

echo "=========================================="
echo "DroneVehicle YOLO11 Baseline 全部完成！"
echo "End: $(date)"
echo "=========================================="
