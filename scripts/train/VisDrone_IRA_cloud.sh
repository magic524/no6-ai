#!/bin/bash
# ============================================================
# VisDrone_IRA_cloud.sh
# Purpose: Train yolo11-C3k2_IRA on VisDrone for all scales
# Machine: Cloud-1 (RTX 3090, 24GB, port 44908)
# Env: no6-ai
# ============================================================

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=VisDrone.yaml
DEVICE=0
EPOCHS=200
IMSZ=640

# === 1. n-scale (batch=16) ===
$YOLO detect train model=yolo11n-C3k2_IRA.yaml data=$DATA \
  epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE \
  name=IRA/VisDrone/yolo11n

# === 2. s-scale (batch=16) ===
$YOLO detect train model=yolo11s-C3k2_IRA.yaml data=$DATA \
  epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE \
  name=IRA/VisDrone/yolo11s

# === 3. m-scale (batch=8) ===
$YOLO detect train model=yolo11m-C3k2_IRA.yaml data=$DATA \
  epochs=$EPOCHS batch=8 imgsz=$IMSZ device=$DEVICE \
  name=IRA/VisDrone/yolo11m

# === 4. l-scale (batch=8) ===
$YOLO detect train model=yolo11l-C3k2_IRA.yaml data=$DATA \
  epochs=$EPOCHS batch=8 imgsz=$IMSZ device=$DEVICE \
  name=IRA/VisDrone/yolo11l

# === 5. x-scale (batch=4) ===
$YOLO detect train model=yolo11x-C3k2_IRA.yaml data=$DATA \
  epochs=$EPOCHS batch=4 imgsz=$IMSZ device=$DEVICE \
  name=IRA/VisDrone/yolo11x
