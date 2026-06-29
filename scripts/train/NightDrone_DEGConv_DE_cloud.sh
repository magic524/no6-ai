#!/bin/bash
# NightDrone_DEGConv_DE_cloud.sh
# Machine: Cloud-2 (port 10951, RTX 3090 24GB)
# FIXED: model path uses scale suffix so YOLO detects correct scale

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DATA=NightDrone.yaml
DEVICE=0

# n-scale (batch=16)
$YOLO detect train model=yolo11n-DEGConv_DE.yaml data=$DATA epochs=200 batch=16 imgsz=640 device=$DEVICE name=DEGConv_DE/NightDrone/yolo11n

# s-scale (batch=16)
$YOLO detect train model=yolo11s-DEGConv_DE.yaml data=$DATA epochs=200 batch=16 imgsz=640 device=$DEVICE name=DEGConv_DE/NightDrone/yolo11s

# m-scale (batch=8)
$YOLO detect train model=yolo11m-DEGConv_DE.yaml data=$DATA epochs=200 batch=8 imgsz=640 device=$DEVICE name=DEGConv_DE/NightDrone/yolo11m

# l-scale (batch=8)
$YOLO detect train model=yolo11l-DEGConv_DE.yaml data=$DATA epochs=200 batch=8 imgsz=640 device=$DEVICE name=DEGConv_DE/NightDrone/yolo11l

# x-scale (batch=4)
$YOLO detect train model=yolo11x-DEGConv_DE.yaml data=$DATA epochs=200 batch=4 imgsz=640 device=$DEVICE name=DEGConv_DE/NightDrone/yolo11x
