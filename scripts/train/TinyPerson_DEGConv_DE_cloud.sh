#!/bin/bash
# DEGConv_DE TinyPerson 全尺度训练
# Machine: Cloud-1 (port 44908, RTX 3090 24GB)

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
EPOCHS=200
IMSZ=640
DEVICE=0
DATA=TinyPerson.yaml

# ========== TinyPerson n→s→m→l→x ==========
echo "========== TinyPerson =========="

$YOLO detect train model=yolo11n-DEGConv_DE.yaml data=$DATA epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE name=DEGConv_DE/TinyPerson/yolo11n
$YOLO detect train model=yolo11s-DEGConv_DE.yaml data=$DATA epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE name=DEGConv_DE/TinyPerson/yolo11s
$YOLO detect train model=yolo11m-DEGConv_DE.yaml data=$DATA epochs=$EPOCHS batch=8  imgsz=$IMSZ device=$DEVICE name=DEGConv_DE/TinyPerson/yolo11m
$YOLO detect train model=yolo11l-DEGConv_DE.yaml data=$DATA epochs=$EPOCHS batch=8  imgsz=$IMSZ device=$DEVICE name=DEGConv_DE/TinyPerson/yolo11l
$YOLO detect train model=yolo11x-DEGConv_DE.yaml data=$DATA epochs=$EPOCHS batch=4  imgsz=$IMSZ device=$DEVICE name=DEGConv_DE/TinyPerson/yolo11x

echo "========== ALL DONE =========="
