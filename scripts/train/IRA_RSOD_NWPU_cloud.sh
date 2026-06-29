#!/bin/bash
# IRA RSOD + NWPU_VHR-10 全尺度训练
# Machine: Cloud-1 (port 44908, RTX 3090 24GB)

cd /root/autodl-tmp/no6-ai

YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
EPOCHS=200
IMSZ=640
DEVICE=0

# ============================================
# RSOD
# ============================================
echo "========== RSOD =========="

$YOLO detect train model=yolo11n-C3k2_IRA.yaml data=RSOD.yaml epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE name=IRA/RSOD/yolo11n
$YOLO detect train model=yolo11s-C3k2_IRA.yaml data=RSOD.yaml epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE name=IRA/RSOD/yolo11s
$YOLO detect train model=yolo11m-C3k2_IRA.yaml data=RSOD.yaml epochs=$EPOCHS batch=8  imgsz=$IMSZ device=$DEVICE name=IRA/RSOD/yolo11m
$YOLO detect train model=yolo11l-C3k2_IRA.yaml data=RSOD.yaml epochs=$EPOCHS batch=8  imgsz=$IMSZ device=$DEVICE name=IRA/RSOD/yolo11l
$YOLO detect train model=yolo11x-C3k2_IRA.yaml data=RSOD.yaml epochs=$EPOCHS batch=4  imgsz=$IMSZ device=$DEVICE name=IRA/RSOD/yolo11x

# ============================================
# NWPU_VHR-10
# ============================================
echo "========== NWPU_VHR-10 =========="

$YOLO detect train model=yolo11n-C3k2_IRA.yaml data=NWPU_VHR-10.yaml epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE name=IRA/NWPU_VHR-10/yolo11n
$YOLO detect train model=yolo11s-C3k2_IRA.yaml data=NWPU_VHR-10.yaml epochs=$EPOCHS batch=16 imgsz=$IMSZ device=$DEVICE name=IRA/NWPU_VHR-10/yolo11s
$YOLO detect train model=yolo11m-C3k2_IRA.yaml data=NWPU_VHR-10.yaml epochs=$EPOCHS batch=8  imgsz=$IMSZ device=$DEVICE name=IRA/NWPU_VHR-10/yolo11m
$YOLO detect train model=yolo11l-C3k2_IRA.yaml data=NWPU_VHR-10.yaml epochs=$EPOCHS batch=8  imgsz=$IMSZ device=$DEVICE name=IRA/NWPU_VHR-10/yolo11l
$YOLO detect train model=yolo11x-C3k2_IRA.yaml data=NWPU_VHR-10.yaml epochs=$EPOCHS batch=4  imgsz=$IMSZ device=$DEVICE name=IRA/NWPU_VHR-10/yolo11x