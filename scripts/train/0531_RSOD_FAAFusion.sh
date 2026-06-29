#!/bin/bash
# RSOD FAAFusion 重跑（五尺度）—— 之前命令写错全部无效
# GPU0 (RTX3080), 200 epochs, 正确的 model=yolo11{s/m/l/x}-FAAFusion.yaml
YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo

$YOLO detect train data=RSOD.yaml model=yolo11n-FAAFusion.yaml epochs=200 batch=16 imgsz=640 device=0 name=FAAFusion/RSOD/yolo11n

$YOLO detect train data=RSOD.yaml model=yolo11s-FAAFusion.yaml epochs=200 batch=16 imgsz=640 device=0 name=FAAFusion/RSOD/yolo11s

$YOLO detect train data=RSOD.yaml model=yolo11m-FAAFusion.yaml epochs=200 batch=8 imgsz=640 device=0 name=FAAFusion/RSOD/yolo11m

$YOLO detect train data=RSOD.yaml model=yolo11l-FAAFusion.yaml epochs=200 batch=8 imgsz=640 device=0 name=FAAFusion/RSOD/yolo11l

$YOLO detect train data=RSOD.yaml model=yolo11x-FAAFusion.yaml epochs=200 batch=4 imgsz=640 device=0 name=FAAFusion/RSOD/yolo11x

echo "=== RSOD FAAFusion 五尺度重跑全部完成 ==="
