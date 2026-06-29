#!/bin/bash
# TinyPerson DEGConv V3 Full (Backbone + Neck) — s/m/l/x scales
# GPU1 (RTX2080Ti), 200 epochs, seed=0 deterministic
YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo

# s-scale
$YOLO detect train data=TinyPerson.yaml model=yolo11s-DEGConv-full.yaml epochs=200 batch=16 imgsz=640 device=1 name=DEGConv/TinyPerson/yolo11s-V3_full

# m-scale
$YOLO detect train data=TinyPerson.yaml model=yolo11m-DEGConv-full.yaml epochs=200 batch=8 imgsz=640 device=1 name=DEGConv/TinyPerson/yolo11m-V3_full

# l-scale
$YOLO detect train data=TinyPerson.yaml model=yolo11l-DEGConv-full.yaml epochs=200 batch=8 imgsz=640 device=1 name=DEGConv/TinyPerson/yolo11l-V3_full

# x-scale
$YOLO detect train data=TinyPerson.yaml model=yolo11x-DEGConv-full.yaml epochs=200 batch=4 imgsz=640 device=1 name=DEGConv/TinyPerson/yolo11x-V3_full

echo "=== TinyPerson DEGConv V3 Full s/m/l/x 全部完成 ==="
