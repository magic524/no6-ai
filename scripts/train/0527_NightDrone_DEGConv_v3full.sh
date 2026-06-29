#!/bin/bash
# NightDrone DEGConv V3 Full (Backbone + Neck) — 5 scales
# GPU1 (RTX2080Ti), 200 epochs, seed=0 deterministic
YOLO=/home/magic524/miniconda3/envs/no6-ai/bin/yolo

# n-scale
$YOLO detect train data=NightDrone.yaml model=yolo11n-DEGConv-full.yaml epochs=200 batch=16 imgsz=640 device=1 name=DEGConv/NightDrone/yolo11n-V3_full

# s-scale
$YOLO detect train data=NightDrone.yaml model=yolo11s-DEGConv-full.yaml epochs=200 batch=16 imgsz=640 device=1 name=DEGConv/NightDrone/yolo11s-V3_full

# m-scale
$YOLO detect train data=NightDrone.yaml model=yolo11m-DEGConv-full.yaml epochs=200 batch=8 imgsz=640 device=1 name=DEGConv/NightDrone/yolo11m-V3_full

# l-scale
$YOLO detect train data=NightDrone.yaml model=yolo11l-DEGConv-full.yaml epochs=200 batch=8 imgsz=640 device=1 name=DEGConv/NightDrone/yolo11l-V3_full

# x-scale
$YOLO detect train data=NightDrone.yaml model=yolo11x-DEGConv-full.yaml epochs=200 batch=4 imgsz=640 device=1 name=DEGConv/NightDrone/yolo11x-V3_full

echo "=== NightDrone DEGConv V3 Full 五尺度全部完成 ==="
