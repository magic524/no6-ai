#!/bin/bash
# Write DEGConv_DE RSOD+NWPU_VHR-10 results to Notion (actual COCO eval results)

cd /home/magic524/projects/no6-ai
PY=python3
SCRIPT=scripts/analyze/notion_write.py

# ========== RSOD ==========

# n-scale
$PY $SCRIPT --name "yolo11n-DEGConv_DE-RSOD" --model "DEGConv_DE" --scale "n" --dataset "RSOD" \
  --map50 0.891 --map50-95 0.580 --ap-s 0.386 --ap-m 0.623 --ap-l 0.628 \
  --fps 77.5 --params 2.85 --gflops 8.0

# s-scale
$PY $SCRIPT --name "yolo11s-DEGConv_DE-RSOD" --model "DEGConv_DE" --scale "s" --dataset "RSOD" \
  --map50 0.908 --map50-95 0.614 --ap-s 0.427 --ap-m 0.660 --ap-l 0.658 \
  --fps 208.3 --params 10.33 --gflops 27.5

# m-scale
$PY $SCRIPT --name "yolo11m-DEGConv_DE-RSOD" --model "DEGConv_DE" --scale "m" --dataset "RSOD" \
  --map50 0.932 --map50-95 0.656 --ap-s 0.467 --ap-m 0.686 --ap-l 0.702 \
  --fps 63.7 --params 23.50 --gflops 87.0

# l-scale
$PY $SCRIPT --name "yolo11l-DEGConv_DE-RSOD" --model "DEGConv_DE" --scale "l" --dataset "RSOD" \
  --map50 0.934 --map50-95 0.657 --ap-s 0.492 --ap-m 0.692 --ap-l 0.700 \
  --fps 39.7 --params 28.04 --gflops 105.7

# x-scale
$PY $SCRIPT --name "yolo11x-DEGConv_DE-RSOD" --model "DEGConv_DE" --scale "x" --dataset "RSOD" \
  --map50 0.941 --map50-95 0.646 --ap-s 0.498 --ap-m 0.700 --ap-l 0.686 \
  --fps 35.7 --params 62.69 --gflops 235.6

# ========== NWPU_VHR-10 ==========

# n-scale
$PY $SCRIPT --name "yolo11n-DEGConv_DE-NWPU_VHR-10" --model "DEGConv_DE" --scale "n" --dataset "NWPU_VHR-10" \
  --map50 0.819 --map50-95 0.495 --ap-s 0.078 --ap-m 0.421 --ap-l 0.532 \
  --fps 285.7 --params 2.85 --gflops 8.0

# s-scale
$PY $SCRIPT --name "yolo11s-DEGConv_DE-NWPU_VHR-10" --model "DEGConv_DE" --scale "s" --dataset "NWPU_VHR-10" \
  --map50 0.853 --map50-95 0.532 --ap-s 0.139 --ap-m 0.463 --ap-l 0.572 \
  --fps 250.0 --params 10.33 --gflops 27.5

# m-scale
$PY $SCRIPT --name "yolo11m-DEGConv_DE-NWPU_VHR-10" --model "DEGConv_DE" --scale "m" --dataset "NWPU_VHR-10" \
  --map50 0.863 --map50-95 0.562 --ap-s 0.220 --ap-m 0.498 --ap-l 0.599 \
  --fps 69.0 --params 23.50 --gflops 87.0

# l-scale
$PY $SCRIPT --name "yolo11l-DEGConv_DE-NWPU_VHR-10" --model "DEGConv_DE" --scale "l" --dataset "NWPU_VHR-10" \
  --map50 0.858 --map50-95 0.564 --ap-s 0.219 --ap-m 0.497 --ap-l 0.592 \
  --fps 51.5 --params 28.04 --gflops 105.7

# x-scale
$PY $SCRIPT --name "yolo11x-DEGConv_DE-NWPU_VHR-10" --model "DEGConv_DE" --scale "x" --dataset "NWPU_VHR-10" \
  --map50 0.873 --map50-95 0.570 --ap-s 0.092 --ap-m 0.517 --ap-l 0.615 \
  --fps 44.6 --params 62.69 --gflops 235.6

echo "All 10 entries written to Notion"
