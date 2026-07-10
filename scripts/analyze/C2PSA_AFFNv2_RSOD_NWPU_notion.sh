#!/bin/bash
# C2PSA_AFFNv2 — RSOD + NWPU_VHR-10 → Notion batch write
# Uses REST API directly (MCP fallback)

set -e
PY=/root/miniconda3/envs/no6-ai/bin/python
SCRIPT=/root/autodl-tmp/no6-ai/scripts/analyze/notion_write.py

# ===== RSOD =====
echo "=== RSOD ==="

# n-scale
$PY $SCRIPT --name 'yolo11n-C2PSA_AFFNv2-RSOD' --model C2PSA_AFFNv2 --scale n --dataset RSOD \
  --map50 0.90285 --map50-95 0.58018 --ap-s 0.387 --ap-m 0.633 --ap-l 0.622 \
  --fps 97.9 --params 2.63 --gflops 6.5

# s-scale
$PY $SCRIPT --name 'yolo11s-C2PSA_AFFNv2-RSOD' --model C2PSA_AFFNv2 --scale s --dataset RSOD \
  --map50 0.90966 --map50-95 0.61678 --ap-s 0.445 --ap-m 0.663 --ap-l 0.675 \
  --fps 135.5 --params 9.57 --gflops 21.7

# m-scale
$PY $SCRIPT --name 'yolo11m-C2PSA_AFFNv2-RSOD' --model C2PSA_AFFNv2 --scale m --dataset RSOD \
  --map50 0.89462 --map50-95 0.61305 --ap-s 0.465 --ap-m 0.667 --ap-l 0.678 \
  --fps 75.4 --params 20.20 --gflops 68.3

# l-scale
$PY $SCRIPT --name 'yolo11l-C2PSA_AFFNv2-RSOD' --model C2PSA_AFFNv2 --scale l --dataset RSOD \
  --map50 0.93617 --map50-95 0.65086 --ap-s 0.482 --ap-m 0.691 --ap-l 0.696 \
  --fps 64.0 --params 25.60 --gflops 87.5

# x-scale
$PY $SCRIPT --name 'yolo11x-C2PSA_AFFNv2-RSOD' --model C2PSA_AFFNv2 --scale x --dataset RSOD \
  --map50 0.91483 --map50-95 0.64480 --ap-s 0.475 --ap-m 0.699 --ap-l 0.691 \
  --fps 63.7 --params 57.51 --gflops 196.0

echo "RSOD done"

# ===== NWPU_VHR-10 =====
echo "=== NWPU_VHR-10 ==="

# n-scale
$PY $SCRIPT --name 'yolo11n-C2PSA_AFFNv2-NWPU_VHR-10' --model C2PSA_AFFNv2 --scale n --dataset NWPU_VHR-10 \
  --map50 0.79349 --map50-95 0.48544 --ap-s 0.010 --ap-m 0.374 --ap-l 0.435 \
  --fps 140.1 --params 2.63 --gflops 6.5

# s-scale
$PY $SCRIPT --name 'yolo11s-C2PSA_AFFNv2-NWPU_VHR-10' --model C2PSA_AFFNv2 --scale s --dataset NWPU_VHR-10 \
  --map50 0.85364 --map50-95 0.53785 --ap-s 0.009 --ap-m 0.428 --ap-l 0.481 \
  --fps 138.7 --params 9.57 --gflops 21.7

# m-scale
$PY $SCRIPT --name 'yolo11m-C2PSA_AFFNv2-NWPU_VHR-10' --model C2PSA_AFFNv2 --scale m --dataset NWPU_VHR-10 \
  --map50 0.86033 --map50-95 0.56259 --ap-s 0.046 --ap-m 0.433 --ap-l 0.501 \
  --fps 76.0 --params 20.20 --gflops 68.3

# l-scale
$PY $SCRIPT --name 'yolo11l-C2PSA_AFFNv2-NWPU_VHR-10' --model C2PSA_AFFNv2 --scale l --dataset NWPU_VHR-10 \
  --map50 0.85144 --map50-95 0.56892 --ap-s 0.142 --ap-m 0.435 --ap-l 0.505 \
  --fps 74.2 --params 25.60 --gflops 87.5

# x-scale
$PY $SCRIPT --name 'yolo11x-C2PSA_AFFNv2-NWPU_VHR-10' --model C2PSA_AFFNv2 --scale x --dataset NWPU_VHR-10 \
  --map50 0.85331 --map50-95 0.57327 --ap-s 0.182 --ap-m 0.452 --ap-l 0.505 \
  --fps 61.5 --params 57.51 --gflops 196.0

echo "NWPU_VHR-10 done"
echo "全部写入 Notion 完成！"