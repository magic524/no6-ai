#!/bin/bash
# C2PSA_AFFN — RSOD + NWPU_VHR-10 (Cloud1, RTX 3090)
# Serial: 5 scales × 2 datasets = 10 runs

set -e
YOLO=/root/miniconda3/envs/no6-ai/bin/yolo
DEVICE=0

echo "========================================"
echo "C2PSA_AFFN — RSOD (Cloud1) — $(date)"
echo "========================================"

echo "[1/10] yolo11n — RSOD — batch 16"
$YOLO detect train data=RSOD.yaml model=yolo11n-C2PSA_AFFN.yaml batch=16 device=$DEVICE
echo "yolo11n RSOD DONE at $(date)"

echo "[2/10] yolo11s — RSOD — batch 16"
$YOLO detect train data=RSOD.yaml model=yolo11s-C2PSA_AFFN.yaml batch=16 device=$DEVICE
echo "yolo11s RSOD DONE at $(date)"

echo "[3/10] yolo11m — RSOD — batch 8"
$YOLO detect train data=RSOD.yaml model=yolo11m-C2PSA_AFFN.yaml batch=8 device=$DEVICE
echo "yolo11m RSOD DONE at $(date)"

echo "[4/10] yolo11l — RSOD — batch 8"
$YOLO detect train data=RSOD.yaml model=yolo11l-C2PSA_AFFN.yaml batch=8 device=$DEVICE
echo "yolo11l RSOD DONE at $(date)"

echo "[5/10] yolo11x — RSOD — batch 4"
$YOLO detect train data=RSOD.yaml model=yolo11x-C2PSA_AFFN.yaml batch=4 device=$DEVICE
echo "yolo11x RSOD DONE at $(date)"

echo ""
echo "========================================"
echo "C2PSA_AFFN — NWPU_VHR-10 (Cloud1) — $(date)"
echo "========================================"

echo "[6/10] yolo11n — NWPU_VHR-10 — batch 16"
$YOLO detect train data=NWPU_VHR-10.yaml model=yolo11n-C2PSA_AFFN.yaml batch=16 device=$DEVICE
echo "yolo11n NWPU DONE at $(date)"

echo "[7/10] yolo11s — NWPU_VHR-10 — batch 16"
$YOLO detect train data=NWPU_VHR-10.yaml model=yolo11s-C2PSA_AFFN.yaml batch=16 device=$DEVICE
echo "yolo11s NWPU DONE at $(date)"

echo "[8/10] yolo11m — NWPU_VHR-10 — batch 8"
$YOLO detect train data=NWPU_VHR-10.yaml model=yolo11m-C2PSA_AFFN.yaml batch=8 device=$DEVICE
echo "yolo11m NWPU DONE at $(date)"

echo "[9/10] yolo11l — NWPU_VHR-10 — batch 8"
$YOLO detect train data=NWPU_VHR-10.yaml model=yolo11l-C2PSA_AFFN.yaml batch=8 device=$DEVICE
echo "yolo11l NWPU DONE at $(date)"

echo "[10/10] yolo11x — NWPU_VHR-10 — batch 4"
$YOLO detect train data=NWPU_VHR-10.yaml model=yolo11x-C2PSA_AFFN.yaml batch=4 device=$DEVICE
echo "yolo11x NWPU DONE at $(date)"

echo "全部完成！$(date)"
