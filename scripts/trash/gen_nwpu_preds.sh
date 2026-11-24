#!/usr/bin/env bash
# Generate missing predictions for all 6 NWPU_VHR-10 experiments
cd /home/magic524/projects/no6-ai
DATA=/mnt/e/Datasets/Small_Objects_Dataset/NWPU_VHR-10/NWPU_VHR-10.yaml
EXPS=(baseline V1_backbone V2_neck V3_full V4_shallow V5_deep)
EXPS_DIR=(yolo11n-NV-baseline yolo11n-NV-V1_backbone yolo11n-NV-V2_neck yolo11n-NV-V3_full yolo11n-NV-V4_shallow yolo11n-NV-V5_deep)

for i in "${!EXPS[@]}"; do
  exp=${EXPS[$i]}
  dir=${EXPS_DIR[$i]}
  echo "=== Val: $exp ==="

  CUDA_VISIBLE_DEVICES=0 /home/magic524/miniconda3/bin/conda run -n ultralytics-no5 \
    yolo detect val \
    data=$DATA \
    model=runs/detect/DEGConv_ablation/NWPU_VHR-10/$dir/weights/best.pt \
    batch=16 imgsz=640 save_json \
    name="val_NW_${exp}" project="runs/tmp_val_nw" 2>&1 | tail -3

  # Copy predictions to experiment dir
  SRC=runs/tmp_val_nw/val_NW_${exp}/predictions.json
  DST=runs/detect/DEGConv_ablation/NWPU_VHR-10/$dir/predictions.json
  if [ -f "$SRC" ]; then
    cp "$SRC" "$DST"
    echo "Copied to $DST ($(wc -c < $DST) bytes)"
  else
    echo "ERROR: $SRC not found"
  fi
  echo ""
done

# Cleanup
rm -rf runs/tmp_val_nw
echo "All done!"
