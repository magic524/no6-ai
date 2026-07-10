#!/bin/bash
# C2PSA_AFFNv2 — RSOD + NWPU_VHR-10 COCO Eval + FPS (Cloud2)
# Uses training-saved predictions.json (save_json=True)
# Skips category_id +1 bug, only fixes image_id mapping

set -o pipefail
PY=/root/miniconda3/envs/no6-ai/bin/python
MODULE=C2PSA_AFFNv2
BASE=/root/autodl-tmp/no6-ai/runs/detect/${MODULE}
LOG=/root/autodl-tmp/no6-ai/scripts/analyze/C2PSA_AFFNv2_RSOD_NWPU_eval_fps.log

echo "=========================================" | tee $LOG
echo "C2PSA_AFFNv2 — RSOD + NWPU_VHR-10 Eval+FPS" | tee -a $LOG
echo "Date: $(date)" | tee -a $LOG
echo "=========================================" | tee -a $LOG

run_eval_fps() {
    local dataset=$1
    local scale=$2
    local pred_dir="$BASE/${dataset}/yolo11${scale}-${MODULE}"
    local best="$pred_dir/weights/best.pt"
    local pred_json="$pred_dir/predictions.json"
    local ann_file="/root/autodl-tmp/datasets/${dataset}/annotations/instances_val.json"

    echo "" | tee -a $LOG
    echo "--- yolo11${scale}-${MODULE} — ${dataset} ---" | tee -a $LOG

    # Verify files exist
    [ -f "$best" ] || { echo "SKIP: $best not found" | tee -a $LOG; return; }
    [ -f "$pred_json" ] || { echo "SKIP: $pred_json not found" | tee -a $LOG; return; }
    [ -f "$ann_file" ] || { echo "SKIP: $ann_file not found" | tee -a $LOG; return; }

    # === COCO Eval (direct pycocotools, skip category_id +1 bug) ===
    echo "--- COCO Eval ---" | tee -a $LOG
    $PY -c "
import json, sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ann_file = '$ann_file'
pred_file = '$pred_json'

ann = json.load(open(ann_file))
pred = json.load(open(pred_file))

# Build filename → image_id mapping
fn2id = {img['file_name'].split('/')[-1].replace('.jpg',''): img['id']
         for img in ann['images']}

# Only fix image_id, NOT category_id
for p in pred:
    if str(p['image_id']) in fn2id:
        p['image_id'] = fn2id[str(p['image_id'])]
    elif p['image_id'] in fn2id:
        pass  # already correct
    else:
        # Try filename-based matching
        pass

# Save fixed version
fixed_json = '$pred_dir/predictions_cocoeval.json'
json.dump(pred, open(fixed_json, 'w'))

coco_gt = COCO(ann_file)
coco_dt = coco_gt.loadRes(fixed_json)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

stats = coco_eval.stats
print(f'AP: {stats[0]:.6f}')
print(f'AP50: {stats[1]:.6f}')
print(f'AP75: {stats[2]:.6f}')
print(f'AP_s: {stats[3]:.6f}')
print(f'AP_m: {stats[4]:.6f}')
print(f'AP_l: {stats[5]:.6f}')

# Save to val log
with open('$pred_dir/val_cocoeval.log', 'w') as f:
    f.write(f'AP: {stats[0]:.6f}\nAP50: {stats[1]:.6f}\nAP75: {stats[2]:.6f}\n')
    f.write(f'AP_s: {stats[3]:.6f}\nAP_m: {stats[4]:.6f}\nAP_l: {stats[5]:.6f}\n')
" 2>&1 | tee -a $LOG

    # === FPS ===
    echo "--- FPS ---" | tee -a $LOG
    $PY -c "
import torch, time, sys
from ultralytics import YOLO

try:
    m = YOLO('$best')
    d = torch.rand(1, 3, 640, 640).cuda()
    for _ in range(30):
        m.predict(d, verbose=False)
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(200):
        m.predict(d, verbose=False)
    torch.cuda.synchronize()
    fps = 200.0 / (time.time() - s)
    print(f'FPS: {fps:.1f}')
except Exception as e:
    print(f'FPS ERROR: {e}')
" 2>&1 | tee -a $LOG
}

# ===== RSOD =====
echo "" | tee -a $LOG
echo "========== RSOD ==========" | tee -a $LOG
for s in n s m l x; do
    run_eval_fps "RSOD" "$s"
done

# ===== NWPU_VHR-10 =====
echo "" | tee -a $LOG
echo "========== NWPU_VHR-10 ==========" | tee -a $LOG
for s in n s m l x; do
    run_eval_fps "NWPU_VHR-10" "$s"
done

echo "" | tee -a $LOG
echo "=========================================" | tee -a $LOG
echo "全部完成！$(date)" | tee -a $LOG
echo "=========================================" | tee -a $LOG