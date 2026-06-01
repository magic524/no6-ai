#!/usr/bin/env python3
"""Batch post-process: eval_ap + FPS + Notion write for DEGConv ablation. (FIXED v2)."""

import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notion_write import write_experiment

RUNS = "runs/detect/DEGConv_ablation"
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")


def measure_fps(weights_path):
    model = YOLO(weights_path)
    dummy = torch.randn(1, 3, 640, 640).cuda()
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for _ in range(50):
            model.predict(dummy, verbose=False)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(500):
            model.predict(dummy, verbose=False)
        torch.cuda.synchronize()
    return round(500 / (time.time() - start), 1)


def run_eval_ap(gt_json, pred_json):
    """Run COCO eval. Returns (AP_s, AP_m, AP_l) or (0,0,0) on failure."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json)

    with open(pred_json) as f:
        preds = json.load(f)

    # Fix image_ids: match predictions to GT by basename
    gt_img_ids = {}
    for img in coco_gt.dataset["images"]:
        bn = os.path.splitext(os.path.basename(img["file_name"]))[0]
        gt_img_ids[bn] = img["id"]

    matched = 0
    for p in preds:
        fname = os.path.splitext(os.path.basename(p.get("file_name", "")))[0]
        if fname in gt_img_ids:
            p["image_id"] = gt_img_ids[fname]
            matched += 1

    if matched == 0:
        print(f"  ⚠ eval_ap: 0/{len(preds)} predictions matched GT!")
        return 0.0, 0.0, 0.0

    # Save fixed predictions
    fixed_pred = pred_json.replace(".json", "_fixed.json")
    with open(fixed_pred, "w") as f:
        json.dump(preds, f)

    coco_dt = coco_gt.loadRes(fixed_pred)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return stats[3], stats[4], stats[5]


# ── Configuration ──
GT = {
    "RSOD": "/mnt/e/Datasets/Small_Objects_Dataset/RSOD/annotations/instances_val.json",
    "NWPU_VHR-10": "/mnt/e/Datasets/Small_Objects_Dataset/NWPU_VHR-10/annotations/instances_val.json",
}

EXPS = [
    # (dataset, exp_dir_name, variant_display, yaml_name_for_model_info)
    ("RSOD", "yolo11n-RSOD-baseline", "baseline", "yolo11n", "yolo11"),
    ("RSOD", "yolo11n-RSOD-V1_backbone", "V1_backbone", "yolo11-DEGConv", "yolo11+DEGConv"),
    ("RSOD", "yolo11n-RSOD-V2_neck", "V2_neck", "yolo11-DEGConv-neck", "yolo11+DEGConv"),
    ("RSOD", "yolo11n-RSOD-V3_full", "V3_full", "yolo11-DEGConv-full", "yolo11+DEGConv"),
    ("RSOD", "yolo11n-RSOD-V4_shallow", "V4_shallow", "yolo11-DEGConv-shallow", "yolo11+DEGConv"),
    ("RSOD", "yolo11n-RSOD-V5_deep", "V5_deep", "yolo11-DEGConv-deep", "yolo11+DEGConv"),
    ("NWPU_VHR-10", "yolo11n-NV-baseline", "baseline", "yolo11n", "yolo11"),
    ("NWPU_VHR-10", "yolo11n-NV-V1_backbone", "V1_backbone", "yolo11-DEGConv", "yolo11+DEGConv"),
    ("NWPU_VHR-10", "yolo11n-NV-V2_neck", "V2_neck", "yolo11-DEGConv-neck", "yolo11+DEGConv"),
    ("NWPU_VHR-10", "yolo11n-NV-V3_full", "V3_full", "yolo11-DEGConv-full", "yolo11+DEGConv"),
    ("NWPU_VHR-10", "yolo11n-NV-V4_shallow", "V4_shallow", "yolo11-DEGConv-shallow", "yolo11+DEGConv"),
    ("NWPU_VHR-10", "yolo11n-NV-V5_deep", "V5_deep", "yolo11-DEGConv-deep", "yolo11+DEGConv"),
]


# Resolve yaml path
def resolve_yaml(name):
    if name == "yolo11n":
        return "ultralytics/cfg/models/11/yolo11.yaml"
    return f"ultralytics/cfg/models/11/DEGConv/{name}.yaml"


print("=" * 60)
print("DEGConv Ablation - Batch Post-Processing + Notion Write (v2)")
print("=" * 60)

# Warm up
print("Warming up...")
YOLO(resolve_yaml("yolo11n"))
print("Ready.\n")

# Cache model info (params + GFLOPs) using proper API
cache = {}
for _, _, _, yn, _ in EXPS:
    if yn not in cache:
        yp = resolve_yaml(yn)
        m = YOLO(yp)
        p = round(get_num_params(m.model) / 1e6, 2)
        g = round(get_flops(m.model, 640), 2)
        cache[yn] = (p, g)
        print(f"  {yn}: {p}M params, {g} GFLOPs")

print()

for ds, dname, variant, yn, model_name in EXPS:
    exp_dir = f"{RUNS}/{ds}/{dname}"
    csv_path = f"{exp_dir}/results.csv"
    weights = f"{exp_dir}/weights/best.pt"
    pred_json = f"{exp_dir}/predictions.json"

    if not os.path.exists(weights):
        print(f"SKIP {ds}/{variant}: no weights at {weights}")
        continue

    if not os.path.exists(csv_path):
        print(f"SKIP {ds}/{variant}: no results.csv")
        continue

    name = f"yolo11n-DEGConv-{variant}-{ds}"
    print(f"\n--- {ds} / {variant} ---")

    # mAP from results.csv
    with open(csv_path) as f:
        rows = [r for r in csv.reader(f) if len(r) >= 9]
    last = rows[-1]
    map50 = round(float(last[7]), 4)
    map50_95 = round(float(last[8]), 4)
    print(f"  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")

    # Params & GFLOPs
    params, gflops = cache[yn]
    print(f"  Params={params}M  GFLOPs={gflops}")

    # AP_s/m/l via eval_ap
    ap_s = ap_m = ap_l = 0.0
    if os.path.exists(pred_json):
        try:
            ap_s, ap_m, ap_l = run_eval_ap(GT[ds], pred_json)
            print(f"  AP_s={ap_s:.4f}  AP_m={ap_m:.4f}  AP_l={ap_l:.4f}")
        except Exception as e:
            print(f"  eval_ap error: {e}")
    else:
        print(f"  predictions.json not found at {pred_json}")

    # FPS
    fps = measure_fps(weights)
    print(f"  FPS={fps:.1f}")

    # Write to Notion
    write_experiment(
        name=name,
        model=model_name,
        scale="n",
        dataset=ds,
        map50=map50,
        map50_95=map50_95,
        ap_s=round(ap_s, 4),
        ap_m=round(ap_m, 4),
        ap_l=round(ap_l, 4),
        fps=fps,
        params=params,
        gflops=gflops,
    )
    print(f"  => Notion written (model={model_name})")

print(f"\n{'=' * 60}")
print("ALL DONE!")
