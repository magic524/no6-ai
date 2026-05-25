"""COCO evaluation script — extract AP_S / AP_M / AP_L and per-size AP50 from predictions.json.

stats[] array layout (fixed by pycocotools):
  stats[0] = AP @ IoU=0.50:0.95, area=all
  stats[1] = AP @ IoU=0.50,      area=all (AP50)
  stats[2] = AP @ IoU=0.75,      area=all (AP75)
  stats[3] = AP @ IoU=0.50:0.95, area=small  (AP_s)   ← correct small-object AP
  stats[4] = AP @ IoU=0.50:0.95, area=medium (AP_m)
  stats[5] = AP @ IoU=0.50:0.95, area=large  (AP_l)

NOTE: stats[1] is AP50 for ALL sizes, NOT AP_s!
Previous code incorrectly used stats[1] as AP_s (stats[3] is correct).
"""
import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def fix_prediction_ids(ann_file: str, pred_file: str, output: str):
    """Fix image_id and category_id in predictions.json to match GT."""
    with open(pred_file) as f:
        preds = json.load(f)
    coco_gt = COCO(ann_file)

    # Build basename → image_id mapping from GT
    # GT file_name: "val/0000001_02999_d_0000005.jpg" → strip prefix
    gt_img_ids = {}
    for img in coco_gt.dataset["images"]:
        basename = img["file_name"].rsplit("/", 1)[-1]  # strip 'val/' prefix
        gt_img_ids[basename] = img["id"]

    for p in preds:
        fname = p.get("file_name", "")
        if fname in gt_img_ids:
            p["image_id"] = gt_img_ids[fname]
        # category_id: ultralytics save_json already produces 1-indexed,
        # matching GT categories (1-10 for VisDrone). Leave as-is.

    with open(output, "w") as f:
        json.dump(preds, f)
    print(f"Fixed predictions saved to {output}")
    return output


def run_coco_eval(ann_file: str, pred_file: str):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(pred_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    # Per-size AP50 (for reference — previous code incorrectly used this as AP_s)
    precision = coco_eval.eval["precision"]
    # precision[T=10, R=101, K=10, A=4, M=3] where A=[all,small,medium,large], M=[1,10,100]
    ap50_all   = float(np.mean(precision[0, :, :, 0, 2]))  # IoU=0.50, area=all
    ap50_small = float(np.mean(precision[0, :, :, 1, 2]))  # IoU=0.50, area=small
    ap50_med   = float(np.mean(precision[0, :, :, 2, 2]))  # IoU=0.50, area=medium
    ap50_large = float(np.mean(precision[0, :, :, 3, 2]))  # IoU=0.50, area=large

    return {
        # Standard COCO AP (IoU=0.50:0.95)
        "AP_all": stats[0],
        "AP50":   stats[1],
        "AP75":   stats[2],
        "AP_s":   stats[3],  # AP @ IoU=0.50:0.95, small objects  — CORRECT
        "AP_m":   stats[4],  # AP @ IoU=0.50:0.95, medium objects
        "AP_l":   stats[5],  # AP @ IoU=0.50:0.95, large objects
        # Per-size AP50 (supplementary)
        "AP50_all":   ap50_all,
        "AP50_small": ap50_small,
        "AP50_med":   ap50_med,
        "AP50_large": ap50_large,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True,
                        help="Ground truth COCO JSON (instances_val.json)")
    parser.add_argument("--pred", required=True,
                        help="Predictions JSON from ultralytics save_json=True")
    parser.add_argument("--output", default=None,
                        help="Fixed predictions output path")
    args = parser.parse_args()

    fixed_pred = fix_prediction_ids(
        args.gt, args.pred,
        args.output or args.pred.replace(".json", "_fixed.json")
    )
    results = run_coco_eval(args.gt, fixed_pred)

    print("\n=== COCO Evaluation Results ===")
    print(f"{'Metric':<20} {'Value':<10} {'Note'}")
    print("-" * 55)
    print(f"{'AP_all':<20} {results['AP_all']:.4f}    AP @ IoU=0.50:0.95, all areas")
    print(f"{'AP50':<20} {results['AP50']:.4f}    AP @ IoU=0.50, all areas")
    print(f"{'AP75':<20} {results['AP75']:.4f}    AP @ IoU=0.75, all areas")
    print(f"{'AP_s':<20} {results['AP_s']:.4f}    AP @ IoU=0.50:0.95, small (area<32²)")
    print(f"{'AP_m':<20} {results['AP_m']:.4f}    AP @ IoU=0.50:0.95, medium")
    print(f"{'AP_l':<20} {results['AP_l']:.4f}    AP @ IoU=0.50:0.95, large")
    print()
    print(f"{'AP50_small':<20} {results['AP50_small']:.4f}  AP @ IoU=0.50, small (for reference)")
