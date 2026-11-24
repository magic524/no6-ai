#!/usr/bin/env python3
"""Validate BAV1 NightDrone all scales via YOLO Python API + eval_ap."""

import io
import re
import subprocess
import sys
from pathlib import Path

GT = "/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json"
BASE = Path("/root/autodl-tmp/no6-ai/runs/detect")
EVAL = "/root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py"
PROJ = BASE / "val" / "BAV1_ND_final"
DATA = "NightDrone.yaml"

import shutil

if PROJ.exists():
    shutil.rmtree(PROJ)
PROJ.mkdir(parents=True)

from ultralytics import YOLO

results = {}

for scale in ["n", "s", "m", "l", "x"]:
    best = BASE / f"BinaryAttentionV1/NightDrone/yolo11{scale}/weights/best.pt"
    print(f"\n{'=' * 60}")
    print(f"yolo11{scale} BAV1 NightDrone")

    # Load model, get params
    model = YOLO(str(best))
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Val via Python API (creates its own dir)
    model.val(
        data=DATA,
        batch=32,
        imgsz=640,
        save_json=True,
        project=str(PROJ),
        name=f"yolo11{scale}",
        device=0,
        exist_ok=True,
        verbose=False,
    )

    # Re-run with verbose to capture mAP
    val_out = model.val(
        data=DATA, batch=32, imgsz=640, project=str(PROJ), name=f"yolo11{scale}", device=0, exist_ok=True, plots=False
    )

    # Get mAP
    rd = val_out.results_dict if hasattr(val_out, "results_dict") else {}
    map50 = rd.get("metrics/mAP50(B)", 0)
    map95 = rd.get("metrics/mAP50-95(B)", 0)

    # Find actual output dir
    out_dir = PROJ / f"yolo11{scale}"
    if not out_dir.exists():
        cands = sorted(PROJ.glob(f"yolo11{scale}*"))
        if cands:
            out_dir = cands[-1]

    # Get GFLOPs
    gflops = 0
    buf = io.StringIO()
    from contextlib import redirect_stdout

    with redirect_stdout(buf):
        model.info()
    for line in buf.getvalue().split("\n"):
        m = re.search(r"([\d.]+)\s*GFLOPs", line)
        if m:
            gflops = float(m.group(1))
            break

    print(f"mAP50={map50:.4f}  mAP50-95={map95:.4f}  Params={params_m:.2f}M  GFLOPs={gflops:.1f}")

    # eval_ap.py
    pred_json = None
    for p in out_dir.rglob("predictions.json"):
        pred_json = p
        break

    ap_s = ap_m = ap_l = 0.0
    if pred_json and pred_json.exists():
        print(f"eval_ap on {pred_json.name}...")
        r = subprocess.run(
            [sys.executable, EVAL, "--gt", GT, "--pred", str(pred_json)], capture_output=True, text=True, timeout=600
        )
        (out_dir / "eval_ap.log").write_text(r.stdout + r.stderr)
        for line in (r.stdout + r.stderr).split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                k = parts[0].rstrip(":")
                try:
                    v = float(parts[1])
                    if k == "AP_s":
                        ap_s = v
                    elif k == "AP_m":
                        ap_m = v
                    elif k == "AP_l":
                        ap_l = v
                except ValueError:
                    pass
    else:
        print("WARNING: predictions.json not found")

    print(f"AP_s={ap_s:.4f}  AP_m={ap_m:.4f}  AP_l={ap_l:.4f}")

    # Save summary
    summary = f"""SCALE={scale}
mAP50={map50:.4f}
mAP50-95={map95:.4f}
AP_s={ap_s:.4f}
AP_m={ap_m:.4f}
AP_l={ap_l:.4f}
Params(M)={params_m:.2f}
GFLOPs={gflops:.1f}
"""
    (out_dir / "summary.txt").write_text(summary)
    print(f"Summary saved to {out_dir}/summary.txt")

    results[scale] = {
        "mAP50": map50,
        "mAP50-95": map95,
        "AP_s": ap_s,
        "AP_m": ap_m,
        "AP_l": ap_l,
        "Params(M)": params_m,
        "GFLOPs": gflops,
    }

print(f"\n{'=' * 60}")
print("ALL SCALES COMPLETE!")
print(f"{'=' * 60}")
header = f"{'Scale':>7} {'mAP50':>7} {'mAP50-95':>9} {'AP_s':>7} {'AP_m':>7} {'AP_l':>7} {'Params':>7} {'GFLOPs':>7}"
print(header)
for s in ["n", "s", "m", "l", "x"]:
    r = results[s]
    print(
        f"{s:>7} {r['mAP50']:.4f} {r['mAP50-95']:.4f} {r['AP_s']:.4f} {r['AP_m']:.4f} {r['AP_l']:.4f} {r['Params(M)']:.2f}M {r['GFLOPs']:.1f}"
    )
