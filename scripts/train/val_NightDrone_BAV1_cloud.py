#!/usr/bin/env python3
"""Validate BinaryAttentionV1 on NightDrone via YOLO Python API."""
import sys, os, re, json
from pathlib import Path

# ── config ────────────────────────────────────────────
GT = "/root/autodl-tmp/datasets/NightDrone/annotations/instances_val.json"
ROOT = Path("/root/autodl-tmp/no6-ai/runs/detect")
EVAL_AP = "/root/autodl-tmp/no6-ai/scripts/analyze/eval_ap.py"
DATA_YAML = "NightDrone.yaml"
scales = ["n", "s", "m", "l", "x"]
# ──────────────────────────────────────────────────────

from ultralytics import YOLO

for scale in scales:
    best = str(ROOT / f"BinaryAttentionV1/NightDrone/yolo11{scale}/weights/best.pt")
    val_dir = ROOT / "val" / f"val_bav1_nd/yolo11{scale}"
    val_dir.mkdir(parents=True, exist_ok=True)
    # Clear old files
    for f in val_dir.iterdir():
        if f.is_file():
            f.unlink()

    print(f"\n{'='*60}")
    print(f"Validating: yolo11{scale}-BinaryAttentionV1 NightDrone")
    print(f"{'='*60}")

    # Step 1 & 3: yolo val (produces predictions.json) + get Params/GFLOPs
    print("[Step 1+3/4] yolo val + Params/GFLOPs ...")
    model = YOLO(best)

    # Get Params
    total_params = sum(p.numel() for p in model.model.parameters())
    params_m = total_params / 1e6
    print(f"Params(M): {params_m:.2f}")

    # Get GFLOPs from model info
    info = model.info()
    # info returns: [num_layers, params, gflops] in some versions
    # or just print and parse
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.info()
    info_str = buf.getvalue()
    gflops = None
    for line in info_str.split("\n"):
        m = re.search(r'([\d.]+)\s*GFLOPs', line)
        if m:
            gflops = float(m.group(1))
            break
    if gflops is None:
        # Fallback: try info() return value
        if isinstance(info, (list, tuple)) and len(info) >= 3:
            gflops = info[2]
        else:
            gflops = 0
    print(f"GFLOPs: {gflops:.1f}")

    # Run val
    val_results = model.val(
        data=DATA_YAML,
        batch=32,
        imgsz=640,
        save_json=True,
        project=str(ROOT / "val"),
        name=f"val_bav1_nd/yolo11{scale}",
        device=0,
    )

    # Extract mAP from val_results
    mAP50 = getattr(val_results, "box", {}).get("map50", 0) if hasattr(val_results, "box") else 0
    mAP50_95 = getattr(val_results, "box", {}).get("map", 0) if hasattr(val_results, "box") else 0

    # Fallback: read results.csv if val_results doesn't have the attributes
    results_csv = val_dir / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                mAP50 = float(last.get("metrics/mAP50(B)", mAP50))
                mAP50_95 = float(last.get("metrics/mAP50-95(B)", mAP50_95))

    print(f"mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")

    # Save val stdout
    with open(val_dir / "val_stdout.log", "w") as f:
        f.write(f"mAP50={mAP50}\nmAP50-95={mAP50_95}\nParams(M)={params_m:.2f}\nGFLOPs={gflops:.1f}\n")

    # Step 2: eval_ap.py via subprocess
    print("[Step 2/4] eval_ap.py ...")
    pred_json = None
    for p in val_dir.rglob("predictions.json"):
        # prefer _fixed.json if exists (from a prior run)
        pred_json = p
    if pred_json is None:
        # Also check the main detect dir
        for p in Path(f"{ROOT}/val/val_bav1_nd/yolo11{scale}").rglob("predictions.json"):
            pred_json = p

    AP_S = AP_M = AP_L = 0.0
    if pred_json and pred_json.exists():
        print(f"Running eval_ap on {pred_json}")
        import subprocess
        result = subprocess.run(
            [sys.executable, EVAL_AP, "--gt", GT, "--pred", str(pred_json)],
            capture_output=True, text=True, timeout=600
        )
        out = result.stdout + result.stderr
        (val_dir / "eval_ap_result.log").write_text(out)
        print(out)

        # Parse AP_s/m/l
        for line in out.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                if key == "AP_s":
                    try: AP_S = float(parts[1])
                    except: pass
                elif key == "AP_m":
                    try: AP_M = float(parts[1])
                    except: pass
                elif key == "AP_l":
                    try: AP_L = float(parts[1])
                    except: pass
    else:
        print("WARNING: predictions.json not found, skipping eval_ap")

    # Save summary
    summary = f"""SCALE={scale}
mAP50={mAP50}
mAP50-95={mAP50_95}
AP_s={AP_S}
AP_m={AP_M}
AP_l={AP_L}
Params(M)={params_m:.2f}
GFLOPs={gflops:.1f}
"""
    (val_dir / "summary.txt").write_text(summary)
    print(f"\nSummary:\n{summary}")

print(f"\n{'='*60}")
print("ALL VALIDATION COMPLETE!")
print(f"{'='*60}")
