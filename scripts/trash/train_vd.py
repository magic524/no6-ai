#!/usr/bin/env python3
"""VisDrone YOLO26n Baseline Training."""

import sys
import time

log_path = "/home/magic524/projects/no6-ai/train_visdrone.log"
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

print(f"=== Training started at {time.ctime()} ===")
print("Importing ultralytics...")
from ultralytics import YOLO

print("Ultralytics imported OK")

print("Building model yolo26n.yaml...")
model = YOLO("yolo26n.yaml")
print(f"Model summary: {model.info(verbose=False)}")

print("Starting training...")
results = model.train(
    data="VisDrone.yaml",
    epochs=200,
    batch=16,
    imgsz=640,
    device=0,
    workers=0,
    cache="ram",
    name="no6-baseline/VisDrone/yolo26n-VD200-d0",
    exist_ok=True,
)

print(f"=== Training completed at {time.ctime()} ===")
print(f"Results: {results}")
