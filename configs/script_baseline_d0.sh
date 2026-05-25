#!/bin/bash
#######      chmod +x script_baseline_d0.sh
#######      ./script_baseline_d0.sh

##### YOLOv26 D0 Baseline — no6-ai #####
# GPU: RTX 3080 (20GB)
# Batch tuned to ~half VRAM
# YOLO26n summary: 260 layers, 2,507,700 parameters, 5.8 GFLOPs

# VisDrone200
yolo detect train data=VisDrone.yaml model=yolo26n.yaml epochs=200 batch=16 imgsz=640 device=0 name='Baseline_Model_Experiment/VisDrone/yolo26n-VD200-d0'

# TinyPerson200
yolo detect train data=TinyPerson.yaml model=yolo26n.yaml epochs=200 batch=16 imgsz=640 device=0 name='Baseline_Model_Experiment/TinyPerson/yolo26n-TP200-d0'

# RSOD200
yolo detect train data=RSOD.yaml model=yolo26n.yaml epochs=200 batch=16 imgsz=640 device=0 name='Baseline_Model_Experiment/RSOD/yolo26n-RSOD200-d0'

# NWPU_VHR-10 200 (含无目标图片)
yolo detect train data=NWPU_VHR-10.yaml model=yolo26n.yaml epochs=200 batch=16 imgsz=640 device=0 name='Baseline_Model_Experiment/NWPU_VHR-10/yolo26n-NV200-d0'

# AI-TOD200
yolo detect train data=AI-TOD.yaml model=yolo26n.yaml epochs=200 batch=16 imgsz=640 device=0 name='Baseline_Model_Experiment/AI-TOD/yolo26n-AT200-d0'
