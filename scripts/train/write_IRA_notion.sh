#!/bin/bash
cd /home/magic524/projects/no6-ai
PY=/home/magic524/miniconda3/envs/no6-ai/bin/python

$PY scripts/analyze/notion_write.py --name 'yolo11s-IRA-RSOD' --model YOLO11 --scale s --dataset RSOD --map50 0.870 --map50-95 0.579 --ap-s 0.4176 --ap-m 0.6507 --ap-l 0.6292 --fps 110.8 --params 10.64 --gflops 31.6
$PY scripts/analyze/notion_write.py --name 'yolo11m-IRA-RSOD' --model YOLO11 --scale m --dataset RSOD --map50 0.895 --map50-95 0.609 --ap-s 0.4339 --ap-m 0.6654 --ap-l 0.6546 --fps 91.9 --params 23.01 --gflops 106.1
$PY scripts/analyze/notion_write.py --name 'yolo11l-IRA-RSOD' --model YOLO11 --scale l --dataset RSOD --map50 0.892 --map50-95 0.600 --ap-s 0.4481 --ap-m 0.6478 --ap-l 0.6512 --fps 60.5 --params 30.99 --gflops 158.4
$PY scripts/analyze/notion_write.py --name 'yolo11x-IRA-RSOD' --model YOLO11 --scale x --dataset RSOD --map50 0.850 --map50-95 0.587 --ap-s 0.4399 --ap-m 0.6606 --ap-l 0.6313 --fps 36.3 --params 69.59 --gflops 355.4

$PY scripts/analyze/notion_write.py --name 'yolo11n-IRA-NWPU_VHR-10' --model YOLO11 --scale n --dataset 'NWPU_VHR-10' --map50 0.804 --map50-95 0.483 --ap-s 0.0434 --ap-m 0.4173 --ap-l 0.5441 --fps 119.7 --params 2.90 --gflops 8.9
$PY scripts/analyze/notion_write.py --name 'yolo11s-IRA-NWPU_VHR-10' --model YOLO11 --scale s --dataset 'NWPU_VHR-10' --map50 0.852 --map50-95 0.531 --ap-s 0.0358 --ap-m 0.4735 --ap-l 0.5623 --fps 118.8 --params 10.64 --gflops 31.6
$PY scripts/analyze/notion_write.py --name 'yolo11m-IRA-NWPU_VHR-10' --model YOLO11 --scale m --dataset 'NWPU_VHR-10' --map50 0.861 --map50-95 0.556 --ap-s 0.1666 --ap-m 0.4943 --ap-l 0.5878 --fps 94.5 --params 23.02 --gflops 106.1
$PY scripts/analyze/notion_write.py --name 'yolo11l-IRA-NWPU_VHR-10' --model YOLO11 --scale l --dataset 'NWPU_VHR-10' --map50 0.851 --map50-95 0.559 --ap-s 0.1914 --ap-m 0.4961 --ap-l 0.5881 --fps 61.0 --params 30.99 --gflops 158.4
$PY scripts/analyze/notion_write.py --name 'yolo11x-IRA-NWPU_VHR-10' --model YOLO11 --scale x --dataset 'NWPU_VHR-10' --map50 0.883 --map50-95 0.578 --ap-s 0.1356 --ap-m 0.5136 --ap-l 0.6039 --fps 36.4 --params 69.59 --gflops 355.4

echo 'All done!'