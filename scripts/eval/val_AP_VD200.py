"""从Notion页面保存的val_AP脚本 - 用pycocotools计算AP_s/AP_m/AP_l."""

import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def fix_json(input_file, output_file, annotations_file):
    """修正 category_id 和 image_id 并保存新的 JSON 文件."""
    # 读取预测 JSON 文件
    with open(input_file) as f:
        predictions = json.load(f)

    # 读取 COCO 标注文件
    with open(annotations_file) as f:
        coco_gt = json.load(f)

    # 创建文件名到 image_id 的映射
    filename_to_id = {img["file_name"].split("/")[-1].replace(".jpg", ""): img["id"] for img in coco_gt["images"]}

    # 修正 category_id 和 image_id
    for item in predictions:
        # VisDrone需要修正category_id (+1)
        if "category_id" in item:
            item["category_id"] += 1

        # 修正 image_id
        filename = item["image_id"]  # 可能是 "0000364_00589_d_0000798"
        if filename in filename_to_id:
            item["image_id"] = filename_to_id[filename]
        else:
            print(f"警告: {filename} 在 instances_val.json 中找不到匹配项")

    # 保存修正后的 JSON 文件
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"修正完成，结果保存在: {output_file}")


def evaluate_coco(ann_file, pred_file, output_dir=None):
    """使用 COCO 评测框架计算 AP_s, AP_m, AP_l."""
    # 加载 COCO 标注和预测
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(pred_file)

    # 初始化评估器
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # 配置评估参数（分小目标、中目标、大目标）
    coco_eval.params.areaRng = [
        [0, 1e5],  # all (仅占位，实际不计算)
        [0, 32**2],  # small (0-1024像素)
        [32**2, 96**2],  # medium (1024-9216像素)
        [96**2, 1e5**2],  # large (9216+像素)
    ]
    coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]

    # 运行 COCO 评测
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取 AP 结果
    stats = coco_eval.stats
    ap_small = stats[1]  # AP_s
    ap_medium = stats[2]  # AP_m
    ap_large = stats[3]  # AP_l

    # 输出到终端
    print(f"\nAP_s (Small):  {ap_small:.3f}")
    print(f"AP_m (Medium): {ap_medium:.3f}")
    print(f"AP_l (Large):  {ap_large:.3f}")

    # 保存到文件
    if output_dir:
        import os
        from datetime import datetime

        results_file = os.path.join(output_dir, "ap_results.txt")
        with open(results_file, "w") as f:
            f.write("COCO AP Evaluation Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Annotations file: {ann_file}\n")
            f.write(f"Predictions file: {pred_file}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"AP_s (Small):  {ap_small:.6f}\n")
            f.write(f"AP_m (Medium): {ap_medium:.6f}\n")
            f.write(f"AP_l (Large):  {ap_large:.6f}\n\n")
            f.write("Detailed COCO stats:\n")
            f.writelines(f"stats[{i}]: {stat:.6f}\n" for i, stat in enumerate(stats))

        print(f"\nResults saved to: {results_file}")

    return ap_small, ap_medium, ap_large


def main():
    parser = argparse.ArgumentParser(description="修正 JSON 并计算 COCO AP 结果")
    parser.add_argument("input_file", help="输入原始预测 JSON 文件路径")
    parser.add_argument("annotations_file", help="COCO 格式的标注 JSON 文件路径")
    parser.add_argument("--output-dir", help="结果保存目录（可选）")
    args = parser.parse_args()

    # 生成修正后的 JSON 文件路径
    output_file = args.input_file.replace(".json", "_fixed.json")

    # 1. 修正 JSON 文件
    fix_json(args.input_file, output_file, args.annotations_file)

    # 2. 计算 COCO AP
    evaluate_coco(args.annotations_file, output_file, args.output_dir)


if __name__ == "__main__":
    main()
