# no6-ai 脚本 & 工具索引

> 整理日期: 2026-05-19
> 之前散落在根目录的30+脚本已按功能分类到 `scripts/` 子目录

## 目录结构

```
no6-ai/
├── scripts/
│   ├── train/          ← 训练脚本（按数据集分）
│   ├── eval/           ← 评估/验证/后处理脚本
│   ├── analyze/        ← 分析工具（eval_ap, FPS, Notion）
│   ├── launch/         ← tmux 启动脚本
│   └── archive/        ← 废弃/冗余旧脚本（不删，保留参考）
├── configs/            ← YAML配置、baseline启动脚本
├── runs/               ← 训练输出
├── datasets/           ← 数据集链接
├── ultralytics/        ← no6 ultralytics 代码（含自定义模块）
└── README.md, LICENSE, pyproject.toml  ← 项目文件（不动）
```

## 当前运行中的实验

| GPU | 实验 | 开始时间 | 预计完成 |
|-----|------|---------|---------|
| GPU0 (RTX 3080) | RSOD DEGConv消融 (6配置) | 2026-05-19 ~10:07 | ~12:30 |
| GPU0 (续) | NWPU_VHR-10 DEGConv消融 (6配置) | ~12:30 | ~14:30 |
| GPU1 (RTX 2080 Ti) | VisDrone DEGConv消融 (6配置, 串行) | 2026-05-19 ~10:07 | ~5/20 08:00 |

> 实验脚本位于: `/home/magic524/projects/no5/ultralytics/scripts/run_gpu0.sh` 和 `run_gpu1.sh`
> 实验输出: `/home/magic524/projects/no5/ultralytics/runs/detect/DEGConv_ablation/`

## 快速引用

```bash
# 训练
bash scripts/train/train_visdrone_series.sh

# 验证
bash scripts/eval/batch_eval_rsod.sh

# 后处理 (val + eval_ap + FPS + Notion)
bash scripts/eval/post_process.sh

# 分析
python scripts/analyze/eval_ap.py --model path/to/best.pt --data dataset.yaml
```
