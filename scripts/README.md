# Scripts

## 目录结构

```
scripts/
├── train/        # 训练脚本（命名：YYMMDD_内容.sh）
├── analyze/      # 分析工具
│   ├── eval_ap.py      # COCO eval（AP_s/m/l）
│   ├── notion_write.py # Notion 录入
│   └── bench_fps.py    # FPS 测量
└── trash/        # 旧脚本存档（可恢复）
```

## 规范

- 训练脚本：`YYMMDD_内容.sh`（如 `0521_rsod_baseline.sh`）
- 脚本必须包含：日期、目的、GPU、环境注释
- 使用 `set -euo pipefail` 防静默失败
- YOLO 路径：`/home/magic524/miniconda3/envs/no6-ai/bin/yolo`
- 旧脚本从 trash/ 恢复，不复用则不再出现
