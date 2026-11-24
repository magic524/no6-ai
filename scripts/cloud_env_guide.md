# Cloud 环境快速参考

> 记录云机路径、训练配置、Git分支等关键信息，避免新会话重复排查。

## Git 信息

- **GitHub 仓库**: `magic524/no6-ai` (分支: `agent`)
- **本地路径**: `/home/magic524/projects/no6-ai/ultralytics` (扁平布局，ultralytics 包在根目录)
- **Cloud 路径**: `/root/autodl-tmp/no6-ai/ultralytics` (结构和本地一致)
- **三方对齐**: `git pull origin agent` 即可同步

## 目录结构（扁平布局）

```
<repo_root>/               ← 也就是 ultralytics 包
├── nn/block/              ← 模块实现 (WDAM.py, DEGConv.py 等)
├── cfg/datasets/          ← 数据集 YAML (统一用相对路径 ../../../datasets/<Name>)
├── cfg/models/11/         ← yolo11 模型配置
├── cfg/models/26/         ← yolo26 模型配置
├── nn/tasks.py            ← 模块注册表
├── .git/
└── runs/detect/           ← 训练输出
```

## 数据集路径体系

- `cfg/datasets/<Name>.yaml` 中 `path: ../../../datasets/<Name>` **(相对路径)**
- `datasets/<Name>` 是 symlink → `/root/autodl-tmp/datasets/<Name>` (真实数据)
- 绝对路径旧版已被注释，仅保留为历史参考

## YOLO 训练惯例

| 尺度 | batch | patience | epochs | imgsz |
| ---- | ----- | -------- | ------ | ----- |
| n/s  | 16    | 无       | 200    | 640   |
| m/l  | 8     | 无       | 200    | 640   |
| x    | 4     | 无       | 200    | 640   |

- 训练命令格式: `$YOLO train ...` (非 `yolo detect train`)
- 实验名格式: `{Model}/{Dataset}/yolo11{n}` (三级路径)
- 数据写入 Notion 前必须查库、验证格式

## Shell 脚本规范

- `set -euo pipefail`
- `DATASET` 变量在脚本开头定义
- 多行 `\` 续行缩进
- 每尺度独立注释块 + `echo "--- [$(date)]..."` 时间戳
- 串行训练，不支持背景并行

## WDAM 模块

- **文件**: `ultralytics/nn/block/WDAM.py` (DWT→低频窗口注意力+高频融合→IDWT)
- **Wrapper**: `C2PSA_WDAM` (替换 C2PSA)
- **YOLO11 YAML**: `cfg/models/11/WDAM/yolo11-C2PSA_WDAM.yaml`
- **YOLO26 YAML**: `cfg/models/26/WDAM/yolo26-C2PSA_WDAM.yaml`
- **tasks.py 注册**: import (line ~23) + base_modules (line ~1607)
- **AMP 兼容**: forward 中 `torch.cuda.amp.autocast(enabled=False)` 包裹 DWT 操作
- **依赖**: pytorch-wavelets==1.3.0, PyWavelets==1.9.0 (conda env no6-ai)
- **训练状态**: 未完成过任何完整训练

## 远程训练注意事项

- 如果使用后台进程 (`&`) 保活，需 `nohup` 或 `screen`/`tmux`，否则 SSH 断开时进程被 SIGHUP 杀死
- 部署到云端后必须检查 dataset YAML 路径是否正确
- GPU 分工(本地): RTX 3080(device=0, 20GB) 用于小数据集/推理; RTX 2080 Ti(device=1, 22.5GB) 用于大数据集训练
