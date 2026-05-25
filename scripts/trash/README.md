# 评估脚本 (scripts/eval/)

| 文件 | 说明 | 状态 |
|------|------|------|
| `batch_eval_nwpu.sh` | NWPU_VHR-10 批量验证 | ✅ 可用 |
| `batch_eval_rsod.sh` | RSOD 批量验证 | ✅ 可用 |
| `batch_eval_rsod_yolo11.sh` | RSOD yolo11 批量验证 | ✅ 可用 |
| `batch_eval_tinyperson.sh` | TinyPerson 批量验证 | ✅ 可用 |
| `batch_postprocess_visdrone.sh` | VisDrone 训练后处理（验证+eval+FPS+Notion） | ✅ 可用 |
| `post_process.sh` | 通用训练后处理脚本 | ✅ 可用 |

## 使用

```bash
# 单数据集批量验证
bash scripts/eval/batch_eval_rsod.sh

# 训练后全自动处理（val + eval_ap + FPS + Notion写入）
bash scripts/eval/post_process.sh
```
