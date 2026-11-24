#!/bin/bash
# Cloud GPU monitor — checks both AutoDL instances
# Silent when both training; alerts when any GPU goes idle
# Called by cron every 6 hours

CLOUD1_PORT=44908
CLOUD2_PORT=10951

check_cloud() {
  local port=$1
  ssh -o ConnectTimeout=10 -p $port root@connect.nmb2.seetacloud.com "
        util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        mem=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        temp=\$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
        yolo_count=\$(ps aux | grep 'yolo detect train' | grep -v grep | wc -l)
        disk=\$(df -h /root/autodl-tmp/ 2>/dev/null | tail -1 | awk '{print \$3\"/\"\$2, \"(\"\$5\")\"}')
        echo \"\$util|\$mem|\$temp|\$yolo_count|\$disk\"
    " 2> /dev/null
}

c1=$(check_cloud $CLOUD1_PORT)
c2=$(check_cloud $CLOUD2_PORT)

IFS='|' read -r c1_util c1_mem c1_temp c1_yolo c1_disk <<< "$c1"
IFS='|' read -r c2_util c2_mem c2_temp c2_yolo c2_disk <<< "$c2"

c1_idle=false
c2_idle=false
[ "$c1_yolo" -eq 0 ] && c1_idle=true
[ "$c2_yolo" -eq 0 ] && c2_idle=true

# Silent exit if both still training
[ "$c1_idle" = false ] && [ "$c2_idle" = false ] && exit 0

# ── At least one cloud is idle — alert ──
echo "🔍 云 GPU 状态变动 — 有 GPU 已空闲"
echo ""

if [ "$c1_idle" = true ]; then
  echo "⏹️  云1 (DEGConv_DE VisDrone) 已空闲"
  echo "   最终: disk=${c1_disk}"
else
  c1_epoch=$(ssh -o ConnectTimeout=10 -p $CLOUD1_PORT root@connect.nmb2.seetacloud.com "tail -1 /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/VisDrone/yolo11x-DE/results.csv 2>/dev/null | cut -d',' -f1")
  c1_map=$(ssh -o ConnectTimeout=10 -p $CLOUD1_PORT root@connect.nmb2.seetacloud.com "tail -1 /root/autodl-tmp/no6-ai/runs/detect/DEGConv_DE/VisDrone/yolo11x-DE/results.csv 2>/dev/null | awk -F',' '{print \$8}'")
  echo "🟢 云1 DEGConv_DE VisDrone x: epoch ${c1_epoch:-?}, mAP50=${c1_map:-?}, util=${c1_util}%, disk=${c1_disk}"
fi

if [ "$c2_idle" = true ]; then
  echo "⏹️  云2 (BAV2 TinyPerson) 已空闲"
  echo "   最终: disk=${c2_disk}"
else
  c2_epoch=$(ssh -o ConnectTimeout=10 -p $CLOUD2_PORT root@connect.nmb2.seetacloud.com "tail -1 /root/autodl-tmp/no6-ai/runs/detect/BinaryAttentionV2/TinyPerson/yolo11x/results.csv 2>/dev/null | cut -d',' -f1")
  c2_map=$(ssh -o ConnectTimeout=10 -p $CLOUD2_PORT root@connect.nmb2.seetacloud.com "tail -1 /root/autodl-tmp/no6-ai/runs/detect/BinaryAttentionV2/TinyPerson/yolo11x/results.csv 2>/dev/null | awk -F',' '{print \$8}'")
  echo "🟢 云2 BAV2 TinyPerson x: epoch ${c2_epoch:-?}, mAP50=${c2_map:-?}, util=${c2_util}%, disk=${c2_disk}"
fi

echo ""
echo "========== 当前实验状态 =========="
echo ""
echo "✅ 已完成的实验（需 val + eval + Notion）:"
echo "  - DEGConv_DE VisDrone s/m/l/x（云1）"
echo "  - BinaryAttentionV2 TinyPerson n/s/m/l/x（云2）"
echo ""
echo "📋 待办（按优先级）:"
echo "  [1] 🔴 云训练结果 → val → eval_ap → Notion"
echo "  [2] 🟡 FAAFusion VisDrone n-scale 重训"
echo "  [3] 🟡 FAAFusion NightDrone l/x 补训"
echo "  [4] 🟢 DEGConv V3 Full RSOD s/m/l/x 补训"
echo "  [5] 🟢 BAV2 VisDrone 全尺度训练"
echo ""
echo "💡 空闲云建议: 跑 VisDrone 数据集（最具区分度）"
