#!/usr/bin/env python3
"""Complete rewrite of the Notion tutorial page."""

import json
import subprocess
from pathlib import Path

env = Path("/home/magic524/projects/no6-ai/.hermes/.env").read_bytes()
start = env.find(b"NOTION_API_KEY")
end = env.find(b"\n", start)
TOKEN = env[start:end].split(b"=")[1].decode()
PAGE_ID = "366de3eb-6ec0-811a-a5a2-f1816917b1e9"
DATA_SOURCE_ID = "35fde3eb-6ec0-80e1-a463-000bf5e295ba"


def curl_patch(blocks):
    """Append blocks to page using PATCH."""
    payload = json.dumps({"children": blocks})
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "PATCH",
            f"https://api.notion.com/v1/blocks/{PAGE_ID}/children",
            "-H",
            f"Authorization: Bearer {TOKEN}",
            "-H",
            "Notion-Version: 2025-09-03",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload,
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    if "results" in data:
        return len(data["results"])
    else:
        print(f"ERROR: {data}")
        return 0


def t(text, bold=False, code=False):
    ann = {}
    if bold:
        ann["bold"] = True
    if code:
        ann["code"] = True
    return {"type": "text", "text": {"content": text}, "annotations": ann}


def p(text, bold=False):
    return {"type": "paragraph", "paragraph": {"rich_text": [t(text, bold=bold)]}}


def h2(text):
    return {"type": "heading_2", "heading_2": {"rich_text": [t(text, bold=True)]}}


def h3(text):
    return {"type": "heading_3", "heading_3": {"rich_text": [t(text, bold=True)]}}


def b(text):
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [t(text)]}}


def n(text):
    return {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [t(text)]}}


def code(text, lang="python"):
    return {"type": "code", "code": {"rich_text": [t(text)], "language": lang}}


def div():
    return {"type": "divider", "divider": {}}


def callout(text, emoji="💡"):
    return {"type": "callout", "callout": {"rich_text": [t(text)], "icon": {"emoji": emoji}}}


blocks = [
    p("本文档记录了 Notion Internal Integration Token 的配置方法及完整凭证，供私人 Agent 直接调用。"),
    div(),
    h2("1️⃣ Token 凭证"),
    p("以下为当前可用的 Notion Internal Integration Token（已关联工作空间，具备 Read/Insert/Update 权限，直接使用）："),
    code(TOKEN, "plain text"),
    p("无需额外创建 Integration。Token 已永久有效，更换 Token 时更新本文档即可。"),
    div(),
    h2("2️⃣ 配置方式"),
    h3("方式 A：Hermes Agent 本地环境变量"),
    p("Agent 将 Token 注入运行环境，脚本通过 os.environ 读取："),
    code('export NOTION_API_KEY="ntn_59...ghA"  # 替换为上方 Token', "bash"),
    h3("方式 B：项目 .hermes/.env 文件（本地脚本用）"),
    p("在项目根目录的 .hermes/.env 中添加："),
    code(f"NOTION_API_KEY={TOKEN}", "bash"),
    p("⚠️ .env 文件已配置跳过 git，安全无忧。多项目共用可统一放在 ~/.hermes/.env。"),
    div(),
    h2("3️⃣ 将 Integration 连接到目标数据库"),
    p("每个需要用到的 Notion 数据库或页面，都需要手动授权一次："),
    n("打开目标 Notion 页面或数据库"),
    n("点击右上角 ⋮ → Add connections"),
    n("搜索并选择你的 Integration 名称"),
    callout("连接一次后永久生效，后续 Agent 可直接通过 REST API 读写，不受平台账号切换影响。"),
    div(),
    h2("4️⃣ Agent 调用方式"),
    p("Agent 读取本文档获取 Token 后，通过 Notion REST API 进行操作。核心代码如下："),
    code(
        """import json, os, urllib.request

NOTION_TOKEN = os.environ["NOTION_API_KEY"]  # 或从本文档硬编码
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}
BASE = "https://api.notion.com/v1"

def notion_req(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())""",
        "python",
    ),
    p("关键 API 路径："),
    b("查询数据库/页面：POST /search"),
    b("读取页面内容：GET /blocks/{id}/children"),
    b("创建页面：POST /pages（parent.data_source_id）"),
    b("更新属性：PATCH /pages/{id}"),
    b("追加内容：PATCH /blocks/{id}/children"),
    div(),
    h2("5️⃣ 实验数据库（Baseline）配置"),
    p("实验记录数据库的数据源 ID，Agent 直接使用："),
    code(f'DATA_SOURCE_ID = "{DATA_SOURCE_ID}"', "python"),
    code(
        "数据库 URL：https://www.notion.so/368de3eb6ec080b9b724cb585eaf7b65?v=368de3eb6ec081ddb241000c9f6c362a",
        "plain text",
    ),
    p('此数据库用于记录 Baseline + 消融实验的各项指标（mAP50/AP_s/m/l/FPS/Params/GFLOPs），类型列区分"基线"和"消融"。'),
    div(),
    h2("6️⃣ 注意事项"),
    b("Token 明文存储，不要提交到 git（已在 .gitignore 中排除）"),
    b("每个数据库需要手动 Add connections 授权一次"),
    b("API Version: 2025-09-03（版本号写死在 HEADERS 中）"),
    b("速率限制：平均 3 请求/秒，突发可到 30 请求/秒"),
    b("⚠️ Notion Search API 对 Integration Token 返回 0 结果（已知限制），创建后需通过返回的 page_id 直接 fetch 验证"),
    callout("如需更换 Token：在 https://www.notion.so/my-integrations 重新生成，然后更新本文档第 1️⃣ 节的凭证。"),
]

# Append blocks in batches of 10 (Notion limit is 100 per request)
batch_size = 10
total = 0
for i in range(0, len(blocks), batch_size):
    batch = blocks[i : i + batch_size]
    count = curl_patch(batch)
    total += count
    print(f"Batch {i // batch_size + 1}: added {count} blocks")

print(f"\n✅ Total: {total} blocks added")
