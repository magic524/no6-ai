#!/usr/bin/env python3
"""Query Notion DB for DEGConv_DE entries."""

import json
import os
import sys
import urllib.error
import urllib.request

env_file = os.path.join(os.path.dirname(__file__) or ".", ".hermes", ".env")
if not os.path.exists(env_file):
    env_file = os.path.expanduser("~/projects/no6-ai/.hermes/.env")

env = {}
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k.strip()] = v.strip()

API_KEY = env.get("NOTION_API_KEY", "")
if not API_KEY:
    print("NO_API_KEY")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}
DB_ID = "1a5de922117980be9d4ac9d0c2c84a43"

# Try database query endpoint
body = {"filter": {"property": "模型", "select": {"equals": "DEGConv_DE"}}}
req = urllib.request.Request(
    f"https://api.notion.com/v1/databases/{DB_ID}/query", data=json.dumps(body).encode(), headers=HEADERS, method="POST"
)
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        print(f"Total results: {len(data.get('results', []))}")
        for r in data.get("results", []):
            props = r.get("properties", {})
            name = props.get("实验名称", {}).get("title", [{}])[0].get("plain_text", "?")
            ds = props.get("数据集", {}).get("select", {}).get("name", "?")
            sc = props.get("YOLO尺度", {}).get("select", {}).get("name", "?")
            m50 = props.get("mAP50", {}).get("number", "?")
            m95 = props.get("mAP50-95", {}).get("number", "?")
            print(f"{name:40s} | {ds:15s} | {sc:5s} | mAP50={m50} | mAP50-95={m95}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}: {body}")
    # Fallback: search
    search_body = {"query": "DEGConv_DE", "filter": {"property": "object", "value": "page"}}
    req2 = urllib.request.Request(
        "https://api.notion.com/v1/search", data=json.dumps(search_body).encode(), headers=HEADERS, method="POST"
    )
    with urllib.request.urlopen(req2) as resp2:
        data2 = json.loads(resp2.read())
        for r in data2.get("results", []):
            props = r.get("properties", {})
            name = props.get("实验名称", {}).get("title", [{}])[0].get("plain_text", "?")
            ds = props.get("数据集", {}).get("select", {}).get("name", "?")
            sc = props.get("YOLO尺度", {}).get("select", {}).get("name", "?")
            m50 = props.get("mAP50", {}).get("number", "?")
            print(f"{name:40s} | {ds:15s} | {sc:5s} | mAP50={m50}")
        print(f"Search total: {len(data2.get('results', []))}")
