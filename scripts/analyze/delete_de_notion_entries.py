#!/usr/bin/env python3
"""Delete the 10 incorrect DEGConv_DE Notion entries."""

import json
import os
import sys
import urllib.error
import urllib.request

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

# Page IDs extracted from write output URLs
# URL format: https://app.notion.com/p/{name}-{uuid}
page_ids = [
    "38ade3eb-6ec0-8182-8e7e-fd8d5912cfa4",  # yolo11n-DEGConv_DE-RSOD
    "38ade3eb-6ec0-811f-8d34-fb66eb454537",  # yolo11s-DEGConv_DE-RSOD
    "38ade3eb-6ec0-81d8-b76e-de1164cb6285",  # yolo11m-DEGConv_DE-RSOD
    "38ade3eb-6ec0-8119-bbd6-e426dc6a7cba",  # yolo11l-DEGConv_DE-RSOD
    "38ade3eb-6ec0-8191-b056-d4e8f89c4408",  # yolo11x-DEGConv_DE-RSOD
    "38ade3eb-6ec0-8199-b32e-c2b1b680741b",  # yolo11n-DEGConv_DE-NWPU_VHR-10
    "38ade3eb-6ec0-811c-a3eb-f8f30424ddee",  # yolo11s-DEGConv_DE-NWPU_VHR-10
    "38ade3eb-6ec0-8115-81c3-df63ac3978fb",  # yolo11m-DEGConv_DE-NWPU_VHR-10
    "38ade3eb-6ec0-81fe-a612-cb12d9fd63b7",  # yolo11l-DEGConv_DE-NWPU_VHR-10
    "38ade3eb-6ec0-8145-906b-d6342ee99da6",  # yolo11x-DEGConv_DE-NWPU_VHR-10
]

for pid in page_ids:
    body = json.dumps({"archived": True}).encode()
    req = urllib.request.Request(f"https://api.notion.com/v1/pages/{pid}", data=body, headers=HEADERS, method="PATCH")
    try:
        with urllib.request.urlopen(req) as resp:
            r = json.loads(resp.read())
            title = r.get("properties", {}).get("实验名称", {}).get("title", [{}])[0].get("plain_text", "?")
            print(f"Deleted: {title}")
    except urllib.error.HTTPError as e:
        print(f"ERROR deleting {pid}: {e.code} {e.read().decode()}")

print("Done")
