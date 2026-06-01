#!/usr/bin/env python3
"""Query Notion to verify DEGConv data was written correctly."""

import json
import urllib.request

API_KEY = open("/home/magic524/projects/no6-ai/.hermes/.env").read().split("NOTION_API_KEY=")[1].split("\n")[0].strip()
BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}

# We need to search for DEGConv pages
# Since we can't query the DB directly, let's search
body = {"query": "DEGConv", "filter": {"value": "page", "property": "object"}}
req = urllib.request.Request(f"{BASE}/search", data=json.dumps(body).encode(), headers=HEADERS, method="POST")
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
results = data.get("results", [])
print(f"Found {len(results)} DEGConv pages")

for r in results:
    props = r.get("properties", {})
    name = ""
    for pname, pval in props.items():
        if pval.get("type") == "title":
            titles = pval.get("title", [])
            if titles:
                name = titles[0].get("plain_text", "")
                break

    model = ""
    yolo_scale = ""
    dataset = ""
    map50 = ""
    ap_s = ""
    gflops = ""
    fps = ""

    for pname, pval in props.items():
        ptype = pval.get("type", "")
        if ptype == "select":
            s = pval.get("select")
            val = s.get("name", "") if s else ""
            if "模型" in pname or "model" in pname.lower():
                model = val
            elif "尺度" in pname or "scale" in pname.lower():
                yolo_scale = val
            elif "数据集" in pname or "dataset" in pname.lower():
                dataset = val
        elif ptype == "number":
            val = pval.get("number", "")
            if "mAP50" in pname and "95" not in pname:
                map50 = val
            elif "AP_s" in pname:
                ap_s = val
            elif "GFLOP" in pname:
                gflops = val
            elif "FPS" in pname:
                fps = val

    bad = []
    gflops_s = str(gflops) if gflops is not None else ""
    ap_s_s = str(ap_s) if ap_s is not None else ""
    fps_s = str(fps) if fps is not None else ""
    map50_s = str(map50) if map50 is not None else ""

    bad = []
    try:
        if gflops is not None and float(gflops) == 0:
            bad.append(f"GFLOPs={gflops}")
    except:
        pass
    try:
        if ap_s is not None and float(ap_s) == 0:
            bad.append(f"AP_s={ap_s}")
    except:
        pass

    flag = " ⚠️ " + ", ".join(bad) if bad else " ✅"
    print(
        f"  {name:45s} model={model:20s} mAP50={map50_s:>8s} GFLOPs={gflops_s:>6s} AP_s={ap_s_s:>6s} FPS={fps_s:>6s}{flag}"
    )
