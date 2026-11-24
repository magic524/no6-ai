#!/usr/bin/env python3
"""Query Notion database - using the write module approach."""

import json
import sys

sys.path.insert(0, "/home/magic524/projects/no6-ai/scripts/analyze")
from notion_write import _req

# Search for all pages
result = _req("POST", "/search", {"query": "", "filter": {"property": "object", "value": "page"}})
results = result.get("results", [])
print(f"Total results: {len(results)}")

# Also try querying the database directly
db_id = "35fde3eb-6ec0-80e1-a463-000bf5e295ba"
try:
    db_result = _req("GET", f"/databases/{db_id}")
    print(f"\nDB title: {[t.get('plain_text', '') for t in db_result.get('title', [])]}")
except Exception as e:
    print(f"Can't query db directly: {e}")

if results:
    by_dataset = {}
    for r in results:
        props = r.get("properties", {})
        title_prop = props.get("实验名称", props.get("title", {}))
        titles = title_prop.get("title", [])
        name = titles[0].get("plain_text", "") if titles else "?"

        model = props.get("模型", {}).get("select", {}).get("name", "?")
        scale = props.get("YOLO尺度", {}).get("select", {}).get("name", "?")
        dataset = props.get("数据集", {}).get("select", {}).get("name", "?")
        map50 = props.get("mAP50", {}).get("number", "")

        key = f"{model} / {dataset}"
        if key not in by_dataset:
            by_dataset[key] = []
        m50 = f"{map50:.4f}" if isinstance(map50, (int, float)) else str(map50)
        by_dataset[key].append(f"  [{scale}] {name:40s} mAP50={m50}")

    for key in sorted(by_dataset.keys()):
        items = by_dataset[key]
        print(f"\n--- {key} ({len(items)} records) ---")
        for item in items:
            print(item)
else:
    print("No results. Might need different query approach.")
    # Try querying database directly
    body = {"page_size": 100}
    db_result = _req("POST", f"/databases/{db_id}/query", body)
    db_results = db_result.get("results", [])
    print(f"Direct DB query: {len(db_results)} results")
    for r in db_results[:3]:
        print(json.dumps(r.get("properties", {}), indent=2, ensure_ascii=False)[:1000])
