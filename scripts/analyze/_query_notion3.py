#!/usr/bin/env python3
"""Check specific records in Notion."""
import sys, json
sys.path.insert(0, '/home/magic524/projects/no6-ai/scripts/analyze')
from notion_write import _req

# Search for YOLO26 NightDrone
for query in ['yolo26-NightDrone', 'yolo26n-Night', 'yolo26x-Night', 'DEGConv_V3Full-RSOD']:
    result = _req("POST", "/search", {
        "query": query,
        "filter": {"property": "object", "value": "page"}
    })
    items = result.get('results', [])
    for r in items:
        props = r.get('properties', {})
        title_prop = props.get('实验名称', props.get('title', {}))
        titles = title_prop.get('title', [])
        name = titles[0].get('plain_text', '') if titles else '?'
        model = props.get('模型', {}).get('select', {}).get('name', '?')
        scale = props.get('YOLO尺度', {}).get('select', {}).get('name', '?')
        dataset = props.get('数据集', {}).get('select', {}).get('name', '?')
        map50 = props.get('mAP50', {}).get('number', '')
        parent_id = r.get('parent', {}).get('data_source_id', '')
        print(f"  {name:45s} model={model:15s} scale={scale:3s} dataset={dataset:12s} mAP50={map50}  (ds={parent_id[:20]})")
    if not items:
        print(f"'{query}': 0 results")
