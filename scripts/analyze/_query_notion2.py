#!/usr/bin/env python3
"""Query Notion - search for experiment records."""
import sys, json
sys.path.insert(0, '/home/magic524/projects/no6-ai/scripts/analyze')
from notion_write import _req

# Search specifically for experiment names  
for query in ['yolo11n-DEGConv', 'yolo11n-FAAFusion', 'yolo11n-Baseline', 'yolo26n-Baseline']:
    result = _req("POST", "/search", {
        "query": query,
        "filter": {"property": "object", "value": "page"}
    })
    items = result.get('results', [])
    print(f"\n=== Search '{query}' => {len(items)} results ===")
    for r in items:
        props = r.get('properties', {})
        title_prop = props.get('实验名称', props.get('title', {}))
        titles = title_prop.get('title', [])
        name = titles[0].get('plain_text', '') if titles else '?'
        model = props.get('模型', {}).get('select', {}).get('name', '?')
        scale = props.get('YOLO尺度', {}).get('select', {}).get('name', '?')
        dataset = props.get('数据集', {}).get('select', {}).get('name', '?')
        map50 = props.get('mAP50', {}).get('number', '')
        parent = r.get('parent', {})
        print(f"  {name:45s} model={model:15s} scale={scale:3s} dataset={dataset:10s} map50={map50}")
