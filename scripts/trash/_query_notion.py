#!/usr/bin/env python3
"""Query Notion RSOD records."""
import json, urllib.request
from pathlib import Path

env_file = Path.home() / '.hermes' / '.env'
env = {}
for line in env_file.read_text().splitlines():
    if '=' in line and not line.startswith('#'):
        k, v = line.strip().split('=', 1)
        env[k.strip()] = v.strip()
API_KEY = env.get('NOTION_API_KEY', '')

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Notion-Version': '2022-06-28',
    'Content-Type': 'application/json',
}
body = {
    'page_size': 25,
    'filter': {
        'property': '数据集',
        'select': {'equals': 'RSOD'}
    }
}
req = urllib.request.Request(
    'https://api.notion.com/v1/databases/4e9993ff-3bc6-4aa5-a984-91a1ac63b496/query',
    data=json.dumps(body).encode(), headers=headers)
resp = json.loads(urllib.request.urlopen(req).read())
for r in resp.get('results', []):
    props = r.get('properties', {})
    name = props.get('实验名称', {}).get('title', [{}])[0].get('plain_text', '')
    model = props.get('模型', {}).get('select', {}).get('name', '')
    scale = props.get('YOLO尺度', {}).get('select', {}).get('name', '')
    print(f'{name:55s} | 模型={model:10s} | 尺度={scale}')
