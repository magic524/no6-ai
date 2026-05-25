#!/usr/bin/env python3
"""Add select option 'yolo11+DEGConv' to Notion DB model column."""
import json, os, urllib.request

API_KEY = open('/home/magic524/projects/no6-ai/.hermes/.env').read().split('NOTION_API_KEY=')[1].split('\n')[0].strip()
BASE = 'https://api.notion.com/v1'
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Notion-Version': '2025-09-03',
    'Content-Type': 'application/json',
}

db_id = '4e9993ff3bc64aa5a98491a1ac63b496'

# First check current state by getting the DB
req = urllib.request.Request(f'{BASE}/databases/{db_id}', headers=HEADERS)
resp = urllib.request.urlopen(req)
db = json.loads(resp.read())

# Check model column
model_prop = None
for name, prop in db['properties'].items():
    if '模型' in name or 'model' in name.lower():
        model_prop = prop
        print(f"Found model column: '{name}' type={prop.get('type')}")
        break

if not model_prop:
    print("No model column found!")
    print("Properties:", list(db['properties'].keys()))
    exit(1)

options = model_prop.get('select', {}).get('options', [])
existing = [o['name'] for o in options]
print(f"Existing options: {existing}")

if 'yolo11+DEGConv' not in existing:
    new_option = {"name": "yolo11+DEGConv", "color": "green"}
    options.append(new_option)
    
    update_body = {
        "properties": {
            "模型": {
                "select": {
                    "options": options
                }
            }
        }
    }
    
    req = urllib.request.Request(
        f'{BASE}/databases/{db_id}',
        data=json.dumps(update_body).encode(),
        headers=HEADERS,
        method='PATCH'
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print(f"Added 'yolo11+DEGConv'. Status OK.")
else:
    print("'yolo11+DEGConv' already exists.")
