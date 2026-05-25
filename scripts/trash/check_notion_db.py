#!/usr/bin/env python3
import json, os, urllib.request

API_KEY = open('/home/magic524/projects/no6-ai/.hermes/.env').read().split('NOTION_API_KEY=')[1].split('\n')[0].strip()
BASE = 'https://api.notion.com/v1'
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Notion-Version': '2025-09-03',
    'Content-Type': 'application/json',
}

db_id = '4e9993ff3bc64aa5a98491a1ac63b496'

# Query pages in database
body = {"page_size": 5}
req = urllib.request.Request(
    f'{BASE}/databases/{db_id}/query',
    data=json.dumps(body).encode(),
    headers=HEADERS,
    method='POST'
)
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
results = data.get('results', [])
print(f"Total pages in DB: {data.get('has_more', False)} (has_more)")
print(f"Results count: {len(results)}")

if results:
    r = results[0]
    props = r.get('properties', {})
    print("\nProperties from first page:")
    for name, prop in props.items():
        ptype = prop.get('type', '?')
        print(f"  '{name}' -> type={ptype}")
        if ptype == 'select':
            s = prop.get('select', {})
            print(f"    value: {s.get('name', 'None') if s else 'None'}")
        elif ptype == 'multi_select':
            vals = prop.get('multi_select', [])
            print(f"    values: {[v['name'] for v in vals]}")
        elif ptype == 'number':
            print(f"    value: {prop.get('number', 'None')}")
        elif ptype == 'title':
            t = prop.get('title', [])
            print(f"    value: {t[0].get('plain_text', '') if t else ''}")

# Also dump full properties of first page
print("\n\nFull first page properties JSON:")
print(json.dumps(results[0].get('properties', {}), indent=2, ensure_ascii=False)[:5000] if results else "NONE")
