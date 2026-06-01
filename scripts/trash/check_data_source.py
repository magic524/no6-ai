#!/usr/bin/env python3
"""Check Notion data source structure."""

import json
import urllib.request

API_KEY = open("/home/magic524/projects/no6-ai/.hermes/.env").read().split("NOTION_API_KEY=")[1].split("\n")[0].strip()
BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}

data_source_id = "35fde3eb-6ec0-80e1-a463-000bf5e295ba"
print(f"Data source ID: {data_source_id}")

# Try to query the data source
try:
    req = urllib.request.Request(f"{BASE}/data-sources/{data_source_id}", headers=HEADERS)
    resp = urllib.request.urlopen(req)
    ds = json.loads(resp.read())
    print(f"Data source type: {ds.get('type')}")
    print(f"Data source name: {ds.get('name')}")
    print(f"Keys: {list(ds.keys())}")
    print(json.dumps(ds, indent=2, ensure_ascii=False)[:3000])
except urllib.error.HTTPError as e:
    print(f"Error querying data source: {e.code}")
    print(e.read().decode()[:500])
except Exception as e:
    print(f"Error: {e}")

# Maybe it's a database connection?
print("\n\nTrying database endpoint...")
try:
    req = urllib.request.Request(f"{BASE}/databases/{data_source_id}", headers=HEADERS)
    resp = urllib.request.urlopen(req)
    db = json.loads(resp.read())
    print(f"Database title: {db.get('title', [{}])[0].get('plain_text', '?')}")
    print(f"Keys: {list(db.keys())}")
    if "properties" in db:
        print(f"Properties: {list(db['properties'].keys())}")
    else:
        print("NO properties key in database response")
        print(json.dumps(db, indent=2, ensure_ascii=False)[:2000])
except urllib.error.HTTPError as e:
    print(f"Error: {e.code}")
    print(e.read().decode()[:500])
