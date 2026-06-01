#!/usr/bin/env python3
import json
import os
import urllib.request
from pathlib import Path

_env = {}
_env_file = Path(__file__).resolve().parent / ".hermes" / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            _env[_k.strip()] = _v.strip()
API_KEY = _env.get("NOTION_API_KEY", "")
if not API_KEY:
    # fallback to ~/.hermes/.env
    _env_file = Path(os.path.expanduser("~/.hermes/.env"))
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                _env[_k.strip()] = _v.strip()
    API_KEY = _env.get("NOTION_API_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}
db_id = "4e9993ff-3bc6-4aa5-a984-91a1ac63b496"

req = urllib.request.Request(
    f"https://api.notion.com/v1/databases/{db_id}/query",
    data=json.dumps({"page_size": 25}).encode(),
    headers=HEADERS,
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read())

for r in data.get("results", []):
    props = r["properties"]
    name = props.get("实验名称", {}).get("title", [{}])[0].get("text", {}).get("content", "?")
    map50 = props.get("mAP50", {}).get("number", "?")
    ap_s = props.get("AP_s", {}).get("number", "?")
    ap_l = props.get("AP_l", {}).get("number", "?")
    page_id = r["id"]
    print(f"{name:30s} mAP50={map50!s:>6s}  AP_s={ap_s!s:>6s}  AP_l={ap_l!s:>6s}  {page_id}")

print(f"\nTotal: {len(data.get('results', []))} TinyPerson records")
