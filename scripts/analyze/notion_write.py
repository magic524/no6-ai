#!/usr/bin/env python3
"""no6-ai Notion helper — write experiment results to the Baseline database."""
import json, os, sys, urllib.request, urllib.error

from pathlib import Path as _Path
_env = {}
_env_file = _Path(__file__).resolve().parent.parent.parent / ".hermes" / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            _env[_k.strip()] = _v.strip()
API_KEY = _env.get("NOTION_API_KEY", "")
BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}

# ── config: update these when the database changes ──
DATA_SOURCE_ID = "35fde3eb-6ec0-80e1-a463-000bf5e295ba"


def _req(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        print(f"ERROR {e.code}: {err.get('message', str(e))}", file=sys.stderr)
        sys.exit(1)


def write_experiment(name, model, scale, dataset, map50, map50_95, ap_s, ap_m, ap_l, fps, params, gflops):
    """Create or update an experiment row in the Notion database."""
    # search for existing page with same name
    existing = _req("POST", "/search", {"query": name, "filter": {"property": "object", "value": "page"}})
    page_id = None
    for r in existing.get("results", []):
        props = r.get("properties", {})
        title_prop = props.get("实验名称", {}) or props.get("title", {})
        titles = title_prop.get("title", [])
        if titles and titles[0].get("plain_text", "") == name:
            page_id = r["id"]
            break

    properties = {
        "实验名称": {"title": [{"text": {"content": name}}]},
        "模型": {"select": {"name": model}},
        "YOLO尺度": {"select": {"name": scale}},
        "数据集": {"select": {"name": dataset}},
        "mAP50": {"number": map50},
        "mAP50-95": {"number": map50_95},
        "AP_s": {"number": ap_s},
        "AP_m": {"number": ap_m},
        "AP_l": {"number": ap_l},
        "FPS": {"number": fps},
        "Params(M)": {"number": params},
        "GFLOPs": {"number": gflops},
    }

    if page_id:
        _req("PATCH", f"/pages/{page_id}", {"properties": {k: v for k, v in properties.items() if k != "实验名称"}})
        print(f"Updated: {name}")
    else:
        result = _req("POST", "/pages", {
            "parent": {"data_source_id": DATA_SOURCE_ID},
            "properties": properties,
        })
        print(f"Created: {result.get('url', '?')}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--map50", type=float, required=True)
    p.add_argument("--map50-95", type=float, required=True)
    p.add_argument("--ap-s", type=float, required=True)
    p.add_argument("--ap-m", type=float, required=True)
    p.add_argument("--ap-l", type=float, required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--params", type=float, required=True)
    p.add_argument("--gflops", type=float, required=True)
    args = p.parse_args()
    write_experiment(args.name, args.model, args.scale, args.dataset, args.map50, args.map50_95,
                     args.ap_s, args.ap_m, args.ap_l, args.fps, args.params, args.gflops)
