#!/usr/bin/env python3
from __future__ import annotations
import json, os, zipfile
from pathlib import Path
import requests

ARTICLE_ID = 14896881
OUT = Path("research/tma_pd_onoff_results")
OUT.mkdir(parents=True, exist_ok=True)

r = requests.get(f"https://api.figshare.com/v2/articles/{ARTICLE_ID}", timeout=60)
r.raise_for_status()
meta = r.json()
(OUT / "figshare_article.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

rows = []
for f in meta.get("files", []):
    rows.append({
        "id": f.get("id"),
        "name": f.get("name"),
        "size": f.get("size"),
        "download_url": f.get("download_url"),
        "computed_md5": f.get("computed_md5"),
    })

lines = ["# Figshare ON/OFF gait dataset probe", "", f"Article: {ARTICLE_ID}", ""]
for x in rows:
    lines.append(f"- {x['name']} | id={x['id']} | size={x['size']} | {x['download_url']}")

# Download only compact metadata / processed-gait archives during the probe.
downloaded = []
for x in rows:
    name = str(x.get("name") or "")
    size = int(x.get("size") or 0)
    low = name.lower()
    wanted = (
        size <= 5_000_000
        or "pdginfo" in low
        or ("gait" in low and size <= 250_000_000)
    )
    if not wanted:
        continue
    url = x.get("download_url") or f"https://api.figshare.com/v2/file/download/{x['id']}"
    dest = OUT / name
    with requests.get(url, stream=True, timeout=120) as rr:
        rr.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in rr.iter_content(chunk_size=1024*1024):
                if chunk:
                    fh.write(chunk)
    downloaded.append(dest)
    lines.append(f"  - downloaded: {name} ({dest.stat().st_size} bytes)")
    if zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as z:
            names = z.namelist()
        (OUT / f"{name}.members.txt").write_text("\n".join(names), encoding="utf-8")
        lines.append(f"    zip members: {len(names)}; first 30:")
        lines.extend([f"      - {n}" for n in names[:30]])

(OUT / "PROBE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((OUT / "PROBE.md").read_text(encoding="utf-8"))
