#!/usr/bin/env python3
from __future__ import annotations
import json, zipfile
from pathlib import Path
import requests

OUT=Path("research/tma_pd_onoff_results")
OUT.mkdir(parents=True, exist_ok=True)
URL="https://ndownloader.figshare.com/files/28739484"
ZIP=Path("/tmp/C3Dfiles.zip")

if not ZIP.exists() or ZIP.stat().st_size < 1_000_000_000:
    with requests.get(URL,stream=True,timeout=180) as r:
        r.raise_for_status()
        with open(ZIP,"wb") as f:
            for chunk in r.iter_content(4*1024*1024):
                if chunk: f.write(chunk)

lines=["# Raw C3D archive probe","",f"archive_bytes: {ZIP.stat().st_size}",""]
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    (OUT/"C3D_MEMBERS.txt").write_text("\n".join(names),encoding="utf-8")
    lines += [f"members: {len(names)}","", "## First 100 members"]
    lines += [f"- {n}" for n in names[:100]]
    c3d=[n for n in names if n.lower().endswith(".c3d")]
    csv=[n for n in names if n.lower().endswith(".csv")]
    txt=[n for n in names if n.lower().endswith(".txt")]
    lines += ["",f"c3d: {len(c3d)}",f"csv: {len(csv)}",f"txt: {len(txt)}"]
    for cond in ("off","on"):
        cand=[n for n in c3d if "SUB01" in n.upper() and f"_{cond}_" in n.lower()]
        if not cand:
            cand=[n for n in c3d if "SUB01" in n.upper() and cond in n.lower()]
        lines += ["",f"## SUB01 {cond} C3D candidates"]
        lines += [f"- {n}" for n in cand[:20]]
        for n in cand[:1]:
            dest=Path("/tmp")/Path(n).name
            with z.open(n) as src, open(dest,"wb") as dst:
                while True:
                    b=src.read(1024*1024)
                    if not b: break
                    dst.write(b)
            try:
                import ezc3d
                c=ezc3d.c3d(str(dest))
                pl=list(c["parameters"]["POINT"]["LABELS"]["value"])
                al=list(c["parameters"]["ANALOG"]["LABELS"]["value"])
                rate=float(c["parameters"]["POINT"]["RATE"]["value"][0])
                arate=float(c["parameters"]["ANALOG"]["RATE"]["value"][0])
                pts=c["data"]["points"]
                ana=c["data"]["analogs"]
                lines += [
                    f"sample_file: {n}",
                    f"point_rate: {rate}; point_frames: {pts.shape[2]}; n_points: {len(pl)}",
                    f"analog_rate: {arate}; analog_frames: {ana.shape[2]}; n_analogs: {len(al)}",
                    "point_labels: "+", ".join(pl),
                    "analog_labels: "+", ".join(al),
                ]
            except Exception as e:
                lines.append(f"ezc3d error for {n}: {e!r}")

(OUT/"C3D_PROBE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"C3D_PROBE.md").read_text(encoding="utf-8"))
