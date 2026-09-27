#!/usr/bin/env python3
from __future__ import annotations
import zipfile
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

lines=["# Raw C3D archive probe v2","",f"archive_bytes: {ZIP.stat().st_size}",""]
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    (OUT/"C3D_MEMBERS.txt").write_text("\n".join(names),encoding="utf-8")
    c3d=[n for n in names if n.lower().endswith(".c3d")]
    csv=[n for n in names if n.lower().endswith(".csv")]
    txt=[n for n in names if n.lower().endswith(".txt")]
    lines += [f"members: {len(names)}",f"c3d: {len(c3d)}",f"csv: {len(csv)}",f"txt: {len(txt)}",""]

    for cond in ("off","on"):
        cand=[n for n in c3d if f"C3Dfiles/SUB01_{cond}/SUB01_{cond}_walk_" in n]
        lines += [f"## SUB01 {cond} walk C3D candidates ({len(cand)})"]
        lines += [f"- {n}" for n in cand[:5]]
        if cand:
            n=sorted(cand)[0]
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
                rate=float(c["parameters"]["POINT"]["RATE"]["value"][0])
                pts=c["data"]["points"]
                lines += [
                    f"sample_walk_file: {n}",
                    f"point_rate: {rate}; point_frames: {pts.shape[2]}; n_points: {len(pl)}",
                    "point_labels: "+", ".join(pl),
                ]
            except Exception as e:
                lines.append(f"ezc3d error for {n}: {e!r}")

        tnames=sorted([n for n in txt if f"C3Dfiles/SUB01_{cond}/SUB01_{cond}_walk_" in n and n.endswith("_temporal_distance.txt")])
        lines += ["",f"## SUB01 {cond} temporal_distance samples ({len(tnames)})"]
        for n in tnames[:3]:
            raw=z.read(n)
            text=raw.decode("utf-8",errors="replace")
            lines += [f"### {n}","~~~",text[:8000],"~~~"]

        lnames=sorted([n for n in csv if f"C3Dfiles/SUB01_{cond}/SUB01_{cond}_walk_" in n and n.endswith("_linear_kinematics.csv")])
        if lnames:
            raw=z.read(lnames[0]).decode("utf-8",errors="replace")
            lines += ["",f"## Linear kinematics header sample {lnames[0]}","~~~",raw[:5000],"~~~"]

(OUT/"C3D_PROBE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"C3D_PROBE.md").read_text(encoding="utf-8"))
