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

lines=["# Raw C3D archive probe v3","",f"archive_bytes: {ZIP.stat().st_size}",""]
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    c3d=[n for n in names if n.lower().endswith(".c3d")]
    for cond in ("off","on"):
        cand=sorted([n for n in c3d if f"C3Dfiles/SUB01_{cond}/SUB01_{cond}_walk_" in n])
        lines += [f"## SUB01 {cond} walking C3D"]
        for n in cand[:3]:
            dest=Path("/tmp")/Path(n).name
            with z.open(n) as src, open(dest,"wb") as dst:
                while True:
                    b=src.read(1024*1024)
                    if not b: break
                    dst.write(b)
            import ezc3d
            c=ezc3d.c3d(str(dest))
            lines += [f"### {n}", "parameter_groups: "+", ".join(c["parameters"].keys())]
            if "EVENT" in c["parameters"]:
                ev=c["parameters"]["EVENT"]
                lines.append("EVENT keys: "+", ".join(ev.keys()))
                for key,val in ev.items():
                    try:
                        lines.append(f"{key}: {val.get('value')}")
                    except Exception:
                        lines.append(f"{key}: {val}")
            else:
                lines.append("EVENT group: absent")
            for gname in c["parameters"].keys():
                if any(x in gname.upper() for x in ("EVENT","ANALYSIS","TRIAL","PROCESSING")) and gname!="EVENT":
                    lines.append(f"{gname} keys: "+", ".join(c["parameters"][gname].keys()))
                    for key,val in c["parameters"][gname].items():
                        try:
                            v=val.get("value")
                            s=str(v)
                            if len(s)>2500: s=s[:2500]+"..."
                            lines.append(f"{gname}.{key}: {s}")
                        except Exception:
                            pass

(OUT/"C3D_PROBE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"C3D_PROBE.md").read_text(encoding="utf-8"))
