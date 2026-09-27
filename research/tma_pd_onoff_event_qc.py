#!/usr/bin/env python3
from __future__ import annotations
import re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import ezc3d

OUT=Path("research/tma_pd_onoff_results")
OUT.mkdir(parents=True,exist_ok=True)
C3D_URL="https://ndownloader.figshare.com/files/28739484"
INFO_URL="https://ndownloader.figshare.com/files/37907001"
ZIP=Path("/tmp/C3Dfiles.zip")
INFO=Path("/tmp/PDGinfo.xlsx")

def download(url,path,min_size=1):
    if path.exists() and path.stat().st_size>=min_size:return
    with requests.get(url,stream=True,timeout=180) as r:
        r.raise_for_status()
        with open(path,"wb") as f:
            for ch in r.iter_content(4*1024*1024):
                if ch:f.write(ch)

download(C3D_URL,ZIP,1_000_000_000)
download(INFO_URL,INFO,10_000)

def first_between(vals,lo,hi):
    vals=np.asarray(vals,float)
    i=np.searchsorted(vals,lo+1e-9)
    return float(vals[i]) if i<len(vals) and vals[i]<hi-1e-9 else None

def extract_annotated_cycles(path):
    c=ezc3d.c3d(str(path))
    rate=float(c["parameters"]["POINT"]["RATE"]["value"][0])
    if "EVENT" not in c["parameters"]:
        return [],rate,"no EVENT group"
    ev=c["parameters"]["EVENT"]
    contexts=[str(x).strip().upper() for x in ev["CONTEXTS"]["value"]]
    times=np.asarray(ev["TIMES"]["value"],float)
    if times.ndim!=2 or times.shape[0]<2:
        return [],rate,"invalid EVENT times"
    sec=times[1,:]
    by={k:np.array(sorted([float(t) for t,ctx in zip(sec,contexts) if ctx==k]),float) for k in ("LHS","RTO","RHS","LTO")}
    lhs=by["LHS"]; cycles=[]
    for a,b in zip(lhs[:-1],lhs[1:]):
        dur=b-a
        if not (0.55<=dur<=2.5):continue
        rto=first_between(by["RTO"],a,b)
        if rto is None:continue
        rhs=first_between(by["RHS"],rto,b)
        if rhs is None:continue
        lto=first_between(by["LTO"],rhs,b)
        if lto is None:continue
        if not (a<rto<rhs<lto<b):continue
        cycles.append({"a":a,"rto":rto,"rhs":rhs,"lto":lto,"b":b,
                       "cycle":dur,"left_stance":lto-a,"left_swing":b-lto,
                       "rate":rate})
    return cycles,rate,""

def parse_temporal(raw):
    text=raw.decode("utf-8",errors="replace")
    lines=[l for l in text.splitlines() if l.strip()]
    if len(lines)<6:return{}
    names=lines[1].strip().split("\t")
    vals=lines[-1].strip().split("\t")
    if vals and vals[0]=="1":vals=vals[1:]
    out={}
    for k,v in zip(names,vals):
        try:out[k]=float(v)
        except:pass
    return out

rows=[]
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    c3ds=[n for n in names if re.search(r"/SUB\d+_(?:on|off)_walk_\d+\.c3d$",n,re.I)]
    def key(n):
        m=re.search(r"SUB(\d+)_(on|off)_walk_(\d+)\.c3d$",n,re.I)
        return(int(m.group(1)),m.group(2).lower(),int(m.group(3)))
    c3ds=sorted(c3ds,key=key)
    tmp=Path("/tmp/walk.c3d")
    for ii,n in enumerate(c3ds,1):
        sid,cond,trial=key(n)
        with z.open(n) as src,open(tmp,"wb") as dst:
            while True:
                b=src.read(1024*1024)
                if not b:break
                dst.write(b)
        cycles,rate,err=extract_annotated_cycles(tmp)
        tname=n[:-4]+"_temporal_distance.txt"
        ref=parse_temporal(z.read(tname)) if tname in names else{}
        cyc=np.mean([c["cycle"] for c in cycles]) if cycles else np.nan
        st=np.mean([c["left_stance"] for c in cycles]) if cycles else np.nan
        sw=np.mean([c["left_swing"] for c in cycles]) if cycles else np.nan
        rows.append({
            "subject":sid,"condition":cond,"trial":trial,"rate":rate,
            "n_cycles":len(cycles),"error":err,
            "det_cycle_time":cyc,"ref_cycle_time":ref.get("Cycle_Time_Mean",np.nan),
            "cycle_abs_error":abs(cyc-ref.get("Cycle_Time_Mean",np.nan)) if np.isfinite(cyc) and "Cycle_Time_Mean" in ref else np.nan,
            "det_left_stance":st,"ref_left_stance":ref.get("Left_Stance_Time_Mean",np.nan),
            "stance_abs_error":abs(st-ref.get("Left_Stance_Time_Mean",np.nan)) if np.isfinite(st) and "Left_Stance_Time_Mean" in ref else np.nan,
            "det_left_swing":sw,"ref_left_swing":ref.get("Left_Swing_Time_Mean",np.nan),
            "swing_abs_error":abs(sw-ref.get("Left_Swing_Time_Mean",np.nan)) if np.isfinite(sw) and "Left_Swing_Time_Mean" in ref else np.nan,
        })
        if ii%150==0:print("processed",ii,"/",len(c3ds),flush=True)

tr=pd.DataFrame(rows)
tr.to_csv(OUT/"onoff_event_qc_trials.csv",index=False)
subs=[]
for sid in sorted(tr.subject.unique()):
    row={"subject":int(sid)}
    for cond in("off","on"):
        q=tr[(tr.subject==sid)&(tr.condition==cond)]
        row[f"{cond}_trials"]=len(q)
        row[f"{cond}_valid_trials"]=int((q.n_cycles>0).sum())
        row[f"{cond}_cycles"]=int(q.n_cycles.sum())
        row[f"{cond}_cycle_mae"]=float(q.cycle_abs_error.dropna().median()) if q.cycle_abs_error.notna().any() else np.nan
        row[f"{cond}_stance_mae"]=float(q.stance_abs_error.dropna().median()) if q.stance_abs_error.notna().any() else np.nan
        row[f"{cond}_swing_mae"]=float(q.swing_abs_error.dropna().median()) if q.swing_abs_error.notna().any() else np.nan
    row["paired_min_cycles"]=min(row["off_cycles"],row["on_cycles"])
    subs.append(row)
sd=pd.DataFrame(subs)
sd.to_csv(OUT/"onoff_event_qc_subjects.csv",index=False)

info=pd.read_excel(INFO)
info.to_csv(OUT/"PDGinfo.csv",index=False)

errs=tr.cycle_abs_error.dropna()
for cutoff in(8,12,16,20,24,32,40,48,64):
    sd[f"eligible_{cutoff}"]=sd.paired_min_cycles>=cutoff
lines=["# ON/OFF annotated gait-event QC","",
       f"walking C3D trials processed: {len(tr)}",
       f"subjects represented: {tr.subject.nunique()}",
       f"trials with >=1 complete annotated LHS-RTO-RHS-LTO-LHS cycle: {(tr.n_cycles>0).sum()} / {len(tr)}",
       f"total reconstructed annotated cycles: {int(tr.n_cycles.sum())}",
       f"median abs cycle-time error vs supplied temporal metric: {errs.median():.6f} s" if len(errs) else "",
       f"90th percentile abs cycle-time error: {errs.quantile(.9):.6f} s" if len(errs) else "",
       "",
       "## Paired fixed-cycle eligibility"]
for cutoff in(8,12,16,20,24,32,40,48,64):
    lines.append(f"- >= {cutoff} cycles in BOTH conditions: {int((sd.paired_min_cycles>=cutoff).sum())} subjects")
lines += ["","## Per-subject counts",sd.to_markdown(index=False)]
(OUT/"EVENT_QC.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"EVENT_QC.md").read_text(encoding="utf-8"))
