#!/usr/bin/env python3
from __future__ import annotations
import io, re, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import ezc3d
from scipy.signal import find_peaks, savgol_filter

OUT=Path("research/tma_pd_onoff_results")
OUT.mkdir(parents=True,exist_ok=True)
C3D_URL="https://ndownloader.figshare.com/files/28739484"
INFO_URL="https://ndownloader.figshare.com/files/37907001"
ZIP=Path("/tmp/C3Dfiles.zip")
INFO=Path("/tmp/PDGinfo.xlsx")

def download(url,path,min_size=1):
    if path.exists() and path.stat().st_size>=min_size: return
    with requests.get(url,stream=True,timeout=180) as r:
        r.raise_for_status()
        with open(path,"wb") as f:
            for ch in r.iter_content(4*1024*1024):
                if ch: f.write(ch)

download(C3D_URL,ZIP,1_000_000_000)
download(INFO_URL,INFO,10_000)

def first_between(vals,lo,hi):
    vals=np.asarray(vals,int)
    i=np.searchsorted(vals,lo+1)
    return int(vals[i]) if i<len(vals) and vals[i]<hi else None

def smooth(x):
    x=np.asarray(x,float)
    if len(x)<9: return x
    w=min(21, len(x)//2*2-1)
    if w<7: return x
    return savgol_filter(x,w,3,mode="interp")

def detect_events(c3d_path):
    c=ezc3d.c3d(str(c3d_path))
    labels=[str(x).strip() for x in c["parameters"]["POINT"]["LABELS"]["value"]]
    pts=np.asarray(c["data"]["points"][:3],float)
    rate=float(c["parameters"]["POINT"]["RATE"]["value"][0])
    idx={lab:i for i,lab in enumerate(labels)}
    req=["L.Heel","R.Heel","L.MT2","R.MT2"]
    if any(k not in idx for k in req):
        return None,"missing foot markers"
    if "V_Mid_PSIS" in idx:
        pelvis=pts[:,idx["V_Mid_PSIS"],:].T
    elif "L.PSIS" in idx and "R.PSIS" in idx:
        pelvis=((pts[:,idx["L.PSIS"],:]+pts[:,idx["R.PSIS"],:])/2).T
    else:
        return None,"missing pelvis marker"
    # Identify walking axis from pelvis displacement.
    valid=np.isfinite(pelvis).all(axis=1)
    if valid.sum()<20: return None,"invalid pelvis trajectory"
    pr=np.nanpercentile(pelvis[valid],[5,95],axis=0)
    axis=int(np.argmax(pr[1]-pr[0]))
    start=np.nanmedian(pelvis[valid][:max(3,valid.sum()//10),axis])
    end=np.nanmedian(pelvis[valid][-max(3,valid.sum()//10):,axis])
    direction=1.0 if end>=start else -1.0

    def rel(marker):
        return smooth(direction*(pts[axis,idx[marker],:]-pelvis[:,axis]))
    lheel,rheel=rel("L.Heel"),rel("R.Heel")
    ltoe,rtoe=rel("L.MT2"),rel("R.MT2")

    min_dist=max(1,int(0.40*rate))
    def extrema(sig,mode):
        finite=np.isfinite(sig)
        if finite.sum()<20: return np.array([],int)
        x=sig.copy()
        med=float(np.nanmedian(x))
        x[~finite]=med
        rng=float(np.nanpercentile(x,95)-np.nanpercentile(x,5))
        prom=max(10.0,0.10*rng)
        y=x if mode=="max" else -x
        peaks,_=find_peaks(y,distance=min_dist,prominence=prom)
        return peaks.astype(int)

    lhs=extrema(lheel,"max"); rhs=extrema(rheel,"max")
    lto=extrema(ltoe,"min"); rto=extrema(rtoe,"min")

    cycles=[]
    for a,b in zip(lhs[:-1],lhs[1:]):
        dur=(b-a)/rate
        if not (0.55<=dur<=2.5): continue
        er=first_between(rto,a,b)
        if er is None: continue
        rh=first_between(rhs,er,b)
        if rh is None: continue
        lt=first_between(lto,rh,b)
        if lt is None: continue
        if not (a<er<rh<lt<b): continue
        cycles.append({"a":a,"rto":er,"rhs":rh,"lto":lt,"b":b,
                       "cycle":dur,"left_stance":(lt-a)/rate,"left_swing":(b-lt)/rate})
    return {"rate":rate,"axis":axis,"direction":direction,"lhs":lhs,"rhs":rhs,"lto":lto,"rto":rto,"cycles":cycles},None

def parse_temporal(raw):
    text=raw.decode("utf-8",errors="replace")
    lines=[l for l in text.splitlines() if l.strip()]
    if len(lines)<6: return {}
    # Header names are second nonempty row, values are final row after ITEM.
    names=lines[1].strip().split("\t")
    vals=lines[-1].strip().split("\t")
    if vals and vals[0]=="1": vals=vals[1:]
    out={}
    for k,v in zip(names,vals):
        try: out[k]=float(v)
        except: pass
    return out

trial_rows=[]
sub_cycles={}
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    c3ds=[n for n in names if re.search(r"/SUB\d+_(?:on|off)_walk_\d+\.c3d$",n,re.I)]
    def key(n):
        m=re.search(r"SUB(\d+)_(on|off)_walk_(\d+)\.c3d$",n,re.I)
        return (int(m.group(1)),m.group(2).lower(),int(m.group(3)))
    c3ds=sorted(c3ds,key=key)
    tmp=Path("/tmp/walk.c3d")
    for ii,n in enumerate(c3ds,1):
        sid,cond,trial=key(n)
        with z.open(n) as src, open(tmp,"wb") as dst:
            while True:
                b=src.read(1024*1024)
                if not b: break
                dst.write(b)
        det,err=detect_events(tmp)
        tname=n[:-4]+"_temporal_distance.txt"
        ref=parse_temporal(z.read(tname)) if tname in names else {}
        cycles=[] if det is None else det["cycles"]
        sub_cycles.setdefault((sid,cond),[]).extend([(trial,c) for c in cycles])
        cyc=np.mean([c["cycle"] for c in cycles]) if cycles else np.nan
        st=np.mean([c["left_stance"] for c in cycles]) if cycles else np.nan
        sw=np.mean([c["left_swing"] for c in cycles]) if cycles else np.nan
        trial_rows.append({
            "subject":sid,"condition":cond,"trial":trial,
            "n_cycles":len(cycles),"error":err or "",
            "det_cycle_time":cyc,"ref_cycle_time":ref.get("Cycle_Time_Mean",np.nan),
            "cycle_abs_error":abs(cyc-ref.get("Cycle_Time_Mean",np.nan)) if np.isfinite(cyc) and "Cycle_Time_Mean" in ref else np.nan,
            "det_left_stance":st,"ref_left_stance":ref.get("Left_Stance_Time_Mean",np.nan),
            "stance_abs_error":abs(st-ref.get("Left_Stance_Time_Mean",np.nan)) if np.isfinite(st) and "Left_Stance_Time_Mean" in ref else np.nan,
            "det_left_swing":sw,"ref_left_swing":ref.get("Left_Swing_Time_Mean",np.nan),
            "swing_abs_error":abs(sw-ref.get("Left_Swing_Time_Mean",np.nan)) if np.isfinite(sw) and "Left_Swing_Time_Mean" in ref else np.nan,
            "axis":det["axis"] if det else np.nan,
            "direction":det["direction"] if det else np.nan,
        })
        if ii%100==0: print("processed",ii,"/",len(c3ds),flush=True)

tr=pd.DataFrame(trial_rows)
tr.to_csv(OUT/"onoff_event_qc_trials.csv",index=False)
subs=[]
for sid in sorted(tr.subject.unique()):
    row={"subject":int(sid)}
    for cond in ("off","on"):
        q=tr[(tr.subject==sid)&(tr.condition==cond)]
        row[f"{cond}_trials"]=len(q)
        row[f"{cond}_valid_trials"]=int((q.n_cycles>0).sum())
        row[f"{cond}_cycles"]=int(q.n_cycles.sum())
        row[f"{cond}_cycle_mae"]=float(q.cycle_abs_error.dropna().median()) if q.cycle_abs_error.notna().any() else np.nan
        row[f"{cond}_stance_mae"]=float(q.stance_abs_error.dropna().median()) if q.stance_abs_error.notna().any() else np.nan
        row[f"{cond}_swing_mae"]=float(q.swing_abs_error.dropna().median()) if q.swing_abs_error.notna().any() else np.nan
    subs.append(row)
sd=pd.DataFrame(subs)
sd.to_csv(OUT/"onoff_event_qc_subjects.csv",index=False)

# Metadata preview.
info=pd.read_excel(INFO)
info.to_csv(OUT/"PDGinfo.csv",index=False)
(OUT/"PDGINFO_COLUMNS.txt").write_text("\n".join(map(str,info.columns)),encoding="utf-8")

errs=tr.cycle_abs_error.dropna()
lines=["# ON/OFF gait-event reconstruction QC","",
       f"walking C3D trials processed: {len(tr)}",
       f"subjects: {tr.subject.nunique()}",
       f"trials with >=1 complete LHS-RTO-RHS-LTO-LHS cycle: {(tr.n_cycles>0).sum()} / {len(tr)}",
       f"total reconstructed cycles: {int(tr.n_cycles.sum())}",
       f"median absolute cycle-time error vs dataset temporal metric: {errs.median():.4f} s" if len(errs) else "cycle-time error unavailable",
       f"90th percentile absolute cycle-time error: {errs.quantile(.9):.4f} s" if len(errs) else "",
       "",
       "## Cycle counts per subject/condition",
       sd.to_markdown(index=False),
       "",
       "## Metadata columns",
       ", ".join(map(str,info.columns))]
(OUT/"EVENT_QC.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"EVENT_QC.md").read_text(encoding="utf-8"))
