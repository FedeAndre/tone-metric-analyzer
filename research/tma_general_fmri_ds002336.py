#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, re, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal, stats

from research.tma_mri_exact_utils import S3, BUCKET, math_audit
from research.tma_clinical_exact_utils import bh_adjust
from research.tma_general_fmri_ds004796 import (
    SURR_BASE, sample_indices, quadratic_peak_time, build_frames, safe_analyze, surrogate_z
)

DS="ds002336"
OUT=Path("research/tma_general_fmri_ds002336_results")
TASKS=["motorloc","fmriNF","MIpre","MIpost"]

def list_bold(task):
    pat=re.compile(rf"/sub-[^/]+/func/.*_task-{re.escape(task)}_bold\.nii\.gz$")
    out=[]; token=None
    while True:
        kw={"Bucket":BUCKET,"Prefix":f"{DS}/"}
        if token: kw["ContinuationToken"]=token
        r=S3.list_objects_v2(**kw)
        for x in r.get("Contents",[]):
            if pat.search(x["Key"]): out.append(x["Key"])
        if not r.get("IsTruncated"): break
        token=r["NextContinuationToken"]
    return sorted(out)

def download_temp(key):
    fd,p=tempfile.mkstemp(suffix=".nii.gz"); os.close(fd)
    S3.download_file(BUCKET,key,p); return Path(p)

def read_root_events(task):
    key=f"{DS}/task-{task}_events.tsv"
    b=S3.get_object(Bucket=BUCKET,Key=key)["Body"].read()
    # Only onset times define the block scaffold. Parse column 1 directly so
    # a malformed trailing field in the public MIpost TSV cannot discard a block.
    onsets=[]
    for line in b.decode("utf-8-sig").splitlines()[1:]:
        parts=line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        try:
            onsets.append(float(parts[0]))
        except ValueError:
            continue
    return pd.DataFrame({"onset":onsets})

def subject(key):
    m=re.search(r"/(sub-[^/]+)/",key); return m.group(1) if m else None

def preprocess(path,events,key):
    img=nib.load(str(path))
    if len(img.shape)!=4 or img.shape[3]<80: raise RuntimeError(f"unexpected shape {img.shape}")
    tr=float(img.header.get_zooms()[3]); data=img.get_fdata(dtype=np.float32); T=data.shape[-1]; dur=T*tr
    mean=np.nanmean(data[...,::max(1,T//30)],axis=-1); pos=mean[np.isfinite(mean)&(mean>0)]
    thr=.20*np.nanpercentile(pos,98); mask=np.isfinite(mean)&(mean>thr)
    if mask.sum()<2000: raise RuntimeError("small mask")
    coords=np.argwhere(mask); mids=np.median(coords,axis=0); flat=data.reshape((-1,T)); grid=np.indices(mask.shape)
    sectors=[]
    for bx in (0,1):
      for by in (0,1):
       for bz in (0,1):
        m=mask.copy()
        m &= (grid[0]>mids[0]) if bx else (grid[0]<=mids[0])
        m &= (grid[1]>mids[1]) if by else (grid[1]<=mids[1])
        m &= (grid[2]>mids[2]) if bz else (grid[2]<=mids[2])
        idx=sample_indices(m,1800)
        if len(idx)>=80: sectors.append(np.nanmean(flat[idx,:],axis=0))
    sec=np.asarray(sectors,float); sec=signal.detrend(sec,axis=-1,type="linear")
    sec=(sec-sec.mean(axis=1,keepdims=True))/(sec.std(axis=1,keepdims=True)+1e-9)
    coact=np.sqrt(np.mean(sec**2,axis=0)); global_sig=np.mean(sec,axis=0)
    dist=max(1,int(round(2.0/tr))); peaks,_=signal.find_peaks(coact,distance=dist,prominence=max(.10*np.std(coact),1e-6))
    if len(peaks)<15: peaks,_=signal.find_peaks(coact,distance=dist)
    peak_times=np.array([quadratic_peak_time(coact,int(i),tr) for i in peaks],float)
    anchors=np.sort(pd.to_numeric(events["onset"],errors="coerce").dropna().to_numpy(float))
    anchors=anchors[(anchors>=0)&(anchors<dur)]
    if len(anchors)<8: raise RuntimeError("too few anchors")
    frames=build_frames(anchors,peak_times); tol=max(.125,tr/2)
    feat,evdf,_,_=safe_analyze(frames,tol)
    surr=surrogate_z(anchors,peak_times,feat,tol,dur,key,n_surr=50)
    idx=sample_indices(mask,2200); vox=flat[idx,:].astype(float); vm=np.mean(vox,axis=1,keepdims=True)
    rel=(vox-vm)/(np.abs(vm)+1e-9)*100
    dvars=float(np.median(np.sqrt(np.mean(np.diff(rel,axis=1)**2,axis=0))))
    tsnr=float(np.median(np.abs(vm[:,0])/(np.std(vox,axis=1,ddof=1)+1e-9)))
    return {**feat,**surr,"tr":tr,"n_volumes":T,"n_anchors":len(anchors),"n_peaks":len(peaks),
            "dvars_pct":dvars,"tsnr":tsnr}, {"anchors":anchors[:12].tolist(),"peaks":peak_times[:12].tolist(),"events":evdf.head(12).to_dict(orient="records")}

def run_task(task):
    OUT.mkdir(parents=True,exist_ok=True); ev=read_root_events(task); rows=[]; fails={}; worked={}
    keys=list_bold(task)
    for i,key in enumerate(keys,1):
        p=None; print(f"[{i}/{len(keys)}] {key}",flush=True)
        try:
            p=download_temp(key); feat,ex=preprocess(p,ev,key); rows.append({"key":key,"subject":subject(key),"task":task,**feat})
            if len(worked)<2: worked[key]=ex
        except Exception as e:
            fails[key]=repr(e)
            print("FAIL",key,repr(e),flush=True)
        finally:
            if p and p.exists(): p.unlink()
    pd.DataFrame(rows).to_csv(OUT/f"{task}_scan_features.csv",index=False)
    (OUT/f"{task}_failures.json").write_text(json.dumps(fails,indent=2))
    (OUT/f"{task}_worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    print({"task":task,"success":len(rows),"fail":len(fails)})

def summarize(df):
    rows=[]
    for c in [f"surz_{x}" for x in SURR_BASE]:
        if c not in df: continue
        x=df[c].dropna().to_numpy(float)
        if len(x)<5: continue
        tt=stats.ttest_1samp(x,0)
        rows.append({"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(tt.statistic),"p":float(tt.pvalue),"positive_fraction":float((x>0).mean())})
    z=pd.DataFrame(rows)
    if len(z): z["q_BH"]=bh_adjust(z.p.to_numpy(float)); z=z.sort_values("p")
    return z

def aggregate():
    pairs=[]
    for t in TASKS:
        p=OUT/f"{t}_scan_features.csv"
        if not p.exists() or p.stat().st_size<=1:
            continue
        try:
            q=pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if len(q):
            pairs.append((t,q))
    if not pairs:
        raise RuntimeError("no successful task scans")
    scan=pd.concat([q for _,q in pairs],ignore_index=True); scan.to_csv(OUT/"all_scan_features.csv",index=False)
    tabs={}; reps=[]
    for task in TASKS:
        z=scan[scan.task==task]
        if not len(z):
            continue
        tab=summarize(z); tab.to_csv(OUT/f"{task}_task_locking.csv",index=False); tabs[task]=tab.to_dict(orient="records")
    # Across tasks, require the same direction; count independent task replication.
    feats=[f"surz_{x}" for x in SURR_BASE]
    for f in feats:
        row={"feature":f}; signs=[]; fdr=0; nominal=0
        for task in TASKS:
            if task not in tabs:
                continue
            tab=pd.DataFrame(tabs[task])
            q=tab[tab.feature==f]
            if len(q):
                m=float(q.iloc[0].mean_z); row[f"{task}_mean_z"]=m; row[f"{task}_q"]=float(q.iloc[0].q_BH)
                signs.append(np.sign(m)); fdr+=int(float(q.iloc[0].q_BH)<.05); nominal+=int(float(q.iloc[0].p)<.05)
        row["same_direction_all"]=len(signs)>=2 and len(set(signs))==1; row["n_fdr_tasks"]=fdr; row["n_nominal_tasks"]=nominal; reps.append(row)
    cross=pd.DataFrame(reps); cross.to_csv(OUT/"cross_task_replication.csv",index=False)
    fails={p.name:len(json.loads(p.read_text())) for p in OUT.glob("*_failures.json")}
    summary={"dataset":DS,"description":"Healthy motor/motor-imagery exact TMA task-locking validation",
             "math_audit":math_audit(),"n_scans":int(len(scan)),"n_subjects":int(scan.subject.nunique()),
             "tasks":TASKS,"task_locking":tabs,"cross_task_replication":cross.to_dict(orient="records"),"failures":fails}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=TASKS); ap.add_argument("--aggregate",action="store_true"); a=ap.parse_args()
    if a.aggregate: aggregate()
    elif a.task: run_task(a.task)
    else: raise SystemExit("task required")
if __name__=="__main__": main()
