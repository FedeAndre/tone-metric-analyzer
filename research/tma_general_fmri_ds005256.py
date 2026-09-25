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
from research.tma_general_fmri_ds004796 import SURR_BASE, sample_indices, quadratic_peak_time, build_frames, safe_analyze, surrogate_z

DS="ds005256"
OUT=Path("research/tma_general_fmri_ds005256_results")

def list_bold(run):
    pat=re.compile(rf"/sub-[^/]+/ses-02/func/.*_task-faces_.*_run-{run:02d}_bold\.nii\.gz$")
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
    fd,p=tempfile.mkstemp(suffix=".nii.gz"); os.close(fd); S3.download_file(BUCKET,key,p); return Path(p)

def read_events(key):
    ekey=key.replace("_bold.nii.gz","_events.tsv")
    b=S3.get_object(Bucket=BUCKET,Key=ekey)["Body"].read()
    return pd.read_csv(io.BytesIO(b),sep="\t")

def subject(key):
    m=re.search(r"/(sub-[^/]+)/",key); return m.group(1) if m else None

def preprocess(path,events,key):
    img=nib.load(str(path))
    if len(img.shape)!=4 or img.shape[3]<100: raise RuntimeError(f"unexpected shape {img.shape}")
    tr=float(img.header.get_zooms()[3]); data=img.get_fdata(dtype=np.float32); T=data.shape[-1]; dur=T*tr
    mean=np.nanmean(data[...,::max(1,T//30)],axis=-1); pos=mean[np.isfinite(mean)&(mean>0)]
    if len(pos)<1000: raise RuntimeError("insufficient voxels")
    mask=np.isfinite(mean)&(mean>.20*np.nanpercentile(pos,98))
    coords=np.argwhere(mask); mids=np.median(coords,axis=0); flat=data.reshape((-1,T)); grid=np.indices(mask.shape)
    sectors=[]
    for bx in (0,1):
      for by in (0,1):
       for bz in (0,1):
        m=mask.copy()
        m &= (grid[0]>mids[0]) if bx else (grid[0]<=mids[0])
        m &= (grid[1]>mids[1]) if by else (grid[1]<=mids[1])
        m &= (grid[2]>mids[2]) if bz else (grid[2]<=mids[2])
        idx=sample_indices(m,1500)
        if len(idx)>=80: sectors.append(np.nanmean(flat[idx,:],axis=0))
    if len(sectors)<6: raise RuntimeError("too few sectors")
    sec=np.asarray(sectors,float); sec=signal.detrend(sec,axis=-1,type="linear")
    sec=(sec-sec.mean(axis=1,keepdims=True))/(sec.std(axis=1,keepdims=True)+1e-9)
    coact=np.sqrt(np.mean(sec**2,axis=0))
    dist=max(1,int(round(2.0/tr))); peaks,_=signal.find_peaks(coact,distance=dist,prominence=max(.10*np.std(coact),1e-6))
    if len(peaks)<20: peaks,_=signal.find_peaks(coact,distance=dist)
    peak_times=np.array([quadratic_peak_time(coact,int(i),tr) for i in peaks],float)
    tt=events["trial_type"].astype(str).str.strip().str.lower()
    anchors=np.sort(pd.to_numeric(events.loc[tt=="face","onset"],errors="coerce").dropna().to_numpy(float))
    anchors=anchors[(anchors>=0)&(anchors<dur)]
    if len(anchors)<25: raise RuntimeError(f"too few face anchors {len(anchors)}")
    tol=max(.10,tr/2.0); frames=build_frames(anchors,peak_times)
    feat,evdf,_,_=safe_analyze(frames,tol); surr=surrogate_z(anchors,peak_times,feat,tol,dur,key,n_surr=20)
    return {**feat,**surr,"tr":tr,"n_volumes":T,"n_anchors":len(anchors),"n_peaks":len(peaks)}, {"anchors":anchors[:12].tolist(),"peaks":peak_times[:12].tolist(),"events":evdf.head(12).to_dict(orient="records")}

def run_one(run):
    OUT.mkdir(parents=True,exist_ok=True); rows=[]; fails={}; worked={}; keys=list_bold(run)
    for i,key in enumerate(keys,1):
        p=None; print(f"[{i}/{len(keys)}] {key}",flush=True)
        try:
            p=download_temp(key); ev=read_events(key); feat,ex=preprocess(p,ev,key)
            rows.append({"key":key,"subject":subject(key),"run":run,**feat})
            if len(worked)<2: worked[key]=ex
        except Exception as e: fails[key]=repr(e)
        finally:
            if p and p.exists(): p.unlink()
    pd.DataFrame(rows).to_csv(OUT/f"run-{run:02d}_scan_features.csv",index=False)
    (OUT/f"run-{run:02d}_failures.json").write_text(json.dumps(fails,indent=2))
    (OUT/f"run-{run:02d}_worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    print({"run":run,"success":len(rows),"fail":len(fails)})

def summarize(df):
    rows=[]
    for c in [f"surz_{x}" for x in SURR_BASE]:
        if c not in df: continue
        x=df[c].dropna().to_numpy(float)
        if len(x)<10: continue
        tt=stats.ttest_1samp(x,0)
        rows.append({"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(tt.statistic),"p":float(tt.pvalue),"positive_fraction":float((x>0).mean())})
    z=pd.DataFrame(rows)
    if len(z): z["q_BH"]=bh_adjust(z.p.to_numpy(float)); z=z.sort_values("p")
    return z

def aggregate():
    files=[OUT/f"run-{r:02d}_scan_features.csv" for r in (1,2,3)]
    scan=pd.concat([pd.read_csv(p) for p in files],ignore_index=True); scan.to_csv(OUT/"all_scan_features.csv",index=False)
    tabs={}
    for r in (1,2,3):
        tab=summarize(scan[scan.run==r]); tab.to_csv(OUT/f"run-{r:02d}_task_locking.csv",index=False); tabs[str(r)]=tab
    surz=[f"surz_{x}" for x in SURR_BASE]
    sub=scan.groupby("subject",as_index=False)[surz].mean(); sub.to_csv(OUT/"subject_features.csv",index=False)
    combined=summarize(sub); combined.to_csv(OUT/"combined_task_locking.csv",index=False)
    reps=[]
    for f in surz:
        row={"feature":f}; signs=[]; fdr=0; nominal=0
        for r in (1,2,3):
            q=tabs[str(r)]; q=q[q.feature==f]
            if len(q):
                m=float(q.iloc[0].mean_z); row[f"run{r}_mean_z"]=m; row[f"run{r}_q"]=float(q.iloc[0].q_BH)
                signs.append(np.sign(m)); fdr+=int(float(q.iloc[0].q_BH)<.05); nominal+=int(float(q.iloc[0].p)<.05)
        row["same_direction_all3"]=len(signs)==3 and len(set(signs))==1; row["n_fdr_runs"]=fdr; row["n_nominal_runs"]=nominal; reps.append(row)
    rep=pd.DataFrame(reps); rep.to_csv(OUT/"run_replication.csv",index=False)
    fails={p.name:len(json.loads(p.read_text())) for p in OUT.glob("*_failures.json")}
    summary={"dataset":DS,"description":"Healthy affective face-processing task-fMRI exact TMA validation",
             "math_audit":math_audit(),"n_scans":int(len(scan)),"n_subjects":int(scan.subject.nunique()),
             "run_task_locking":{k:v.to_dict(orient="records") for k,v in tabs.items()},
             "combined_task_locking":combined.to_dict(orient="records"),"run_replication":rep.to_dict(orient="records"),"failures":fails}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",type=int,choices=[1,2,3]); ap.add_argument("--aggregate",action="store_true"); a=ap.parse_args()
    if a.aggregate: aggregate()
    elif a.run: run_one(a.run)
    else: raise SystemExit("run required")
if __name__=="__main__": main()
