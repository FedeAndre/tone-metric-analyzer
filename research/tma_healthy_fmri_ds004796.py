#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, re, tempfile, hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal, stats

from research.tma_clinical_exact_utils import analyze_event_frames, bh_adjust, project_offset
from research.tma_mri_exact_utils import S3, BUCKET, math_audit

DS="ds004796"
OUT=Path("research/tma_healthy_fmri_ds004796_results")
TMA=[
 "mean_H","mean_D","mean_lambda","multilevel","pivot_rate",
 "compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time","tree_root_fraction"
]
SURR=["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","tree_span_events","tree_span_time"]

def list_bold(task,direction):
    pat=re.compile(rf"^ds004796/(sub-[^/]+)/func/.*_task-{task}_dir-{direction}_bold\.nii\.gz$")
    out=[]; token=None
    while True:
        kw={"Bucket":BUCKET,"Prefix":f"{DS}/"}
        if token: kw["ContinuationToken"]=token
        r=S3.list_objects_v2(**kw)
        for x in r.get("Contents",[]):
            k=x["Key"]
            if pat.match(k): out.append({"key":k,"size":int(x["Size"])})
        if not r.get("IsTruncated"): break
        token=r["NextContinuationToken"]
    return sorted(out,key=lambda x:x["key"])

def download_temp(key):
    fd,p=tempfile.mkstemp(suffix=".nii.gz"); os.close(fd)
    S3.download_file(BUCKET,key,p)
    return Path(p)

def read_events(key):
    ekey=key.replace("_bold.nii.gz","_events.tsv")
    b=S3.get_object(Bucket=BUCKET,Key=ekey)["Body"].read()
    return pd.read_csv(io.BytesIO(b),sep="\t")

def sample_idx(mask,maxn):
    idx=np.flatnonzero(mask.ravel())
    if len(idx)<=maxn:return idx
    pos=np.linspace(0,len(idx)-1,maxn).round().astype(int)
    return idx[pos]

def quad_peak(y,i,tr):
    if i<=0 or i>=len(y)-1:return float(i*tr)
    a,b,c=float(y[i-1]),float(y[i]),float(y[i+1])
    den=a-2*b+c
    d=0.0 if abs(den)<1e-12 else float(np.clip(0.5*(a-c)/den,-0.5,0.5))
    return float((i+d)*tr)

def norm_event_code(x):
    return re.sub(r"\s+","",str(x)).upper()

def task_anchors(events,task,run_duration):
    if "event_type" not in events.columns: raise RuntimeError("event_type missing")
    ec=events["event_type"].map(norm_event_code)
    tt=events["trial_type"].astype(str).str.strip().str.lower() if "trial_type" in events.columns else pd.Series([""]*len(events))
    if task=="msit":
        # S4/S5 are the repeated MSIT trial stimuli. S10 is an inter-block marker.
        m=(tt=="stimulus") & ec.isin(["S4","S5"])
        lo,hi=1.5,15.0
    elif task=="sternberg":
        # Each Sternberg trial begins with the S3/S4 memory-set presentation.
        m=(tt=="stimulus") & ec.isin(["S3","S4"])
        lo,hi=8.0,30.0
    else: raise ValueError(task)
    x=pd.to_numeric(events.loc[m,"onset"],errors="coerce").dropna().to_numpy(float)
    x=np.sort(x[(x>=0)&(x<run_duration)])
    if len(x)<12: raise RuntimeError(f"only {len(x)} task anchors")
    return x,lo,hi

def build_frames(anchors,peaks,lo,hi):
    p=np.asarray(sorted(peaks),float); frames=[]
    for a,b in zip(anchors[:-1],anchors[1:]):
        dur=float(b-a)
        if not(lo<=dur<=hi):continue
        q=p[(p>=a)&(p<b)]
        frames.append({"duration_s":dur,"events":[{"label":"BOLD","offset_s":float(t-a)} for t in q]})
    return frames

def safe_analyze(frames,tol):
    clean=[];seen=set()
    for i,fr in enumerate(frames):
        evs=[]
        for ev in fr["events"]:
            q,_,_=project_offset(float(ev["offset_s"]),float(fr["duration_s"]),tol)
            # right boundary belongs to next normalized task interval
            if q<0 or q>=Fraction(2):continue
            onset=Fraction(2*i)+q
            if onset in seen:continue
            seen.add(onset);evs.append(ev)
        clean.append({"duration_s":fr["duration_s"],"events":evs})
    if sum(len(x["events"]) for x in clean)<8:raise RuntimeError("too few task-locked BOLD events")
    return analyze_event_frames(clean,tol_s=tol,label_features=False)

def surrogate_z(anchors,peak_times,obs,tol,run_duration,lo,hi,key,n=30):
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%(2**32)
    rng=np.random.default_rng(seed);vals={c:[] for c in SURR}
    p=np.asarray(peak_times,float)
    for _ in range(n):
        shift=float(rng.uniform(max(5,0.1*run_duration),max(5.01,0.9*run_duration)))
        q=np.mod(p+shift,run_duration)
        try:feat,*_=safe_analyze(build_frames(anchors,q,lo,hi),tol)
        except Exception:continue
        for c in SURR:
            if c in feat and np.isfinite(feat[c]):vals[c].append(float(feat[c]))
    out={}
    for c in SURR:
        z=np.asarray(vals[c],float)
        out[f"surz_{c}"]=float((obs[c]-z.mean())/z.std(ddof=1)) if len(z)>=10 and z.std(ddof=1)>1e-12 else np.nan
        out[f"surrmean_{c}"]=float(z.mean()) if len(z) else np.nan
    return out

def preprocess(path,events,task,key):
    img=nib.load(str(path))
    if len(img.shape)!=4 or img.shape[3]<100:raise RuntimeError(f"unexpected shape {img.shape}")
    tr=float(img.header.get_zooms()[3])
    if not(0.2<tr<5):raise RuntimeError(f"TR {tr}")
    data=img.get_fdata(dtype=np.float32)
    T=data.shape[-1];run_duration=T*tr
    mean=np.nanmean(data[...,::max(1,T//30)],axis=-1)
    pos=mean[np.isfinite(mean)&(mean>0)]
    if len(pos)<1000:raise RuntimeError("insufficient positive voxels")
    mask=np.isfinite(mean)&(mean>0.20*np.nanpercentile(pos,98))
    if mask.sum()<3000:raise RuntimeError(f"small mask {mask.sum()}")
    coords=np.argwhere(mask);mids=np.median(coords,axis=0);grid=np.indices(mask.shape)
    flat=data.reshape((-1,T)); sectors=[]
    for bx in (0,1):
      for by in (0,1):
       for bz in (0,1):
        m=mask.copy()
        m&=(grid[0]>mids[0]) if bx else (grid[0]<=mids[0])
        m&=(grid[1]>mids[1]) if by else (grid[1]<=mids[1])
        m&=(grid[2]>mids[2]) if bz else (grid[2]<=mids[2])
        ix=sample_idx(m,2200)
        if len(ix)>=100:sectors.append(np.nanmean(flat[ix,:],axis=0))
    if len(sectors)<6:raise RuntimeError("too few sectors")
    sec=np.asarray(sectors,float)
    sec=signal.detrend(sec,axis=-1,type="linear")
    sec=(sec-sec.mean(1,keepdims=True))/(sec.std(1,keepdims=True)+1e-9)
    coact=np.sqrt(np.mean(sec**2,axis=0)); global_sig=np.mean(sec,axis=0)
    dist=max(1,int(round(2.4/tr)));prom=max(.12*np.std(coact),1e-6)
    peaks,_=signal.find_peaks(coact,distance=dist,prominence=prom)
    if len(peaks)<25:peaks,_=signal.find_peaks(coact,distance=dist,prominence=max(.05*np.std(coact),1e-6))
    if len(peaks)<18:raise RuntimeError(f"only {len(peaks)} peaks")
    pt=np.array([quad_peak(coact,int(i),tr) for i in peaks],float)
    anchors,lo,hi=task_anchors(events,task,run_duration)
    frames=build_frames(anchors,pt,lo,hi)
    if len(frames)<10:raise RuntimeError(f"only {len(frames)} valid task intervals")
    tol=max(.20,tr/2)
    feat,ev,piv,tree=safe_analyze(frames,tol)
    sur=surrogate_z(anchors,pt,feat,tol,run_duration,lo,hi,key)
    f,powr=signal.welch(global_sig,fs=1/tr,nperseg=min(128,T));band=(f>=.01)&(f<=min(.15,f.max()))
    pb=powr[band];pn=pb/pb.sum() if pb.sum()>0 else np.ones_like(pb)/max(1,len(pb))
    sent=float(-np.sum(pn*np.log(pn+1e-15))/np.log(len(pn))) if len(pn)>1 else np.nan
    return {
      "tr":tr,"n_volumes":T,"n_anchors":len(anchors),"n_peaks":len(peaks),"n_frames":len(frames),
      "anchor_interval_mean":float(np.mean(np.diff(anchors))),"peak_rate_per_min":float(len(peaks)/(run_duration/60)),
      "coactivation_mean":float(coact.mean()),"coactivation_sd":float(coact.std(ddof=1)),
      "global_sd":float(global_sig.std(ddof=1)),"spectral_entropy":sent,
      **feat,**sur
    },{"anchors":anchors[:10].tolist(),"peaks":pt[:15].tolist(),"events":ev.head(12).to_dict(orient="records")}

def run_cell(task,direction):
    OUT.mkdir(parents=True,exist_ok=True);keys=list_bold(task,direction)
    rows=[];fails={};worked={}
    for i,it in enumerate(keys,1):
        k=it["key"];p=None;print(f"[{i}/{len(keys)}] {k}",flush=True)
        try:
            p=download_temp(k);e=read_events(k);feat,ex=preprocess(p,e,task,k)
            sub=re.search(r"/(sub-[^/]+)/",k).group(1)
            rows.append({"key":k,"subject":sub,"task":task,"direction":direction,**feat})
            if len(worked)<3:worked[k]=ex
        except Exception as exc:fails[k]=repr(exc);print("FAIL",repr(exc),flush=True)
        finally:
            if p and p.exists():p.unlink()
    tag=f"{task}_{direction}"
    pd.DataFrame(rows).to_csv(OUT/f"{tag}_features.csv",index=False)
    (OUT/f"{tag}_failures.json").write_text(json.dumps(fails,indent=2))
    (OUT/f"{tag}_worked.json").write_text(json.dumps(worked,indent=2,default=str))
    print(json.dumps({"task":task,"direction":direction,"success":len(rows),"failures":len(fails)},indent=2))

def onesample_table(df,group_cols):
    rows=[]
    zcols=[f"surz_{c}" for c in SURR]
    for keys,g in df.groupby(group_cols):
        if not isinstance(keys,tuple):keys=(keys,)
        meta=dict(zip(group_cols,keys))
        for c in zcols:
            x=g[c].dropna().to_numpy(float)
            if len(x)<10:continue
            tt=stats.ttest_1samp(x,0.0)
            rows.append({**meta,"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(tt.statistic),"p":float(tt.pvalue),
                         "positive_fraction":float(np.mean(x>0))})
    out=pd.DataFrame(rows)
    if len(out):
        out["q_BH"]=np.nan
        for _,ix in out.groupby(group_cols).groups.items():
            out.loc[ix,"q_BH"]=bh_adjust(out.loc[ix,"p"].to_numpy(float))
    return out

def aggregate():
    fs=sorted(OUT.glob("*_*_features.csv"))
    if len(fs)<4:raise RuntimeError(f"need 4 cell files, found {len(fs)}")
    scan=pd.concat([pd.read_csv(p) for p in fs],ignore_index=True)
    scan.to_csv(OUT/"all_scan_features.csv",index=False)

    # Replication within task across AP/PA: one row per subject/task/direction.
    cell=scan.groupby(["subject","task","direction"],as_index=False).mean(numeric_only=True)
    cell.to_csv(OUT/"subject_task_direction.csv",index=False)
    cell_tests=onesample_table(cell,["task","direction"]);cell_tests.to_csv(OUT/"task_direction_tasklocking.csv",index=False)

    # Primary task-level inference: average AP/PA first, so subject is the unit.
    task=cell.groupby(["subject","task"],as_index=False).mean(numeric_only=True)
    task.to_csv(OUT/"subject_task.csv",index=False)
    task_tests=onesample_table(task,["task"]);task_tests.to_csv(OUT/"task_tasklocking.csv",index=False)

    # Broad inference across two distinct tasks: average task estimates within participant.
    broad=task.groupby("subject",as_index=False).mean(numeric_only=True)
    broad_tests=onesample_table(broad.assign(scope="across_tasks"),["scope"])
    broad_tests.to_csv(OUT/"broad_tasklocking.csv",index=False)

    # AP/PA subject-level reproducibility and cross-task subject-level reproducibility.
    reps=[]
    for taskname in ["msit","sternberg"]:
      A=cell[(cell.task==taskname)&(cell.direction=="AP")].set_index("subject")
      B=cell[(cell.task==taskname)&(cell.direction=="PA")].set_index("subject")
      ids=sorted(set(A.index)&set(B.index))
      for c in [f"surz_{x}" for x in SURR]:
        if len(ids)>=10:
          r,p=stats.pearsonr(A.loc[ids,c],B.loc[ids,c])
          reps.append({"replication":"AP_vs_PA","task":taskname,"feature":c,"n":len(ids),"r":float(r),"p":float(p)})
    A=task[task.task=="msit"].set_index("subject");B=task[task.task=="sternberg"].set_index("subject");ids=sorted(set(A.index)&set(B.index))
    for c in [f"surz_{x}" for x in SURR]:
      if len(ids)>=10:
        r,p=stats.pearsonr(A.loc[ids,c],B.loc[ids,c])
        reps.append({"replication":"MSIT_vs_Sternberg","task":"both","feature":c,"n":len(ids),"r":float(r),"p":float(p)})
    rep=pd.DataFrame(reps)
    if len(rep):rep["q_BH"]=bh_adjust(rep.p.to_numpy(float))
    rep.to_csv(OUT/"reproducibility.csv",index=False)

    summary={
      "dataset":DS,
      "description":"Healthy cross-task fMRI exact-TMA task-locking validation: MSIT executive control and Sternberg working memory",
      "math_audit":math_audit(),
      "n_successful_scans":int(len(scan)),"n_subjects":int(scan.subject.nunique()),
      "success_by_cell":scan.groupby(["task","direction"]).size().astype(int).to_dict(),
      "primary_task_tests":task_tests.sort_values("p").head(20).to_dict(orient="records"),
      "broad_tests":broad_tests.sort_values("p").to_dict(orient="records"),
      "reproducibility":rep.sort_values("p").head(20).to_dict(orient="records")
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=["msit","sternberg"])
    ap.add_argument("--direction",choices=["AP","PA"])
    ap.add_argument("--aggregate",action="store_true")
    a=ap.parse_args()
    if a.aggregate:aggregate()
    else:
        if not a.task or not a.direction:raise SystemExit("--task and --direction required")
        run_cell(a.task,a.direction)
if __name__=="__main__":main()
