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

DS="ds005256"
OUT=Path("research/tma_healthy_fmri_ds005256_results")
TASKS=["faces","narratives","shortvideo","fractional"]
SURR=["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","tree_span_events","tree_span_time"]

def list_run1(task):
    pat=re.compile(rf"^ds005256/(sub-[^/]+)/ses-[^/]+/func/.*_task-{task}_.*_run-01_bold\.nii\.gz$")
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
    fd,p=tempfile.mkstemp(suffix=".nii.gz");os.close(fd)
    S3.download_file(BUCKET,key,p); return Path(p)

def read_events(key):
    ekey=key.replace("_bold.nii.gz","_events.tsv")
    b=S3.get_object(Bucket=BUCKET,Key=ekey)["Body"].read()
    return pd.read_csv(io.BytesIO(b),sep="\t")

def sample_idx(mask,maxn):
    idx=np.flatnonzero(mask.ravel())
    if len(idx)<=maxn:return idx
    pos=np.linspace(0,len(idx)-1,maxn).round().astype(int)
    return idx[pos]

def qpeak(y,i,tr):
    if i<=0 or i>=len(y)-1:return float(i*tr)
    a,b,c=float(y[i-1]),float(y[i]),float(y[i+1]); den=a-2*b+c
    d=0 if abs(den)<1e-12 else float(np.clip(.5*(a-c)/den,-.5,.5))
    return float((i+d)*tr)

def anchors(events,task,run_duration):
    if "onset" not in events: raise RuntimeError("missing onset")
    if task=="faces":
        m=events["trial_type"].astype(str).str.lower().eq("face"); lo,hi=2.0,18.0
    elif task=="narratives":
        m=events["trial_type"].astype(str).str.lower().eq("narrative_presentation"); lo,hi=8.0,45.0
    elif task=="shortvideo":
        m=events["trial_type"].astype(str).str.lower().eq("video"); lo,hi=8.0,25.0
    elif task=="fractional":
        m=events["event_type"].astype(str).str.lower().eq("stimulus"); lo,hi=20.0,40.0
    else: raise ValueError(task)
    x=pd.to_numeric(events.loc[m,"onset"],errors="coerce").dropna().to_numpy(float)
    x=np.sort(x[(x>=0)&(x<run_duration)])
    if len(x)<10: raise RuntimeError(f"only {len(x)} anchors")
    return x,lo,hi

def build_frames(a,p,lo,hi):
    p=np.asarray(sorted(p),float); out=[]
    for x,y in zip(a[:-1],a[1:]):
        dur=float(y-x)
        if not(lo<=dur<=hi):continue
        q=p[(p>=x)&(p<y)]
        out.append({"duration_s":dur,"events":[{"label":"BOLD","offset_s":float(t-x)} for t in q]})
    return out

def safe_analyze(frames,tol):
    clean=[];seen=set()
    for i,fr in enumerate(frames):
        ev=[]
        for z in fr["events"]:
            q,_,_=project_offset(float(z["offset_s"]),float(fr["duration_s"]),tol)
            if q<0 or q>=Fraction(2): continue
            t=Fraction(2*i)+q
            if t in seen: continue
            seen.add(t); ev.append(z)
        clean.append({"duration_s":fr["duration_s"],"events":ev})
    if sum(len(x["events"]) for x in clean)<8:raise RuntimeError("too few projected events")
    return analyze_event_frames(clean,tol_s=tol,label_features=False)

def surrogate_z(a,pt,obs,tol,run_duration,lo,hi,key,n=30):
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%(2**32)
    rng=np.random.default_rng(seed); vals={c:[] for c in SURR}
    for _ in range(n):
        shift=float(rng.uniform(max(5,.1*run_duration),max(5.01,.9*run_duration)))
        q=np.mod(pt+shift,run_duration)
        try:f,*_=safe_analyze(build_frames(a,q,lo,hi),tol)
        except Exception:continue
        for c in SURR:
            if c in f and np.isfinite(f[c]):vals[c].append(float(f[c]))
    out={}
    for c in SURR:
        x=np.asarray(vals[c],float)
        out[f"surz_{c}"]=float((obs[c]-x.mean())/x.std(ddof=1)) if len(x)>=10 and x.std(ddof=1)>1e-12 else np.nan
        out[f"surrmean_{c}"]=float(x.mean()) if len(x) else np.nan
    return out

def preprocess(path,events,task,key):
    img=nib.load(str(path))
    if len(img.shape)!=4 or img.shape[3]<100:raise RuntimeError(f"shape {img.shape}")
    tr=float(img.header.get_zooms()[3])
    if not(.2<tr<5):raise RuntimeError(f"TR {tr}")
    data=img.get_fdata(dtype=np.float32);T=data.shape[-1];dur=T*tr
    mean=np.nanmean(data[...,::max(1,T//30)],axis=-1)
    pos=mean[np.isfinite(mean)&(mean>0)]
    if len(pos)<1000:raise RuntimeError("few positive voxels")
    mask=np.isfinite(mean)&(mean>.20*np.nanpercentile(pos,98))
    if mask.sum()<3000:raise RuntimeError(f"mask {mask.sum()}")
    coords=np.argwhere(mask); mids=np.median(coords,axis=0); grid=np.indices(mask.shape);flat=data.reshape((-1,T))
    sec=[]
    for bx in (0,1):
      for by in (0,1):
       for bz in (0,1):
        m=mask.copy()
        m&=(grid[0]>mids[0]) if bx else (grid[0]<=mids[0])
        m&=(grid[1]>mids[1]) if by else (grid[1]<=mids[1])
        m&=(grid[2]>mids[2]) if bz else (grid[2]<=mids[2])
        ix=sample_idx(m,1800)
        if len(ix)>=100:sec.append(np.nanmean(flat[ix,:],axis=0))
    if len(sec)<6:raise RuntimeError("few sectors")
    sec=np.asarray(sec,float);sec=signal.detrend(sec,axis=-1,type="linear")
    sec=(sec-sec.mean(1,keepdims=True))/(sec.std(1,keepdims=True)+1e-9)
    co=np.sqrt(np.mean(sec**2,axis=0))
    distance=max(1,int(round(2.5/tr))); prom=max(.12*np.std(co),1e-6)
    peaks,_=signal.find_peaks(co,distance=distance,prominence=prom)
    if len(peaks)<25:peaks,_=signal.find_peaks(co,distance=distance,prominence=max(.05*np.std(co),1e-6))
    if len(peaks)<18:raise RuntimeError(f"peaks {len(peaks)}")
    pt=np.asarray([qpeak(co,int(i),tr) for i in peaks],float)
    a,lo,hi=anchors(events,task,dur); fr=build_frames(a,pt,lo,hi)
    if len(fr)<8:raise RuntimeError(f"frames {len(fr)}")
    tol=max(.20,tr/2)
    feat,ev,piv,tree=safe_analyze(fr,tol); sur=surrogate_z(a,pt,feat,tol,dur,lo,hi,key)
    return {"tr":tr,"n_volumes":T,"n_anchors":len(a),"n_peaks":len(peaks),"n_frames":len(fr),**feat,**sur}

def run_task(task):
    OUT.mkdir(parents=True,exist_ok=True);rows=[];fails={}
    keys=list_run1(task)
    for i,it in enumerate(keys,1):
        k=it["key"];p=None;print(f"[{i}/{len(keys)}] {k}",flush=True)
        try:
            p=download_temp(k);e=read_events(k);f=preprocess(p,e,task,k)
            sub=re.search(r"/(sub-[^/]+)/",k).group(1)
            rows.append({"subject":sub,"task":task,"key":k,**f})
        except Exception as ex:fails[k]=repr(ex);print("FAIL",repr(ex),flush=True)
        finally:
            if p and p.exists():p.unlink()
    pd.DataFrame(rows).to_csv(OUT/f"{task}_run01_features.csv",index=False)
    (OUT/f"{task}_run01_failures.json").write_text(json.dumps(fails,indent=2))
    print(json.dumps({"task":task,"success":len(rows),"failures":len(fails)},indent=2))

def test_table(df,group="task"):
    rows=[]
    for name,g in df.groupby(group):
        ps=[];tmp=[]
        for c in SURR:
            x=g[f"surz_{c}"].dropna().to_numpy(float)
            if len(x)<10:continue
            tt=stats.ttest_1samp(x,0)
            tmp.append({"group":name,"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(tt.statistic),"p":float(tt.pvalue),"positive_fraction":float(np.mean(x>0))})
            ps.append(float(tt.pvalue))
        qs=bh_adjust(np.asarray(ps,float))
        for r,q in zip(tmp,qs):r["q_BH"]=float(q);rows.append(r)
    return pd.DataFrame(rows)

def aggregate():
    fs=sorted(OUT.glob("*_run01_features.csv"))
    if len(fs)<4:raise RuntimeError(f"need 4 task files, found {len(fs)}")
    d=pd.concat([pd.read_csv(p) for p in fs],ignore_index=True)
    d.to_csv(OUT/"all_tasks_run01.csv",index=False)
    tt=test_table(d);tt.to_csv(OUT/"task_tasklocking.csv",index=False)
    # Strict cross-task generalization uses subjects present in all 4 tasks.
    counts=d.groupby("subject").task.nunique()
    ids=counts[counts==4].index
    z=d[d.subject.isin(ids)].groupby("subject",as_index=False).mean(numeric_only=True)
    rows=[];ps=[]
    for c in SURR:
        x=z[f"surz_{c}"].dropna().to_numpy(float)
        t=stats.ttest_1samp(x,0)
        rows.append({"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(t.statistic),"p":float(t.pvalue),"positive_fraction":float(np.mean(x>0))})
        ps.append(float(t.pvalue))
    qs=bh_adjust(np.asarray(ps,float))
    for r,q in zip(rows,qs):r["q_BH"]=float(q)
    broad=pd.DataFrame(rows);broad.to_csv(OUT/"four_task_generalization.csv",index=False)
    # Task-by-feature heterogeneity (within-subject repeated task comparison where complete).
    complete=d[d.subject.isin(ids)].copy()
    heter=[]
    for c in SURR:
        piv=complete.pivot(index="subject",columns="task",values=f"surz_{c}").dropna()
        if len(piv)>=10:
            f,p=stats.friedmanchisquare(*[piv[t].to_numpy(float) for t in TASKS])
            heter.append({"feature":c,"n":len(piv),"friedman_chi2":float(f),"p":float(p)})
    h=pd.DataFrame(heter)
    if len(h):h["q_BH"]=bh_adjust(h.p.to_numpy(float))
    h.to_csv(OUT/"task_heterogeneity.csv",index=False)
    summary={
      "dataset":DS,"analysis":"healthy four-task exact-TMA task-locking validation",
      "math_audit":math_audit(),"n_scans":int(len(d)),"n_subjects":int(d.subject.nunique()),"n_complete_four_task_subjects":int(len(ids)),
      "task_success":{str(k):int(v) for k,v in d.groupby("task").size().items()},
      "task_results":tt.sort_values(["group","p"]).to_dict(orient="records"),
      "four_task_generalization":broad.sort_values("p").to_dict(orient="records"),
      "task_heterogeneity":h.sort_values("p").to_dict(orient="records") if len(h) else []
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=TASKS);p.add_argument("--aggregate",action="store_true");a=p.parse_args()
    if a.aggregate:aggregate()
    else:
        if not a.task:raise SystemExit("--task required")
        run_task(a.task)
if __name__=="__main__":main()
