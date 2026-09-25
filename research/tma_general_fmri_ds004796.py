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
OUT=Path("research/tma_general_fmri_ds004796_results")
TMA_CORE=["mean_H","sd_H","mean_D","sd_D","mean_lambda","sd_lambda","multilevel",
          "pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events",
          "tree_span_time","tree_root_fraction"]
SURR_BASE=["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","tree_span_events","tree_span_time"]

def list_bold(task,direction):
    pat=re.compile(rf"/sub-[^/]+/func/.*_task-{re.escape(task)}_dir-{re.escape(direction)}_bold\.nii\.gz$")
    out=[]; token=None
    while True:
        kw={"Bucket":BUCKET,"Prefix":f"{DS}/"}
        if token: kw["ContinuationToken"]=token
        r=S3.list_objects_v2(**kw)
        for x in r.get("Contents",[]):
            k=x["Key"]
            if pat.search(k):
                out.append({"key":k,"size":int(x["Size"])})
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

def anchors_from_events(df,task):
    onset=pd.to_numeric(df.get("onset"),errors="coerce")
    tt=df.get("trial_type",pd.Series([""]*len(df))).astype(str).str.strip().str.lower()
    et=df.get("event_type",pd.Series([""]*len(df))).astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    dur=pd.to_numeric(df.get("duration"),errors="coerce")
    if task=="msit":
        # Each presented interference/control stimulus is one externally defined trial anchor.
        m=(tt=="stimulus") & dur.between(0.5,2.0,inclusive="both") & onset.notna()
    elif task=="sternberg":
        # S3/S4 mark the beginning of each complete Sternberg memory trial.
        m=(tt=="stimulus") & et.isin(["S 3","S 4"]) & onset.notna()
    else:
        raise ValueError(task)
    a=np.sort(onset[m].to_numpy(float))
    return a

def sample_indices(mask,maxn=3000):
    idx=np.flatnonzero(mask.ravel())
    if len(idx)<=maxn:return idx
    pos=np.linspace(0,len(idx)-1,maxn).round().astype(int)
    return idx[pos]

def quadratic_peak_time(y,i,tr):
    if i<=0 or i>=len(y)-1:return float(i*tr)
    y0,y1,y2=map(float,(y[i-1],y[i],y[i+1]))
    den=y0-2*y1+y2
    d=0.0 if abs(den)<1e-12 else float(np.clip(0.5*(y0-y2)/den,-0.5,0.5))
    return float((i+d)*tr)

def build_frames(anchors,peak_times):
    frames=[]; p=np.asarray(sorted(peak_times),float)
    for a,b in zip(anchors[:-1],anchors[1:]):
        dur=float(b-a)
        if not (1.0<=dur<=30.0): continue
        use=p[(p>=a)&(p<b)]
        frames.append({"duration_s":dur,"events":[{"label":"BOLD","offset_s":float(t-a)} for t in use]})
    return frames

def safe_analyze(frames,tol):
    clean=[]; seen=set()
    for i,fr in enumerate(frames):
        evs=[]
        for ev in fr["events"]:
            q,_,_=project_offset(float(ev["offset_s"]),float(fr["duration_s"]),tol)
            if q < 0 or q >= Fraction(2):
                continue
            onset=Fraction(2*i)+q
            if onset in seen: continue
            seen.add(onset); evs.append(ev)
        clean.append({"duration_s":fr["duration_s"],"events":evs})
    if sum(len(fr["events"]) for fr in clean)<8:
        raise RuntimeError("too few task-locked BOLD events")
    return analyze_event_frames(clean,tol_s=tol,label_features=False)

def surrogate_z(anchors,peak_times,observed,tol,run_duration,key,n_surr=20):
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%(2**32)
    rng=np.random.default_rng(seed); vals={c:[] for c in SURR_BASE}
    p=np.asarray(peak_times,float)
    for _ in range(n_surr):
        shift=float(rng.uniform(5.0,max(5.01,run_duration-5.0)))
        q=np.mod(p+shift,run_duration)
        try:
            feat,*_=safe_analyze(build_frames(anchors,q),tol)
        except Exception:
            continue
        for c in SURR_BASE:
            if c in feat and np.isfinite(feat[c]): vals[c].append(float(feat[c]))
    out={}
    for c in SURR_BASE:
        x=np.asarray(vals[c],float)
        out[f"surz_{c}"]=float((observed[c]-x.mean())/x.std(ddof=1)) if len(x)>=5 and x.std(ddof=1)>1e-12 else np.nan
    return out

def preprocess(path,events,key,task):
    img=nib.load(str(path))
    if len(img.shape)!=4 or img.shape[3]<80: raise RuntimeError(f"unexpected BOLD shape {img.shape}")
    tr=float(img.header.get_zooms()[3])
    data=img.get_fdata(dtype=np.float32); T=data.shape[-1]; run_duration=T*tr
    mean=np.nanmean(data[...,::max(1,T//30)],axis=-1)
    pos=mean[np.isfinite(mean)&(mean>0)]
    if len(pos)<1000: raise RuntimeError("insufficient positive voxels")
    thr=.20*np.nanpercentile(pos,98); mask=np.isfinite(mean)&(mean>thr)
    if mask.sum()<3000: raise RuntimeError(f"small brain mask {int(mask.sum())}")
    coords=np.argwhere(mask); mids=np.median(coords,axis=0)
    flat=data.reshape((-1,T)); grid=np.indices(mask.shape); sectors=[]
    for bx in (0,1):
      for by in (0,1):
       for bz in (0,1):
        m=mask.copy()
        m &= (grid[0]>mids[0]) if bx else (grid[0]<=mids[0])
        m &= (grid[1]>mids[1]) if by else (grid[1]<=mids[1])
        m &= (grid[2]>mids[2]) if bz else (grid[2]<=mids[2])
        idx=sample_indices(m,1800)
        if len(idx)>=100: sectors.append(np.nanmean(flat[idx,:],axis=0))
    if len(sectors)<6: raise RuntimeError("too few sectors")
    sec=np.asarray(sectors,float); sec=signal.detrend(sec,axis=-1,type="linear")
    sec=(sec-sec.mean(axis=1,keepdims=True))/(sec.std(axis=1,keepdims=True)+1e-9)
    coact=np.sqrt(np.mean(sec**2,axis=0)); global_sig=np.mean(sec,axis=0)
    dist=max(1,int(round(2.0/tr))); prom=max(.12*np.std(coact),1e-6)
    peaks,_=signal.find_peaks(coact,distance=dist,prominence=prom)
    if len(peaks)<20: peaks,_=signal.find_peaks(coact,distance=dist,prominence=max(.05*np.std(coact),1e-6))
    if len(peaks)<15: peaks,_=signal.find_peaks(coact,distance=dist)
    if len(peaks)<12: raise RuntimeError(f"only {len(peaks)} coactivation peaks")
    peak_times=np.array([quadratic_peak_time(coact,int(i),tr) for i in peaks],float)
    anchors=anchors_from_events(events,task)
    anchors=anchors[(anchors>=0)&(anchors<run_duration)]
    if len(anchors)<15: raise RuntimeError(f"only {len(anchors)} task anchors")
    frames=build_frames(anchors,peak_times)
    tol=max(.125,tr/2.0)
    feat,evdf,piv,tree=safe_analyze(frames,tol)
    surr=surrogate_z(anchors,peak_times,feat,tol,run_duration,key)

    idx=sample_indices(mask,2500); vox=flat[idx,:].astype(float)
    vm=np.mean(vox,axis=1,keepdims=True); rel=(vox-vm)/(np.abs(vm)+1e-9)*100
    dvars=float(np.median(np.sqrt(np.mean(np.diff(rel,axis=1)**2,axis=0))))
    tsnr=float(np.median(np.abs(vm[:,0])/(np.std(vox,axis=1,ddof=1)+1e-9)))
    f,powr=signal.welch(global_sig,fs=1/tr,nperseg=min(128,len(global_sig)))
    band=(f>=.01)&(f<=min(.15,f.max())); pb=powr[band]
    pn=pb/pb.sum() if pb.sum()>0 else np.ones_like(pb)/max(1,len(pb))
    sent=float(-np.sum(pn*np.log(pn+1e-15))/np.log(len(pn))) if len(pn)>1 else np.nan
    intervals=np.diff(anchors); intervals=intervals[(intervals>=1)&(intervals<=30)]
    conv={"tr":tr,"n_volumes":T,"n_anchors":len(anchors),"n_peaks":len(peaks),
          "anchor_interval_mean":float(intervals.mean()),"anchor_interval_cv":float(intervals.std(ddof=1)/intervals.mean()) if len(intervals)>1 else np.nan,
          "peak_rate_per_min":float(len(peaks)/(run_duration/60.0)),
          "coactivation_mean":float(coact.mean()),"coactivation_sd":float(coact.std(ddof=1)),
          "global_sd":float(global_sig.std(ddof=1)),
          "global_lag1":float(np.corrcoef(global_sig[:-1],global_sig[1:])[0,1]),
          "spectral_entropy":sent,"dvars_pct":dvars,"tsnr":tsnr}
    example={"first_anchors_s":anchors[:10].tolist(),"first_peaks_s":peak_times[:12].tolist(),
             "first_events":evdf.head(12).to_dict(orient="records")}
    return {**conv,**feat,**surr},example

def subject_from_key(key):
    m=re.search(r"/(sub-[^/]+)/",key); return m.group(1) if m else None

def run_combo(task,direction):
    OUT.mkdir(parents=True,exist_ok=True)
    items=list_bold(task,direction); rows=[]; fails={}; worked={}
    for i,item in enumerate(items,1):
        key=item["key"]; p=None; print(f"[{i}/{len(items)}] {key}",flush=True)
        try:
            p=download_temp(key); ev=read_events(key); feat,ex=preprocess(p,ev,key,task)
            rows.append({"key":key,"subject":subject_from_key(key),"task":task,"direction":direction,**feat})
            if len(worked)<3: worked[key]=ex
        except Exception as e:
            fails[key]=repr(e); print("FAIL",repr(e),flush=True)
        finally:
            if p and p.exists(): p.unlink()
    tag=f"{task}_{direction}"
    pd.DataFrame(rows).to_csv(OUT/f"{tag}_scan_features.csv",index=False)
    (OUT/f"{tag}_failures.json").write_text(json.dumps(fails,indent=2))
    (OUT/f"{tag}_worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    print(json.dumps({"task":task,"direction":direction,"success":len(rows),"failures":len(fails)},indent=2))

def one_sample_summary(df,features):
    rows=[]
    for c in features:
        x=df[c].dropna().to_numpy(float)
        if len(x)<10: continue
        t=stats.ttest_1samp(x,0.0)
        rows.append({"feature":c,"n":len(x),"mean_z":float(x.mean()),"sd":float(x.std(ddof=1)),
                     "t":float(t.statistic),"p":float(t.pvalue),
                     "positive_fraction":float((x>0).mean())})
    out=pd.DataFrame(rows)
    if len(out):
        out["q_BH"]=bh_adjust(out.p.to_numpy(float)); out=out.sort_values("p")
    return out

def aggregate():
    files=sorted(OUT.glob("*_scan_features.csv"))
    files=[p for p in files if p.name.split("_")[0] in ("msit","sternberg")]
    if len(files)<4: raise RuntimeError(f"expected four combo files, got {len(files)}")
    scan=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    scan.to_csv(OUT/"all_scan_features.csv",index=False)
    surz=[f"surz_{c}" for c in SURR_BASE if f"surz_{c}" in scan.columns]
    combo_tables={}; combo_summary={}
    for task in ["msit","sternberg"]:
      for direction in ["AP","PA"]:
        z=scan[(scan.task==task)&(scan.direction==direction)]
        tab=one_sample_summary(z,surz); tag=f"{task}_{direction}"
        tab.to_csv(OUT/f"{tag}_task_locking.csv",index=False)
        combo_tables[tag]=tab; combo_summary[tag]=tab.to_dict(orient="records")

    # Direction-averaged subject effects: independent participant is the unit.
    sub=scan.groupby(["subject","task"],as_index=False)[surz].mean()
    sub.to_csv(OUT/"subject_task_features.csv",index=False)
    task_tables={}; task_summary={}
    for task in ["msit","sternberg"]:
        z=sub[sub.task==task]; tab=one_sample_summary(z,surz)
        tab.to_csv(OUT/f"{task}_combined_task_locking.csv",index=False)
        task_tables[task]=tab; task_summary[task]=tab.to_dict(orient="records")

    # Strict AP/PA replication per task.
    rep={}
    for task in ["msit","sternberg"]:
        a=combo_tables[f"{task}_AP"][["feature","mean_z","p","q_BH"]].rename(columns={"mean_z":"mean_AP","p":"p_AP","q_BH":"q_AP"})
        b=combo_tables[f"{task}_PA"][["feature","mean_z","p","q_BH"]].rename(columns={"mean_z":"mean_PA","p":"p_PA","q_BH":"q_PA"})
        q=a.merge(b,on="feature")
        q["same_direction"]=np.sign(q.mean_AP)==np.sign(q.mean_PA)
        q["nominal_both"]=q.same_direction&(q.p_AP<.05)&(q.p_PA<.05)
        q["fdr_both"]=q.same_direction&(q.q_AP<.05)&(q.q_PA<.05)
        q.to_csv(OUT/f"{task}_AP_PA_replication.csv",index=False)
        rep[task]=q.to_dict(orient="records")

    # Cross-task convergence on participant-averaged effects.
    a=task_tables["msit"][["feature","mean_z","p","q_BH"]].rename(columns={"mean_z":"mean_msit","p":"p_msit","q_BH":"q_msit"})
    b=task_tables["sternberg"][["feature","mean_z","p","q_BH"]].rename(columns={"mean_z":"mean_sternberg","p":"p_sternberg","q_BH":"q_sternberg"})
    cross=a.merge(b,on="feature"); cross["same_direction"]=np.sign(cross.mean_msit)==np.sign(cross.mean_sternberg)
    cross["nominal_both"]=cross.same_direction&(cross.p_msit<.05)&(cross.p_sternberg<.05)
    cross["fdr_both"]=cross.same_direction&(cross.q_msit<.05)&(cross.q_sternberg<.05)
    cross.to_csv(OUT/"cross_task_replication.csv",index=False)

    fails={}
    for p in OUT.glob("*_failures.json"):
        fails[p.name]=len(json.loads(p.read_text()))
    summary={"dataset":DS,"description":"Healthy non-PD task-fMRI exact TMA task-locking validation",
             "tasks":["MSIT executive interference","Sternberg working memory"],
             "analysis_unit":"participant after averaging AP/PA for primary task effects",
             "math_audit":math_audit(),"n_successful_scans":int(len(scan)),
             "n_subjects":int(scan.subject.nunique()),"failures":fails,
             "surrogate":"20 deterministic circular shifts of BOLD coactivation peak times relative to unchanged external task anchors",
             "combo_task_locking":combo_summary,"combined_task_locking":task_summary,
             "AP_PA_replication":rep,"cross_task_replication":cross.to_dict(orient="records")}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=["msit","sternberg"]); ap.add_argument("--direction",choices=["AP","PA"]); ap.add_argument("--aggregate",action="store_true")
    a=ap.parse_args()
    if a.aggregate: aggregate()
    else:
        if not a.task or not a.direction: raise SystemExit("--task and --direction required")
        run_combo(a.task,a.direction)
if __name__=="__main__": main()
