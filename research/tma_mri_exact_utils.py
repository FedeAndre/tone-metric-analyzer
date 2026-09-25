#!/usr/bin/env python3
from __future__ import annotations
import math, tempfile, os, re, json
from pathlib import Path
import numpy as np, pandas as pd
import nibabel as nib
from scipy import signal, stats
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from research.tma_clinical_exact_utils import analyze_event_frames, numeric_tma_columns, group_contrast, paired_contrast, bh_adjust
from tone_metric.theory import boundary_sequence,boundary_sequence_recurrence,pascal_binomial_mod,lucas_binomial_mod

BUCKET="openneuro.org"
S3=boto3.client("s3",config=Config(signature_version=UNSIGNED),region_name="us-east-1")
SEED=20260925
RNG=np.random.default_rng(SEED)

def math_audit():
    out={"boundary_match":boundary_sequence(2,12)==boundary_sequence_recurrence(2,12),"lucas_pascal_mismatches":0}
    for n in range(64):
        for k in range(n+1):
            out["lucas_pascal_mismatches"] += pascal_binomial_mod(n,k,2)!=lucas_binomial_mod(n,k,2)
    return out

def list_bold(ds):
    out=[];token=None
    while True:
        kw={"Bucket":BUCKET,"Prefix":ds+"/"}
        if token:kw["ContinuationToken"]=token
        r=S3.list_objects_v2(**kw)
        for x in r.get("Contents",[]):
            k=x["Key"]
            if k.endswith("_bold.nii.gz") and "/func/" in k:
                out.append({"key":k,"size":x["Size"]})
        if not r.get("IsTruncated"):break
        token=r["NextContinuationToken"]
    return out

def read_tsv(ds,path):
    import io
    b=S3.get_object(Bucket=BUCKET,Key=f"{ds}/{path}")["Body"].read()
    return pd.read_csv(io.BytesIO(b),sep="\t")

def get_small_text(ds,path):
    return S3.get_object(Bucket=BUCKET,Key=f"{ds}/{path}")["Body"].read().decode(errors="replace")

def download_temp(key):
    fd,p=tempfile.mkstemp(suffix=".nii.gz");os.close(fd)
    S3.download_file(BUCKET,key,p)
    return Path(p)

def _sample_indices(mask,maxn=3000):
    idx=np.flatnonzero(mask.ravel())
    if len(idx)<=maxn:return idx
    pos=np.linspace(0,len(idx)-1,maxn).round().astype(int)
    return idx[pos]

def _safe_bandpass(x,tr):
    x=np.asarray(x,float)
    x=signal.detrend(x,axis=-1,type="linear")
    fs=1.0/tr;ny=fs/2
    hi=min(0.10,ny*0.80);lo=0.01
    if hi<=lo*1.25:return x
    sos=signal.butter(3,[lo,hi],btype="bandpass",fs=fs,output="sos")
    try:return signal.sosfiltfilt(sos,x,axis=-1)
    except:return x

def preprocess_bold(path):
    img=nib.load(str(path))
    shape=img.shape
    if len(shape)!=4 or shape[3]<60:raise RuntimeError(f"unexpected BOLD shape {shape}")
    tr=float(img.header.get_zooms()[3])
    if not (0.2<tr<5.0):raise RuntimeError(f"invalid TR {tr}")
    data=img.get_fdata(dtype=np.float32)
    data=data[...,5:]
    T=data.shape[-1]
    # Mean-image data-driven brain mask. Avoid dependence on an external atlas.
    mean=np.nanmean(data[...,::max(1,T//30)],axis=-1)
    pos=mean[np.isfinite(mean)&(mean>0)]
    if len(pos)<1000:raise RuntimeError("insufficient positive voxels")
    thr=0.20*np.nanpercentile(pos,98)
    mask=np.isfinite(mean)&(mean>thr)
    if mask.sum()<3000:raise RuntimeError(f"small brain mask {mask.sum()}")
    coords=np.argwhere(mask)
    mids=np.median(coords,axis=0)
    flat=data.reshape((-1,T))
    # Eight coarse sectors; sample equal max voxel count per sector.
    sector=[]
    for bx in (0,1):
      for by in (0,1):
       for bz in (0,1):
        m=mask.copy()
        grid=np.indices(mask.shape)
        m &= (grid[0]>mids[0]) if bx else (grid[0]<=mids[0])
        m &= (grid[1]>mids[1]) if by else (grid[1]<=mids[1])
        m &= (grid[2]>mids[2]) if bz else (grid[2]<=mids[2])
        idx=_sample_indices(m,3000)
        if len(idx)<100:continue
        sector.append(np.nanmean(flat[idx,:],axis=0))
    if len(sector)<6:raise RuntimeError(f"only {len(sector)} brain sectors")
    sec=np.asarray(sector,float)
    # Relative change, then detrend/bandpass/zscore each sector.
    sec=(sec-np.mean(sec,axis=1,keepdims=True))/(np.std(sec,axis=1,keepdims=True)+1e-9)
    sec=_safe_bandpass(sec,tr)
    sec=(sec-np.mean(sec,axis=1,keepdims=True))/(np.std(sec,axis=1,keepdims=True)+1e-9)
    coact=np.sqrt(np.mean(sec**2,axis=0))
    global_sig=np.mean(sec,axis=0)

    # Deterministic QC/conventional features from a sampled voxel set.
    idx=_sample_indices(mask,4000)
    vox=flat[idx,:].astype(float)
    vm=np.mean(vox,axis=1,keepdims=True)
    rel=(vox-vm)/(np.abs(vm)+1e-9)*100.0
    dvars=float(np.median(np.sqrt(np.mean(np.diff(rel,axis=1)**2,axis=0))))
    tsnr=float(np.median(np.abs(vm[:,0])/(np.std(vox,axis=1,ddof=1)+1e-9)))

    # Coactivation peaks. Timing, not amplitude, is passed into TMA.
    dist=max(1,int(round(4.0/tr)))
    prom=max(0.10*np.std(coact),1e-6)
    peaks,_=signal.find_peaks(coact,distance=dist,prominence=prom)
    if len(peaks)<25:peaks,_=signal.find_peaks(coact,distance=dist,prominence=max(0.04*np.std(coact),1e-6))
    if len(peaks)<17:peaks,_=signal.find_peaks(coact,distance=dist)
    if len(peaks)<17:raise RuntimeError(f"only {len(peaks)} coactivation peaks")

    nframe=(len(peaks)-1)//4
    nframe=min(nframe,32)
    # centered contiguous sequence when more than 32 macroframes are available
    start_frame=max(0,((len(peaks)-1)//4-nframe)//2)
    p0=4*start_frame
    frames=[]
    for j in range(nframe):
        q=peaks[p0+4*j:p0+4*j+5]
        if len(q)<5:break
        dur=(q[4]-q[0])*tr
        if dur<=0:continue
        frames.append({"duration_s":float(dur),"events":[
            {"label":"B0","offset_s":0.0},
            {"label":"B1","offset_s":float((q[1]-q[0])*tr)},
            {"label":"B2","offset_s":float((q[2]-q[0])*tr)},
            {"label":"B3","offset_s":float((q[3]-q[0])*tr)},
        ]})
    if len(frames)<4:raise RuntimeError(f"only {len(frames)} macroframes")
    tol=max(0.25,tr/2)
    feat,ev,piv,tree=analyze_event_frames(frames,tol_s=tol,label_features=True)

    ipi=np.diff(peaks)*tr
    f,powr=signal.welch(global_sig,fs=1/tr,nperseg=min(128,len(global_sig)))
    band=(f>=.01)&(f<=min(.10,f.max()))
    pb=powr[band]; pn=pb/pb.sum() if pb.sum()>0 else np.ones_like(pb)/max(1,len(pb))
    spectral_entropy=float(-np.sum(pn*np.log(pn+1e-15))/np.log(len(pn))) if len(pn)>1 else np.nan
    conv={
      "tr":tr,"n_volumes":T,"n_peaks":len(peaks),"n_macroframes":len(frames),
      "peak_interval_mean":float(np.mean(ipi)),"peak_interval_cv":float(np.std(ipi,ddof=1)/np.mean(ipi)),
      "coactivation_mean":float(np.mean(coact)),"coactivation_sd":float(np.std(coact,ddof=1)),
      "global_sd":float(np.std(global_sig,ddof=1)),"global_lag1":float(np.corrcoef(global_sig[:-1],global_sig[1:])[0,1]),
      "spectral_entropy":spectral_entropy,"dvars_pct":dvars,"tsnr":tsnr
    }
    return feat,conv,{"first_frame":frames[0],"first_events":ev.head(12).to_dict(orient="records")}

def process_dataset(ds,keys):
    rows=[];fails={};worked={}
    for i,item in enumerate(keys,1):
        key=item["key"] if isinstance(item,dict) else item
        print(f"[{i}/{len(keys)}] {key}",flush=True)
        p=None
        try:
            p=download_temp(key);feat,conv,ex=preprocess_bold(p)
            row={"key":key,**conv,**feat};rows.append(row)
            if len(worked)<3:worked[key]=ex
        except Exception as e:
            fails[key]=repr(e);print("FAIL",repr(e),flush=True)
        finally:
            if p and p.exists():p.unlink()
    return pd.DataFrame(rows),fails,worked

def hedges_g(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);n0=len(a);n1=len(b)
    if n0<2 or n1<2:return np.nan
    sp=((n0-1)*np.var(a,ddof=1)+(n1-1)*np.var(b,ddof=1))/(n0+n1-2)
    if sp<=0:return np.nan
    d=(np.mean(b)-np.mean(a))/math.sqrt(sp);J=1-3/(4*(n0+n1)-9)
    return float(J*d)

def adjust_group(df,feature,group_col,g0,g1,covars):
    z=df[df[group_col].isin([g0,g1])][[group_col,feature]+covars].replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<12:return None
    y=z[feature].to_numpy(float); y=(y-y.mean())/(y.std(ddof=1)+1e-12)
    g=(z[group_col]==g1).astype(float).to_numpy()
    X=[np.ones(len(z)),g]
    for c in covars:
        v=z[c].to_numpy(float)
        if np.std(v)>0:X.append((v-np.mean(v))/(np.std(v,ddof=1)+1e-12))
    X=np.column_stack(X);b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b;dof=len(y)-X.shape[1]
    cov=np.sum(res**2)/dof*np.linalg.pinv(X.T@X);se=np.sqrt(max(cov[1,1],0));t=b[1]/se if se>0 else np.nan
    p=2*stats.t.sf(abs(t),dof) if np.isfinite(t) else np.nan
    return {"feature":feature,"n":len(z),"beta_std":float(b[1]),"p":float(p),"covars":covars}

def partial_rank(df,x,y,covars):
    z=df[[x,y]+covars].replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<10:return None
    rx=stats.rankdata(z[x]);ry=stats.rankdata(z[y])
    C=[np.ones(len(z))]
    for c in covars:
        C.append(stats.rankdata(z[c]))
    C=np.column_stack(C)
    ex=rx-C@np.linalg.lstsq(C,rx,rcond=None)[0]
    ey=ry-C@np.linalg.lstsq(C,ry,rcond=None)[0]
    r,p=stats.pearsonr(ex,ey)
    return {"feature":y,"outcome":x,"n":len(z),"partial_rank_r":float(r),"p":float(p),"covars":covars}

CONV=["peak_interval_mean","peak_interval_cv","coactivation_mean","coactivation_sd","global_sd","global_lag1","spectral_entropy","dvars_pct","tsnr"]
