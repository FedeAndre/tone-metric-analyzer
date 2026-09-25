#!/usr/bin/env python3
from __future__ import annotations
import io,json,re,tarfile
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy import stats
from itertools import combinations
from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,bh_adjust,hedges_g

URL="https://physionet.org/files/gaitdb/1.0.0/gait-data.tar.gz"
OUT=Path("research/tma_gaitdb_exact_results");OUT.mkdir(parents=True,exist_ok=True)
TOL=1/600
MAX_FRAMES=64

def load_tar():
    r=requests.get(URL,timeout=120,headers={"User-Agent":"Mozilla/5.0 TMA-research"});r.raise_for_status()
    return tarfile.open(fileobj=io.BytesIO(r.content),mode="r:gz")

def group_age(name):
    base=Path(name).name
    if base.startswith("pd"):return "pd",np.nan
    m=re.match(r"o\d+-(\d+)-si\.txt",base)
    if m:return "old",float(m.group(1))
    m=re.match(r"y\d+-(\d+)-si\.txt",base)
    if m:return "young",float(m.group(1))
    return None,np.nan

def subject_features(a):
    t=np.asarray(a[:,0],float);stride=np.asarray(a[:,1],float)
    good=np.isfinite(t)&np.isfinite(stride)&(stride>0.4)&(stride<2.5)
    t=t[good];stride=stride[good]
    if len(t)<150:raise RuntimeError("too few strides")
    tactus=float(np.median(stride))
    dur=4*tactus
    total=t[-1]-t[0];nf=min(MAX_FRAMES,int(total//dur))
    if nf<40:raise RuntimeError(f"only {nf} frames")
    # centered block
    block=nf*dur;start=t[0]+max(0,(total-block)/2)
    frames=[]
    for i in range(nf):
        a0=start+i*dur;b=a0+dur
        ev=[{"label":"HEEL_STRIKE","offset_s":float(x-a0)} for x in t[(t>=a0)&(t<b)]]
        frames.append({"duration_s":dur,"events":ev})
    feat,ev,piv,tree=analyze_event_frames(frames,tol_s=TOL,label_features=False)
    d=np.diff(t)
    conv={"mean_stride":float(np.mean(stride)),"sd_stride":float(np.std(stride,ddof=1)),
          "cv_stride":float(np.std(stride,ddof=1)/np.mean(stride)),
          "rmssd_stride":float(np.sqrt(np.mean(np.diff(stride)**2))),
          "lag1_stride":float(np.corrcoef(stride[:-1],stride[1:])[0,1]) if len(stride)>2 else np.nan,
          "median_stride":tactus}
    return feat,conv

def exact_perm(a,b):
    x=np.r_[a,b];n=len(a);obs=abs(np.mean(b)-np.mean(a));vals=[]
    for ix in combinations(range(len(x)),n):
        m=np.zeros(len(x),bool);m[list(ix)]=True
        vals.append(abs(np.mean(x[~m])-np.mean(x[m])))
    vals=np.asarray(vals)
    return float((np.sum(vals>=obs-1e-15))/len(vals))

def contrast(df,g0,g1,features):
    rows=[]
    for c in features:
        a=df.loc[df.group==g0,c].dropna().values;b=df.loc[df.group==g1,c].dropna().values
        if len(a)<3 or len(b)<3:continue
        rows.append({"feature":c,"contrast":f"{g1}-vs-{g0}","n0":len(a),"n1":len(b),
                     f"{g0}_mean":float(np.mean(a)),f"{g1}_mean":float(np.mean(b)),
                     "diff":float(np.mean(b)-np.mean(a)),"hedges_g":hedges_g(a,b),
                     "exact_permutation_p":exact_perm(a,b)})
    q=pd.DataFrame(rows)
    if len(q):q["q_BH"]=bh_adjust(q.exact_permutation_p.values);q=q.sort_values("exact_permutation_p")
    return q

def main():
    tf=load_tar();rows=[];fail={}
    names=[m.name for m in tf.getmembers() if m.isfile() and m.name.endswith("-si.txt")]
    for n in names:
        g,age=group_age(n)
        if not g:continue
        try:
            a=np.loadtxt(tf.extractfile(n))
            feat,conv=subject_features(a)
            rows.append({"record":Path(n).name,"group":g,"age":age,**conv,**feat})
        except Exception as e:fail[n]=repr(e)
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"record","group","age","mean_stride","sd_stride","cv_stride","rmssd_stride","lag1_stride","median_stride"})
    pdold=contrast(df,"old","pd",tma);pdold.to_csv(OUT/"pd_vs_old.csv",index=False)
    aging=contrast(df,"young","old",tma);aging.to_csv(OUT/"old_vs_young.csv",index=False)
    conv=["mean_stride","sd_stride","cv_stride","rmssd_stride","lag1_stride"]
    cp=contrast(df,"old","pd",conv);cp.to_csv(OUT/"pd_vs_old_conventional.csv",index=False)
    # simple leave-one-out nearest-centroid three-group TMA classification
    X=df[tma];mu=X.mean();sd=X.std(ddof=1).replace(0,np.nan);Z=(X-mu)/sd
    pred=[];labs=df.group.to_numpy()
    for i in range(len(df)):
        tr=np.arange(len(df))!=i;cents={g:Z.loc[tr&(labs==g)].mean().to_numpy() for g in np.unique(labs)}
        xi=Z.iloc[i].to_numpy();pred.append(min(cents,key=lambda g:np.sqrt(np.nanmean((xi-cents[g])**2))))
    acc=float(np.mean(np.array(pred)==labs))
    summary={"dataset":"Gait in Aging and Disease","n":len(df),"groups":df.groupby("group").size().to_dict(),"failures":fail,
             "pd_vs_old_top":pdold.head(10).to_dict(orient="records"),"old_vs_young_top":aging.head(8).to_dict(orient="records"),
             "conventional_pd_vs_old":cp.head(6).to_dict(orient="records"),"three_group_tma_accuracy":acc,
             "chance":1/3,"max_projection_error_ms":float(df.projection_error_ms_max.max())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
