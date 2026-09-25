#!/usr/bin/env python3
from __future__ import annotations
import io,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy import signal,stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,paired_contrast

URL="https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
OUT=Path("research/tma_daphnet_exact_results");OUT.mkdir(parents=True,exist_ok=True)
FS=64.0
TOL=0.008
MAX_CYCLES=64
MIN_CYCLES=24

def read_zip():
    r=requests.get(URL,timeout=180,headers={"User-Agent":"Mozilla/5.0 TMA-research"});r.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(r.content))

def contiguous_segments(a,val,minlen=40):
    m=(a==val).astype(np.int8);d=np.diff(np.r_[0,m,0])
    starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1)
    return [(s,e) for s,e in zip(starts,ends) if e-s>=minlen]

def principal_signal(X):
    X=np.asarray(X,float)
    X=signal.detrend(X,axis=0)
    X=(X-X.mean(0))/(X.std(0)+1e-9)
    _,_,vh=np.linalg.svd(X,full_matrices=False)
    y=X@vh[0]
    sos=signal.butter(4,[0.5,8.0],btype="bandpass",fs=FS,output="sos")
    return signal.sosfiltfilt(sos,y)

def segment_cycles(y,s,e):
    x=y[s:e]
    up=np.flatnonzero((x[:-1]<=0)&(x[1:]>0))+1
    down=np.flatnonzero((x[:-1]>=0)&(x[1:]<0))+1
    out=[]
    for a,b in zip(up[:-1],up[1:]):
        dur=(b-a)/FS
        if not (0.10<=dur<=1.8):continue
        di=np.searchsorted(down,a+1)
        if di>=len(down) or down[di]>=b:continue
        d=int(down[di]);p=a+int(np.argmax(x[a:d]));n=d+int(np.argmin(x[d:b]))
        if not (a<p<d<n<b):continue
        amp=float(np.ptp(x[a:b]))
        events=[{"label":"UP0","offset_s":0.0},{"label":"POSPEAK","offset_s":(p-a)/FS},
                {"label":"DOWN0","offset_s":(d-a)/FS},{"label":"NEGPEAK","offset_s":(n-a)/FS}]
        out.append({"duration_s":dur,"events":events,"amp":amp})
    return out

def condition_features(frames):
    feat,ev,piv,tree=analyze_event_frames(frames,tol_s=TOL,label_features=True)
    dur=np.array([f["duration_s"] for f in frames]);amp=np.array([f["amp"] for f in frames])
    conv={"cycle_period_mean":float(dur.mean()),"cycle_period_cv":float(dur.std(ddof=1)/dur.mean()),
          "cycle_amp_mean":float(amp.mean()),"cycle_amp_cv":float(amp.std(ddof=1)/amp.mean()) if amp.mean()!=0 else np.nan}
    return feat,conv

def auc_loso(df,features):
    q=df.dropna(subset=features+["state"]).copy();q["y"]=(q.state=="freeze").astype(int)
    pred=np.full(len(q),np.nan)
    for sub in q.subject.unique():
        te=q.subject==sub;tr=~te
        if q.loc[tr,"y"].nunique()<2:continue
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        m.fit(q.loc[tr,features],q.loc[tr,"y"]);pred[te]=m.predict_proba(q.loc[te,features])[:,1]
    ok=np.isfinite(pred)
    return float(roc_auc_score(q.loc[ok,"y"],pred[ok])),int(ok.sum())

def main():
    z=read_zip();pool={};fail={}
    files=[n for n in z.namelist() if n.endswith(".txt") and "/dataset/" in n]
    for n in files:
        sub=re.search(r"/(S\d+)R",n).group(1)
        try:
            a=np.loadtxt(io.BytesIO(z.read(n)))
            ann=a[:,-1].astype(int);ank=a[:,1:4]
            exp=ann>0
            if exp.sum()<200:continue
            y=principal_signal(ank)
            for val,state in [(1,"nofreeze"),(2,"freeze")]:
                key=(sub,state);pool.setdefault(key,[])
                for s,e in contiguous_segments(ann,val):
                    pool[key].extend(segment_cycles(y,s,e))
        except Exception as e:fail[n]=repr(e)
    rows=[];excluded={}
    subjects=sorted(set(k[0] for k in pool))
    for sub in subjects:
        a=pool.get((sub,"nofreeze"),[]);b=pool.get((sub,"freeze"),[])
        n=min(MAX_CYCLES,len(a),len(b))
        if n<MIN_CYCLES:
            excluded[sub]={"nofreeze_cycles":len(a),"freeze_cycles":len(b)};continue
        # central matched number of cycles from each state
        def center(x,n):
            k=(len(x)-n)//2;return x[k:k+n]
        for state,fr in [("nofreeze",center(a,n)),("freeze",center(b,n))]:
            feat,conv=condition_features(fr)
            rows.append({"subject":sub,"state":state,"matched_cycles":n,**conv,**feat})
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_state_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"subject","state","matched_cycles","cycle_period_mean","cycle_period_cv","cycle_amp_mean","cycle_amp_cv"})
    pc=paired_contrast(df,"subject","state","nofreeze","freeze",tma);pc.to_csv(OUT/"freeze_vs_nofreeze.csv",index=False)
    conv=["cycle_period_mean","cycle_period_cv","cycle_amp_mean","cycle_amp_cv"]
    pconv=paired_contrast(df,"subject","state","nofreeze","freeze",conv);pconv.to_csv(OUT/"conventional_freeze_vs_nofreeze.csv",index=False)
    compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in df]
    auc0,n0=auc_loso(df,conv);auc1,n1=auc_loso(df,conv+compact)
    inc={"auc_base":auc0,"auc_plus_tma":auc1,"delta_auc":auc1-auc0,"n_predictions":n0,"base":conv,"tma":compact}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summary={"dataset":"Daphnet Freezing of Gait","n_subjects_raw":len(subjects),"n_paired_subjects":int(df.subject.nunique()) if len(df) else 0,
             "excluded":excluded,"n_file_failures":len(fail),"top_tma":pc.head(12).to_dict(orient="records"),
             "conventional":pconv.head(8).to_dict(orient="records"),"incremental":inc,
             "max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
