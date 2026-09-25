#!/usr/bin/env python3
from __future__ import annotations
import io,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,group_contrast
from tone_metric.theory import boundary_sequence,boundary_sequence_recurrence,pascal_binomial_mod,lucas_binomial_mod

URL="https://physionet.org/content/gaitdb/get-zip/1.0.0/"
OUT=Path("research/tma_gait_aging_exact_results");OUT.mkdir(parents=True,exist_ok=True)
N_FRAMES=48
STRIDES_PER_FRAME=4
TOL=0.005
SEED=20260925

def load_zip():
    r=requests.get(URL,timeout=180,headers={"User-Agent":"Mozilla/5.0 TMA-research"});r.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(r.content))

def parse_subject(name):
    base=Path(name).name
    if not base.endswith("-si.txt"):return None
    if base.startswith("pd"):
        m=re.match(r"(pd\d+)-si\.txt",base)
        return {"subject":m.group(1),"group":"pd","age":np.nan}
    if base.startswith("o"):
        m=re.match(r"(o\d+)-(\d+)-si\.txt",base)
        return {"subject":m.group(1),"group":"old","age":float(m.group(2))}
    if base.startswith("y"):
        m=re.match(r"(y\d+)-(\d+)-si\.txt",base)
        return {"subject":m.group(1),"group":"young","age":float(m.group(2))}
    return None

def build_frames(strides):
    x=np.asarray(strides,float)
    x=x[np.isfinite(x)&(x>0.3)&(x<3.0)]
    need=N_FRAMES*STRIDES_PER_FRAME
    if len(x)<need: raise RuntimeError(f"only {len(x)} valid strides; need {need}")
    # central contiguous block to avoid startup/end transients
    k=(len(x)-need)//2;x=x[k:k+need]
    frames=[]
    for i in range(N_FRAMES):
        q=x[i*4:(i+1)*4];dur=float(q.sum())
        offs=np.cumsum(q)[:-1]
        frames.append({"duration_s":dur,"events":[
            {"label":"HS1","offset_s":float(offs[0])},
            {"label":"HS2","offset_s":float(offs[1])},
            {"label":"HS3","offset_s":float(offs[2])},
        ]})
    return frames,x

def conventional(strides):
    x=np.asarray(strides,float)
    return {
      "stride_mean":float(np.mean(x)),
      "stride_sd":float(np.std(x,ddof=1)),
      "stride_cv":float(np.std(x,ddof=1)/np.mean(x)),
      "lag1":float(np.corrcoef(x[:-1],x[1:])[0,1]) if len(x)>2 else np.nan
    }

def auc(df,features,g0,g1):
    q=df[df.group.isin([g0,g1])].dropna(subset=features).reset_index(drop=True)
    y=(q.group==g1).astype(int).to_numpy();pred=np.zeros(len(q))
    for i in range(len(q)):
        tr=np.arange(len(q))!=i
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        model.fit(q.loc[tr,features],y[tr]);pred[i]=model.predict_proba(q.loc[[i],features])[0,1]
    return float(roc_auc_score(y,pred)),len(q)

def main():
    z=load_zip()
    audit={"boundary_match":boundary_sequence(2,12)==boundary_sequence_recurrence(2,12),"lucas_pascal_mismatches":0}
    for n in range(64):
        for k in range(n+1):
            audit["lucas_pascal_mismatches"]+=pascal_binomial_mod(n,k,2)!=lucas_binomial_mod(n,k,2)
    rows=[];fails={}
    for n in z.namelist():
        meta=parse_subject(n)
        if not meta:continue
        try:
            arr=np.loadtxt(io.BytesIO(z.read(n)))
            strides=arr[:,1]
            frames,used=build_frames(strides)
            feat,_,_,_=analyze_event_frames(frames,tol_s=TOL,label_features=True)
            rows.append({**meta,**conventional(used),**feat})
        except Exception as e:fails[Path(n).name]=repr(e)
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"subject","group","age","stride_mean","stride_sd","stride_cv","lag1"})
    pdo=group_contrast(df,"group","old","pd",tma)
    pdy=group_contrast(df,"group","young","pd",tma)
    oy=group_contrast(df,"group","young","old",tma)
    pdo.to_csv(OUT/"pd_vs_old.csv",index=False);pdy.to_csv(OUT/"pd_vs_young.csv",index=False);oy.to_csv(OUT/"old_vs_young.csv",index=False)
    base=["stride_mean","stride_sd","stride_cv","lag1"]
    compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in df]
    a0,n0=auc(df,base,"old","pd");a1,_=auc(df,base+compact,"old","pd")
    inc={"pd_vs_old":{"n":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0}}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summary={"dataset":"Gait in Aging and Disease","paper_math_audit":audit,"n_subjects":len(df),
             "groups":df.groupby("group").size().to_dict(),"failures":fails,
             "top_pd_vs_old":pdo.head(12).to_dict(orient="records"),
             "top_old_vs_young":oy.head(12).to_dict(orient="records"),
             "incremental":inc,"max_projection_error_ms":float(df.projection_error_ms_max.max())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
