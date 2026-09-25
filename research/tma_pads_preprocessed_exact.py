#!/usr/bin/env python3
from __future__ import annotations
import io,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd,requests
from scipy import signal,stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,group_contrast

BASE="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
OUT=Path("research/tma_pads_preprocessed_exact_results");OUT.mkdir(parents=True,exist_ok=True)
CACHE=Path("/tmp/pads_pre");CACHE.mkdir(exist_ok=True)
TASK_ORDER=["Relaxed1","Relaxed2","RelaxedTask1","RelaxedTask2","StretchHold","HoldWeight","DrinkGlas","CrossArms","TouchNose","Entrainment1","Entrainment2"]
TASKS=["HoldWeight","DrinkGlas","CrossArms","TouchNose"];WRISTS=["LeftWrist","RightWrist"];SENSORS=["Acceleration","Rotation"];AXES=["X","Y","Z"]
FS=100.0;N=976;TOL=0.005

CHANNELS=[f"{t}_{s}_{w}_{a}" for t in TASK_ORDER for w in WRISTS for s in SENSORS for a in AXES]
assert len(CHANNELS)==132
CID={c:i for i,c in enumerate(CHANNELS)}

def fetch(url,timeout=60):
    for k in range(5):
        try:
            r=requests.get(url,timeout=timeout,headers={"User-Agent":"Mozilla/5.0 TMA-research"})
            if r.status_code==200:return r.content
        except Exception:pass
    return None

def movement_features(x,wrist):
    frames=[];rms=[];ent=[];rates=[]
    t=np.arange(N)/FS
    for task in TASKS:
        idx=[CID[f"{task}_Rotation_{wrist}_{a}"] for a in AXES]
        g=x[idx,:].T
        mag=np.linalg.norm(g,axis=1)
        mag=signal.detrend(mag)
        mag=signal.savgol_filter(mag,11,3,mode="interp")
        half=(N/FS)/2
        for h in range(2):
            lo=h*half;hi=(h+1)*half;ix=np.flatnonzero((t>=lo)&(t<hi));yy=mag[ix]
            peaks,_=signal.find_peaks(yy,distance=25,prominence=max(np.std(yy)*.10,1e-8))
            if len(peaks)<3:peaks,_=signal.find_peaks(yy,distance=20)
            if len(peaks)<3:raise RuntimeError(f"{task} half{h} <3 peaks")
            chosen=np.sort(ix[peaks[np.argsort(yy[peaks])[-3:]]])
            frames.append({"duration_s":half,"events":[{"label":"MOVE","offset_s":float(t[j]-lo)} for j in chosen]})
        rms.append(float(np.sqrt(np.mean(mag**2))))
        f,p=signal.welch(mag,fs=FS,nperseg=512);m=(f>=.5)&(f<=12);q=p[m]
        if q.sum()>0:q=q/q.sum();ent.append(float(-np.sum(q*np.log(q+1e-15))/np.log(len(q))))
        pk,_=signal.find_peaks(mag,distance=20,prominence=max(np.std(mag)*.1,1e-8));rates.append(len(pk)/(N/FS))
    feat,_,_,_=analyze_event_frames(frames,tol_s=TOL,label_features=False)
    return feat,{"rotation_rms":float(np.mean(rms)),"spectral_entropy":float(np.mean(ent)),"movement_peak_rate":float(np.mean(rates))}

def auc(df,features,g0,g1):
    q=df[df.group.isin([g0,g1])].dropna(subset=features).reset_index(drop=True);y=(q.group==g1).astype(int).to_numpy();p=np.zeros(len(q))
    for i in range(len(q)):
        tr=np.arange(len(q))!=i;m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        m.fit(q.loc[tr,features],y[tr]);p[i]=m.predict_proba(q.loc[[i],features])[0,1]
    return float(roc_auc_score(y,p)),len(q)

def adjust(df,features,g0):
    q=df[df.group.isin([g0,"pd"])].copy();q["pd"]=(q.group=="pd").astype(float);rows=[]
    for c in features[:12]:
        z=q[["pd","age","rotation_rms","movement_peak_rate",c]].dropna()
        if len(z)<25:continue
        y=(z[c]-z[c].mean())/z[c].std(ddof=1);X=np.column_stack([np.ones(len(z)),z.pd,z.age,z.rotation_rms,z.movement_peak_rate])
        b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b;dof=len(z)-X.shape[1];cov=np.sum(res**2)/dof*np.linalg.inv(X.T@X)
        se=np.sqrt(cov[1,1]);tv=b[1]/se;p=2*stats.t.sf(abs(tv),dof)
        rows.append({"feature":c,"n":len(z),"pd_beta_std":float(b[1]),"p":float(p)})
    return pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame()

def main():
    meta=pd.read_csv(io.BytesIO(fetch(BASE+"preprocessed/file_list.csv")))
    meta["sid"]=meta.id.map(lambda x:f"{int(x):03d}")
    meta["group"]=meta.label.map({0:"healthy",1:"pd",2:"other"})
    jobs=[]
    for sid in meta.sid:
        p=CACHE/f"{sid}_ml.bin"
        if not p.exists():jobs.append((sid,p))
    def dl(j):
        sid,p=j;b=fetch(BASE+f"preprocessed/movement/{sid}_ml.bin",90)
        if b is None:return False
        p.write_bytes(b);return True
    with ThreadPoolExecutor(max_workers=24) as ex:
        fut={ex.submit(dl,j):j for j in jobs}
        for k,f in enumerate(as_completed(fut),1):
            _=f.result()
            if k%75==0:print("files",k,"/",len(jobs),flush=True)
    rows=[];fail={}
    for _,m in meta.iterrows():
        p=CACHE/f"{m.sid}_ml.bin"
        if not p.exists():fail[m.sid]="missing";continue
        try:
            x=np.fromfile(p,dtype=np.float32).reshape((-1,N))
            if x.shape!=(132,N):raise RuntimeError(str(x.shape))
            ds=[]
            for w in WRISTS:
                feat,conv=movement_features(x,w);ds.append({**conv,**feat})
            d=pd.DataFrame(ds);rows.append({"sid":m.sid,"group":m.group,"condition":m.condition,"age":m.age,"gender":m.gender,**d.mean(numeric_only=True).to_dict()})
        except Exception as e:fail[m.sid]=repr(e)
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"sid","group","condition","age","gender","rotation_rms","spectral_entropy","movement_peak_rate"})
    ph=group_contrast(df,"group","healthy","pd",tma);po=group_contrast(df,"group","other","pd",tma)
    ph.to_csv(OUT/"pd_vs_healthy.csv",index=False);po.to_csv(OUT/"pd_vs_other.csv",index=False)
    ah=adjust(df,ph.feature.tolist() if len(ph) else tma,"healthy");ao=adjust(df,po.feature.tolist() if len(po) else tma,"other")
    ah.to_csv(OUT/"pd_vs_healthy_adjusted.csv",index=False);ao.to_csv(OUT/"pd_vs_other_adjusted.csv",index=False)
    base=["age","rotation_rms","spectral_entropy","movement_peak_rate"];compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in df]
    a0,n0=auc(df,base,"healthy","pd");a1,_=auc(df,base+compact,"healthy","pd")
    b0,n1=auc(df,base,"other","pd");b1,_=auc(df,base+compact,"other","pd")
    inc={"pd_vs_healthy":{"n":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0},"pd_vs_other":{"n":n1,"auc_base":b0,"auc_plus_tma":b1,"delta_auc":b1-b0}}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summary={"dataset":"PADS official preprocessed rotation channels","tasks":TASKS,"n_subjects":len(df),"groups":df.groupby("group").size().to_dict(),"failures":fail,
      "top_pd_healthy":ph.head(10).to_dict(orient="records"),"top_pd_other":po.head(10).to_dict(orient="records"),
      "adjusted_pd_healthy":ah.head(10).to_dict(orient="records"),"adjusted_pd_other":ao.head(10).to_dict(orient="records"),
      "incremental":inc,"max_projection_error_ms":float(df.projection_error_ms_max.max())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
