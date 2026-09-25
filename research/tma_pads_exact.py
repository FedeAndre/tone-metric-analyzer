#!/usr/bin/env python3
from __future__ import annotations
import json,math,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd,requests
from scipy import signal,stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,group_contrast,bh_adjust

BASE="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
OUT=Path("research/tma_pads_exact_results");OUT.mkdir(parents=True,exist_ok=True)
CACHE=Path("/tmp/pads_tma");CACHE.mkdir(exist_ok=True)
TASKS=["PointFinger","DrinkGlas","CrossArms","TouchNose"]
SIDES=["LeftWrist","RightWrist"]
TOL=0.005
FS=100.0
SEED=20260925

def get(url,timeout=90):
    for k in range(4):
        try:
            r=requests.get(url,timeout=timeout,headers={"User-Agent":"Mozilla/5.0 TMA-research"})
            if r.status_code==200:return r.content
        except: pass
    return None

def group_condition(s):
    s=str(s).lower()
    if "healthy" in s:return "healthy"
    if "parkinson" in s:return "pd"
    return "other"

def select_three(y,t,lo,hi):
    m=(t>=lo)&(t<hi);ix=np.flatnonzero(m)
    if len(ix)<50:return None
    yy=y[ix]
    peaks,_=signal.find_peaks(yy,distance=max(10,int(.25*FS)),prominence=max(np.std(yy)*0.10,1e-8))
    if len(peaks)<3:
        peaks,_=signal.find_peaks(yy,distance=max(8,int(.20*FS)))
    if len(peaks)<3:return None
    order=peaks[np.argsort(yy[peaks])[-3:]]
    sel=np.sort(ix[order])
    return [float(t[j]-lo) for j in sel]

def wrist_features(files):
    frames=[];rms=[];ent=[];peakrates=[]
    for task in TASKS:
        p=files[task]
        a=np.loadtxt(p,delimiter=",")
        t=a[:,0];g=a[:,4:7]
        mag=np.linalg.norm(g,axis=1)
        mag=signal.detrend(mag)
        # light smoothing preserves movement timing but suppresses sample noise
        mag=signal.savgol_filter(mag,11,3,mode="interp")
        dur=float(t[-1]-t[0])
        half=dur/2
        for h in range(2):
            lo=float(t[0]+h*half);hi=float(t[0]+(h+1)*half)
            offs=select_three(mag,t,lo,hi)
            if offs is None:raise RuntimeError(f"{task} half {h}: <3 movement peaks")
            frames.append({"duration_s":half,"events":[{"label":"MOVE","offset_s":x} for x in offs]})
        rms.append(float(np.sqrt(np.mean(mag**2))))
        f,px=signal.welch(mag,fs=FS,nperseg=min(512,len(mag)))
        mm=(f>=0.5)&(f<=12);q=px[mm]
        if q.sum()>0:
            q=q/q.sum();ent.append(float(-np.sum(q*np.log(q+1e-15))/np.log(len(q))))
        peaks,_=signal.find_peaks(mag,distance=20,prominence=max(np.std(mag)*.1,1e-8))
        peakrates.append(float(len(peaks)/dur))
    feat,ev,piv,tree=analyze_event_frames(frames,tol_s=TOL,label_features=False)
    conv={"gyro_rms":float(np.mean(rms)),"spectral_entropy":float(np.mean(ent)),
          "movement_peak_rate":float(np.mean(peakrates))}
    return feat,conv

def auc_loocv(df,features,g0,g1):
    q=df[df.group.isin([g0,g1])].dropna(subset=features).copy()
    q["y"]=(q.group==g1).astype(int);q=q.reset_index(drop=True)
    y=q.y.to_numpy();pred=np.zeros(len(q))
    for i in range(len(q)):
        tr=np.arange(len(q))!=i
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        m.fit(q.loc[tr,features],y[tr]);pred[i]=m.predict_proba(q.loc[[i],features])[0,1]
    return float(roc_auc_score(y,pred)),len(q)

def adjusted_pd(df,features,g0):
    q=df[df.group.isin([g0,"pd"])].copy();q["pd"]=(q.group=="pd").astype(float)
    rows=[]
    for c in features[:12]:
        z=q[["pd","age","gyro_rms","movement_peak_rate",c]].dropna()
        if len(z)<20:continue
        y=(z[c]-z[c].mean())/z[c].std(ddof=1)
        X=np.column_stack([np.ones(len(z)),z.pd,z.age,z.gyro_rms,z.movement_peak_rate])
        b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b;dof=len(z)-X.shape[1]
        s2=float(np.sum(res**2)/dof);cov=s2*np.linalg.inv(X.T@X);se=float(np.sqrt(cov[1,1]))
        tt=float(b[1]/se);p=float(2*stats.t.sf(abs(tt),dof))
        rows.append({"feature":c,"n":len(z),"pd_beta_std":float(b[1]),"p":p})
    return pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame()

def main():
    # patient metadata
    patients=[]
    def fetch_patient(i):
        sid=f"{i:03d}";b=get(BASE+f"patients/patient_{sid}.json",30)
        if b is None:return None
        try:
            x=json.loads(b);return {"sid":sid,"condition":x.get("condition"),"group":group_condition(x.get("condition")),
                "age":pd.to_numeric(x.get("age"),errors="coerce"),"gender":x.get("gender")}
        except:return None
    with ThreadPoolExecutor(max_workers=16) as ex:
        for x in ex.map(fetch_patient,range(1,470)):
            if x:patients.append(x)
    pdf=pd.DataFrame(patients);pdf.to_csv(OUT/"patient_groups.csv",index=False)

    # download selected kinetic tasks
    jobs=[]
    for sid in pdf.sid:
        for task in TASKS:
            for side in SIDES:
                p=CACHE/f"{sid}_{task}_{side}.txt"
                if not p.exists():jobs.append((sid,task,side,p))
    def fetch_job(j):
        sid,task,side,p=j;b=get(BASE+f"movement/timeseries/{sid}_{task}_{side}.txt",90)
        if b is not None:p.write_bytes(b);return True
        return False
    with ThreadPoolExecutor(max_workers=16) as ex:
        fut={ex.submit(fetch_job,j):j for j in jobs}
        for k,f in enumerate(as_completed(fut),1):
            _=f.result()
            if k%500==0:print("downloaded",k,"/",len(jobs),flush=True)

    rows=[];fail={}
    for _,m in pdf.iterrows():
        sid=m.sid
        side_rows=[]
        for side in SIDES:
            files={task:CACHE/f"{sid}_{task}_{side}.txt" for task in TASKS}
            if not all(p.exists() for p in files.values()):
                fail[f"{sid}:{side}"]="missing file";continue
            try:
                feat,conv=wrist_features(files);side_rows.append({"side":side,**conv,**feat})
            except Exception as e:fail[f"{sid}:{side}"]=repr(e)
        if not side_rows:continue
        d=pd.DataFrame(side_rows)
        nums=d.select_dtypes(include=[np.number]).mean().to_dict()
        nums["left_right_feature_distance"]=np.nan
        if len(d)==2:
            fcols=[c for c in d.columns if c not in {"side"} and pd.api.types.is_numeric_dtype(d[c])]
            # normalized later; here raw RMS difference over scale-free ratios is not meaningful, omit.
        rows.append({"sid":sid,"group":m.group,"condition":m.condition,"age":m.age,"gender":m.gender,"n_wrists":len(d),**nums})
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"sid","group","condition","age","gender","n_wrists","gyro_rms","spectral_entropy","movement_peak_rate","left_right_feature_distance"})
    ph=group_contrast(df,"group","healthy","pd",tma);ph.to_csv(OUT/"pd_vs_healthy.csv",index=False)
    po=group_contrast(df,"group","other","pd",tma);po.to_csv(OUT/"pd_vs_other_movement_disorders.csv",index=False)
    adjh=adjusted_pd(df,ph.feature.tolist() if len(ph) else tma,"healthy");adjh.to_csv(OUT/"pd_vs_healthy_adjusted.csv",index=False)
    adjo=adjusted_pd(df,po.feature.tolist() if len(po) else tma,"other");adjo.to_csv(OUT/"pd_vs_other_adjusted.csv",index=False)

    base=["age","gyro_rms","spectral_entropy","movement_peak_rate"]
    compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in df]
    ah0,n1=auc_loocv(df,base,"healthy","pd");ah1,_=auc_loocv(df,base+compact,"healthy","pd")
    ao0,n2=auc_loocv(df,base,"other","pd");ao1,_=auc_loocv(df,base+compact,"other","pd")
    inc={"pd_vs_healthy":{"n":n1,"auc_base":ah0,"auc_plus_tma":ah1,"delta_auc":ah1-ah0},
         "pd_vs_other":{"n":n2,"auc_base":ao0,"auc_plus_tma":ao1,"delta_auc":ao1-ao0}}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summary={"dataset":"PADS smartwatch kinetic tasks","tasks":TASKS,"n_subjects":len(df),"groups":df.groupby("group").size().to_dict(),
             "n_wrist_failures":len(fail),"top_pd_healthy":ph.head(10).to_dict(orient="records"),
             "top_pd_other":po.head(10).to_dict(orient="records"),"adjusted_pd_healthy":adjh.head(8).to_dict(orient="records"),
             "adjusted_pd_other":adjo.head(8).to_dict(orient="records"),"incremental":inc,
             "max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
