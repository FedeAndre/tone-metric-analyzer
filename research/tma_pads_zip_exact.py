#!/usr/bin/env python3
from __future__ import annotations
import json,zipfile,re
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy import signal,stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,group_contrast

ZIPURL="https://physionet.org/content/parkinsons-disease-smartwatch/get-zip/1.0.0/"
OUT=Path("research/tma_pads_exact_results");OUT.mkdir(parents=True,exist_ok=True)
ZPATH=Path("/tmp/pads.zip")
TASKS=["PointFinger","DrinkGlas","CrossArms","TouchNose"]; SIDES=["LeftWrist","RightWrist"]
FS=100.0;TOL=0.005

def ensure_zip():
    if ZPATH.exists() and ZPATH.stat().st_size>100_000_000:return
    with requests.get(ZIPURL,stream=True,timeout=1200,headers={"User-Agent":"Mozilla/5.0 TMA-research"}) as r:
        r.raise_for_status()
        with ZPATH.open("wb") as f:
            for c in r.iter_content(4*1024*1024):
                if c:f.write(c)
    print("zip bytes",ZPATH.stat().st_size,flush=True)

def group_condition(s):
    s=str(s).lower()
    if "healthy" in s:return "healthy"
    if "parkinson" in s:return "pd"
    return "other"

def select_three(y,t,lo,hi):
    ix=np.flatnonzero((t>=lo)&(t<hi))
    if len(ix)<50:return None
    yy=y[ix];peaks,_=signal.find_peaks(yy,distance=25,prominence=max(np.std(yy)*.10,1e-8))
    if len(peaks)<3:peaks,_=signal.find_peaks(yy,distance=20)
    if len(peaks)<3:return None
    sel=np.sort(ix[peaks[np.argsort(yy[peaks])[-3:]]])
    return [float(t[j]-lo) for j in sel]

def parse_ts(raw):
    # numpy fromstring is faster than pandas for 7-column numeric CSV
    a=np.genfromtxt(raw.decode().splitlines(),delimiter=",")
    if a.ndim!=2 or a.shape[1]<7:raise RuntimeError("bad timeseries")
    return a

def wrist_features(z,nmap,sid,side):
    frames=[];rms=[];ent=[];peakrates=[]
    for task in TASKS:
        key=f"{sid}_{task}_{side}.txt";n=nmap.get(key)
        if not n:raise RuntimeError("missing "+key)
        a=parse_ts(z.read(n));t=a[:,0];g=a[:,4:7]
        mag=signal.detrend(np.linalg.norm(g,axis=1));mag=signal.savgol_filter(mag,11,3,mode="interp")
        dur=float(t[-1]-t[0]);half=dur/2
        for h in range(2):
            lo=float(t[0]+h*half);hi=float(t[0]+(h+1)*half)
            offs=select_three(mag,t,lo,hi)
            if offs is None:raise RuntimeError(f"{task} half {h}: <3 peaks")
            frames.append({"duration_s":half,"events":[{"label":"MOVE","offset_s":x} for x in offs]})
        rms.append(float(np.sqrt(np.mean(mag**2))))
        f,px=signal.welch(mag,fs=FS,nperseg=min(512,len(mag)));m=(f>=.5)&(f<=12);q=px[m]
        if q.sum()>0:q=q/q.sum();ent.append(float(-np.sum(q*np.log(q+1e-15))/np.log(len(q))))
        peaks,_=signal.find_peaks(mag,distance=20,prominence=max(np.std(mag)*.1,1e-8));peakrates.append(float(len(peaks)/dur))
    feat,_,_,_=analyze_event_frames(frames,tol_s=TOL,label_features=False)
    return feat,{"gyro_rms":float(np.mean(rms)),"spectral_entropy":float(np.mean(ent)),"movement_peak_rate":float(np.mean(peakrates))}

def auc(df,features,g0,g1):
    q=df[df.group.isin([g0,g1])].dropna(subset=features).reset_index(drop=True);y=(q.group==g1).astype(int).to_numpy();p=np.zeros(len(q))
    for i in range(len(q)):
        tr=np.arange(len(q))!=i;m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        m.fit(q.loc[tr,features],y[tr]);p[i]=m.predict_proba(q.loc[[i],features])[0,1]
    return float(roc_auc_score(y,p)),len(q)

def adjust(df,ordered_features,g0):
    q=df[df.group.isin([g0,"pd"])].copy();q["pd"]=(q.group=="pd").astype(float);rows=[]
    for c in ordered_features[:12]:
        z=q[["pd","age","gyro_rms","movement_peak_rate",c]].dropna()
        if len(z)<20:continue
        y=(z[c]-z[c].mean())/z[c].std(ddof=1);X=np.column_stack([np.ones(len(z)),z.pd,z.age,z.gyro_rms,z.movement_peak_rate])
        b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b;dof=len(z)-X.shape[1];cov=np.sum(res**2)/dof*np.linalg.inv(X.T@X)
        se=np.sqrt(cov[1,1]);tv=b[1]/se;p=2*stats.t.sf(abs(tv),dof);rows.append({"feature":c,"n":len(z),"pd_beta_std":float(b[1]),"p":float(p)})
    return pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame()

def main():
    ensure_zip();z=zipfile.ZipFile(ZPATH)
    # map basename -> path; PADS basenames are subject-specific and unique for our targets
    nmap={Path(n).name:n for n in z.namelist() if n.endswith(".txt") or "/patients/patient_" in n}
    patient_names=[n for n in z.namelist() if re.search(r"/patients/patient_\d+\.json$",n)]
    rows=[];fail={}
    for k,n in enumerate(patient_names,1):
        m=json.loads(z.read(n));sid=str(m["id"]).zfill(3);grp=group_condition(m.get("condition"))
        side_rows=[]
        for side in SIDES:
            try:
                feat,conv=wrist_features(z,nmap,sid,side);side_rows.append({**conv,**feat})
            except Exception as e:fail[f"{sid}:{side}"]=repr(e)
        if side_rows:
            d=pd.DataFrame(side_rows);nums=d.mean(numeric_only=True).to_dict()
            rows.append({"sid":sid,"group":grp,"condition":m.get("condition"),"age":pd.to_numeric(m.get("age"),errors="coerce"),"n_wrists":len(d),**nums})
        if k%50==0:print("subjects",k,"/",len(patient_names),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"sid","group","condition","age","n_wrists","gyro_rms","spectral_entropy","movement_peak_rate"})
    ph=group_contrast(df,"group","healthy","pd",tma);po=group_contrast(df,"group","other","pd",tma)
    ph.to_csv(OUT/"pd_vs_healthy.csv",index=False);po.to_csv(OUT/"pd_vs_other_movement_disorders.csv",index=False)
    ah=adjust(df,ph.feature.tolist() if len(ph) else tma,"healthy");ao=adjust(df,po.feature.tolist() if len(po) else tma,"other")
    ah.to_csv(OUT/"pd_vs_healthy_adjusted.csv",index=False);ao.to_csv(OUT/"pd_vs_other_adjusted.csv",index=False)
    base=["age","gyro_rms","spectral_entropy","movement_peak_rate"];compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in df]
    a0,n0=auc(df,base,"healthy","pd");a1,_=auc(df,base+compact,"healthy","pd")
    b0,n1=auc(df,base,"other","pd");b1,_=auc(df,base+compact,"other","pd")
    inc={"pd_vs_healthy":{"n":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0},"pd_vs_other":{"n":n1,"auc_base":b0,"auc_plus_tma":b1,"delta_auc":b1-b0}}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summary={"dataset":"PADS smartwatch kinetic tasks","tasks":TASKS,"n_subjects":len(df),"groups":df.groupby("group").size().to_dict(),"n_wrist_failures":len(fail),
             "top_pd_healthy":ph.head(10).to_dict(orient="records"),"top_pd_other":po.head(10).to_dict(orient="records"),
             "adjusted_pd_healthy":ah.head(8).to_dict(orient="records"),"adjusted_pd_other":ao.head(8).to_dict(orient="records"),
             "incremental":inc,"max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
