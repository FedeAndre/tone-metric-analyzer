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
from research.tma_clinical_exact_utils import analyze_event_frames,numeric_tma_columns,group_contrast,bh_adjust

URL="https://physionet.org/files/nqmitcsxpd/1.0.0/neuroQWERTY.zip"
OUT=Path("research/tma_neuroqwerty_exact_results");OUT.mkdir(parents=True,exist_ok=True)
SEED=20260925
MAX_FRAMES=64
TOL=0.003

def read_zip():
    r=requests.get(URL,timeout=180,headers={"User-Agent":"Mozilla/5.0 TMA-research"});r.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(r.content))

def load_gt(z,cohort):
    name=[n for n in z.namelist() if n.endswith(f"{cohort}/GT_DataPD_MIT-CSXPD.csv") or n==f"{cohort}/GT_DataPD_MIT-CSXPD.csv"]
    if not name:
        name=[n for n in z.namelist() if cohort in n and n.endswith("GT_DataPD_MIT-CSXPD.csv")]
    if not name:raise RuntimeError(f"GT missing {cohort}")
    g=pd.read_csv(io.BytesIO(z.read(name[0])))
    g.columns=[str(c).strip() for c in g.columns]
    return g

def normalize_gt(v):
    s=str(v).strip().lower()
    if s in {"1","true","pd","parkinson","parkinsons","parkinson's disease","parkinson disease"}:return "pd"
    if s in {"0","false","hc","healthy","control","controls"}:return "control"
    try:return "pd" if float(s)>0 else "control"
    except:return s

def file_subject_visit(name):
    base=Path(name).name
    m=re.search(r"\.([0-9]+)_([0-9]{3})_014\.csv$",base)
    if not m:return None,None
    return int(m.group(1)),m.group(2)

def session_features(raw):
    d=pd.read_csv(io.BytesIO(raw),header=None,names=["key","hold","release","press"])
    for c in ["hold","release","press"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["press"]).sort_values("press")
    press=np.unique(np.round(d.press.to_numpy(float),4))
    if len(press)<80:raise RuntimeError("too few presses")
    dif=np.diff(press);valid=dif[(dif>0.03)&(dif<2.0)]
    if len(valid)<40:raise RuntimeError("too few valid IPIs")
    tactus=float(np.median(valid))
    measure=4*tactus
    start=float(press[0])
    usable=float(press[-1]-start)
    nf=min(MAX_FRAMES,int(usable//measure))
    if nf<20:raise RuntimeError(f"only {nf} frames")
    frames=[]
    for i in range(nf):
        a=start+i*measure;b=a+measure
        ev=[{"label":"PRESS","offset_s":float(t-a)} for t in press[(press>=a)&(press<b)]]
        frames.append({"duration_s":measure,"events":ev})
    feat,ev,piv,tree=analyze_event_frames(frames,tol_s=TOL,label_features=False)
    hold=d.hold.dropna().to_numpy(float)
    conv={"typing_speed_calc":float(len(press)/(press[-1]-press[0])),
          "median_ipi":tactus,"ipi_cv":float(np.std(valid,ddof=1)/np.mean(valid)),
          "hold_mean":float(np.mean(hold)),"hold_sd":float(np.std(hold,ddof=1)),
          "pause_fraction":float(np.mean(dif>2.0))}
    return feat,conv,ev

def auc_loocv(df,features):
    q=df.dropna(subset=features+["is_pd"]).reset_index(drop=True)
    y=q.is_pd.to_numpy(int);pred=np.zeros(len(q))
    for i in range(len(q)):
        tr=np.arange(len(q))!=i
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        m.fit(q.loc[tr,features],y[tr]);pred[i]=m.predict_proba(q.loc[[i],features])[0,1]
    return float(roc_auc_score(y,pred)),len(q)

def main():
    z=read_zip();sessions=[];fail={}
    gt_all=[]
    for cohort in ["MIT-CS1PD","MIT-CS2PD"]:
        gt=load_gt(z,cohort)
        # identify columns flexibly
        pidcol=next(c for c in gt.columns if c.lower()=="pid")
        gt["pid_int"]=pd.to_numeric(gt[pidcol],errors="coerce").astype("Int64")
        gt["cohort"]=cohort
        gt_all.append(gt)
    meta=pd.concat(gt_all,ignore_index=True,sort=False)
    gtcol=next(c for c in meta.columns if c.lower()=="gt")
    meta["group"]=meta[gtcol].map(normalize_gt)
    upcol=next((c for c in meta.columns if c.lower()=="updrs108"),None)
    speedcol=next((c for c in meta.columns if "typing" in c.lower() and "speed" in c.lower()),None)

    datafiles=[n for n in z.namelist() if "/data_" in n and n.lower().endswith(".csv")]
    for i,n in enumerate(datafiles,1):
        cohort="MIT-CS1PD" if "MIT-CS1PD/" in n else "MIT-CS2PD"
        pid,visit=file_subject_visit(n)
        if pid is None:continue
        mm=meta[(meta.cohort==cohort)&(meta.pid_int==pid)]
        if len(mm)!=1:continue
        try:
            feat,conv,ev=session_features(z.read(n))
            row={"cohort":cohort,"pid":pid,"subject":f"{cohort}:{pid}","visit":visit,
                 "group":mm.iloc[0]["group"],**conv,**feat}
            if upcol: row["updrs"]=pd.to_numeric(mm.iloc[0][upcol],errors="coerce")
            if speedcol: row["typing_speed_reported"]=pd.to_numeric(mm.iloc[0][speedcol],errors="coerce")
            sessions.append(row)
        except Exception as e:fail[n]=repr(e)
    sdf=pd.DataFrame(sessions);sdf.to_csv(OUT/"session_features.csv",index=False)
    # average sessions per subject for independent primary analysis
    numeric=sdf.select_dtypes(include=[np.number]).columns.tolist()
    keepmeta=["cohort","pid","subject","group"]
    agg=sdf.groupby(keepmeta,as_index=False)[numeric].mean()
    agg.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(agg,exclude=set(keepmeta)|{"pid","updrs","typing_speed_calc","typing_speed_reported","median_ipi","ipi_cv","hold_mean","hold_sd","pause_fraction"})
    con=group_contrast(agg,"group","control","pd",tma);con.to_csv(OUT/"pd_vs_control.csv",index=False)

    # UPDRS within PD
    sev=[]
    if "updrs" in agg:
        pdd=agg[agg.group=="pd"]
        for c in tma:
            q=pdd[["updrs",c]].dropna()
            if len(q)>=8 and q.updrs.nunique()>2:
                r=stats.spearmanr(q.updrs,q[c])
                sev.append({"feature":c,"n":len(q),"rho":float(r.statistic),"p":float(r.pvalue)})
    sevdf=pd.DataFrame(sev)
    if len(sevdf):sevdf["q_BH"]=bh_adjust(sevdf.p.values);sevdf=sevdf.sort_values("p")
    sevdf.to_csv(OUT/"updrs_correlations.csv",index=False)

    # CS1 repeatability between visits via RMS standardized feature distance
    rep={}
    cs=sdf[sdf.cohort=="MIT-CS1PD"].copy()
    visits=sorted(cs.visit.dropna().unique())
    if len(visits)>=2:
        A=cs[cs.visit==visits[0]].set_index("subject");B=cs[cs.visit==visits[1]].set_index("subject")
        subs=sorted(set(A.index)&set(B.index))
        common=[c for c in tma if c in A and c in B]
        allx=pd.concat([A.loc[subs,common],B.loc[subs,common]])
        mu=allx.mean();sd=allx.std(ddof=1).replace(0,np.nan)
        same=[];diff=[]
        for s in subs:
            same.append(float(np.sqrt(np.nanmean(((A.loc[s,common]-B.loc[s,common])/sd)**2))))
            for t in subs:
                if t!=s:diff.append(float(np.sqrt(np.nanmean(((A.loc[s,common]-B.loc[t,common])/sd)**2))))
        rep={"visits":visits[:2],"n_pairs":len(subs),"same_mean":float(np.mean(same)),"different_mean":float(np.mean(diff)),
             "ratio":float(np.mean(same)/np.mean(diff))}
    (OUT/"repeatability.json").write_text(json.dumps(rep,indent=2))

    base=[c for c in ["typing_speed_calc","median_ipi","ipi_cv","hold_mean","hold_sd","pause_fraction"] if c in agg]
    compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in agg]
    agg["is_pd"]=(agg.group=="pd").astype(int)
    auc0,n0=auc_loocv(agg,base);auc1,n1=auc_loocv(agg,base+compact)
    inc={"auc_base":auc0,"auc_plus_tma":auc1,"delta_auc":auc1-auc0,"n":n0,"base":base,"tma":compact}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summary={"dataset":"neuroQWERTY MIT-CSXPD","n_sessions":len(sdf),"n_subjects":len(agg),"groups":agg.groupby("group").size().to_dict(),
             "n_failed_files":len(fail),"top_pd_control":con.head(10).to_dict(orient="records"),
             "top_updrs":sevdf.head(10).to_dict(orient="records"),"repeatability":rep,"incremental":inc,
             "max_projection_error_ms":float(sdf.projection_error_ms_max.max()) if len(sdf) else None}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
