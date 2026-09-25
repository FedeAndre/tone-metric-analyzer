#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy import stats
from research.tma_pads_exact import BASE,TASKS,SIDES,get,group_condition,wrist_features,auc_loocv,adjusted_pd
from research.tma_clinical_exact_utils import numeric_tma_columns,group_contrast

OUT=Path("research/tma_pads_fast_results");OUT.mkdir(parents=True,exist_ok=True)
CACHE=Path("/tmp/pads_bal");CACHE.mkdir(exist_ok=True)
RNG=np.random.default_rng(20260925);N_PER=20

def main():
    def fp(i):
        sid=f"{i:03d}";b=get(BASE+f"patients/patient_{sid}.json",30)
        if b is None:return None
        x=json.loads(b);return {"sid":sid,"condition":x.get("condition"),"group":group_condition(x.get("condition")),
            "age":pd.to_numeric(x.get("age"),errors="coerce"),"gender":x.get("gender")}
    with ThreadPoolExecutor(max_workers=24) as ex:
        pats=[x for x in ex.map(fp,range(1,470)) if x]
    pdf=pd.DataFrame(pats)
    chosen=[]
    for g in ["healthy","pd","other"]:
        ids=pdf[pdf.group==g].sid.to_numpy()
        RNG.shuffle(ids);chosen.extend(ids[:min(N_PER,len(ids))].tolist())
    pdf=pdf[pdf.sid.isin(chosen)].copy();pdf.to_csv(OUT/"selected_subjects.csv",index=False)
    jobs=[]
    for sid in pdf.sid:
        for task in TASKS:
            for side in SIDES:
                p=CACHE/f"{sid}_{task}_{side}.txt";jobs.append((sid,task,side,p))
    def fj(j):
        sid,task,side,p=j
        if p.exists():return True
        b=get(BASE+f"movement/timeseries/{sid}_{task}_{side}.txt",90)
        if b is None:return False
        p.write_bytes(b);return True
    with ThreadPoolExecutor(max_workers=24) as ex:
        fut={ex.submit(fj,j):j for j in jobs}
        for k,f in enumerate(as_completed(fut),1):
            _=f.result()
            if k%200==0:print("files",k,"/",len(jobs),flush=True)
    rows=[];fail={}
    for _,m in pdf.iterrows():
        ds=[]
        for side in SIDES:
            files={task:CACHE/f"{m.sid}_{task}_{side}.txt" for task in TASKS}
            try:
                feat,conv=wrist_features(files);ds.append({**conv,**feat})
            except Exception as e:fail[f"{m.sid}:{side}"]=repr(e)
        if ds:
            d=pd.DataFrame(ds);rows.append({"sid":m.sid,"group":m.group,"condition":m.condition,"age":m.age,"n_wrists":len(d),**d.mean(numeric_only=True).to_dict()})
    df=pd.DataFrame(rows);df.to_csv(OUT/"subject_features.csv",index=False)
    tma=numeric_tma_columns(df,exclude={"sid","group","condition","age","n_wrists","gyro_rms","spectral_entropy","movement_peak_rate","left_right_feature_distance"})
    ph=group_contrast(df,"group","healthy","pd",tma);po=group_contrast(df,"group","other","pd",tma)
    ph.to_csv(OUT/"pd_vs_healthy.csv",index=False);po.to_csv(OUT/"pd_vs_other.csv",index=False)
    ah=adjusted_pd(df,ph.feature.tolist() if len(ph) else tma,"healthy");ao=adjusted_pd(df,po.feature.tolist() if len(po) else tma,"other")
    ah.to_csv(OUT/"pd_vs_healthy_adjusted.csv",index=False);ao.to_csv(OUT/"pd_vs_other_adjusted.csv",index=False)
    base=["age","gyro_rms","spectral_entropy","movement_peak_rate"];compact=[c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in df]
    a0,n0,p0=auc_loocv(df.assign(is_pd=(df.group=="pd").astype(int))[df.group.isin(["healthy","pd"])],base)
    a1,n1,p1=auc_loocv(df.assign(is_pd=(df.group=="pd").astype(int))[df.group.isin(["healthy","pd"])],base+compact)
    b0,n2,p2=auc_loocv(df.assign(is_pd=(df.group=="pd").astype(int))[df.group.isin(["other","pd"])],base)
    b1,n3,p3=auc_loocv(df.assign(is_pd=(df.group=="pd").astype(int))[df.group.isin(["other","pd"])],base+compact)
    inc={"pd_vs_healthy":{"n":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0},
         "pd_vs_other":{"n":n2,"auc_base":b0,"auc_plus_tma":b1,"delta_auc":b1-b0}}
    (OUT/"incremental.json").write_text(json.dumps(inc,indent=2))
    summ={"dataset":"PADS fast balanced sensitivity","selected_per_group":N_PER,"n_analyzed":len(df),"groups":df.groupby("group").size().to_dict(),
          "failures":len(fail),"top_pd_healthy":ph.head(10).to_dict(orient="records"),"top_pd_other":po.head(10).to_dict(orient="records"),
          "adjusted_healthy":ah.head(8).to_dict(orient="records"),"adjusted_other":ao.head(8).to_dict(orient="records"),
          "incremental":inc,"max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (OUT/"summary.json").write_text(json.dumps(summ,indent=2,default=str));print(json.dumps(summ,indent=2,default=str))
if __name__=="__main__":main()
