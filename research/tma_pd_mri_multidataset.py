#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from research.tma_mri_exact_utils import (
    list_bold,read_tsv,process_dataset,math_audit,CONV,
    numeric_tma_columns,group_contrast,bh_adjust,adjust_group,partial_rank
)

ROOT=Path("research/tma_pd_mri_results")

def loocv_auc(df,features,group_col,g0,g1,subject_col=None):
    q=df[df[group_col].isin([g0,g1])].replace([np.inf,-np.inf],np.nan).dropna(subset=features).copy()
    q["y"]=(q[group_col]==g1).astype(int)
    pred=np.full(len(q),np.nan)
    if subject_col and subject_col in q:
        units=list(q[subject_col].unique())
        for s in units:
            te=q[subject_col]==s;tr=~te
            if q.loc[tr,"y"].nunique()<2:continue
            m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
            m.fit(q.loc[tr,features],q.loc[tr,"y"]);pred[te]=m.predict_proba(q.loc[te,features])[:,1]
    else:
        for i in range(len(q)):
            tr=np.arange(len(q))!=i
            m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
            m.fit(q.iloc[tr][features],q.iloc[tr].y);pred[i]=m.predict_proba(q.iloc[[i]][features])[0,1]
    ok=np.isfinite(pred)
    if ok.sum()<5:return np.nan,int(ok.sum())
    return float(roc_auc_score(q.loc[ok,"y"],pred[ok])),int(ok.sum())

def compact_tma(df):
    return [c for c in ["mean_H","sd_H","mean_D","sd_D","mean_lambda","sd_lambda","multilevel","pivot_rate",
                         "compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time","tree_root_fraction"]
            if c in df and df[c].notna().all() and np.nanstd(df[c])>1e-10]

def feature_cols(df,exclude):
    return numeric_tma_columns(df,exclude=exclude)

def parse_subject(key):
    m=re.search(r"/(sub-[^/]+)/",key)
    return m.group(1) if m else None

def analyze_5892():
    ds="ds005892";out=ROOT/ds;out.mkdir(parents=True,exist_ok=True)
    meta=read_tsv(ds,"participants.tsv")
    keys=list_bold(ds)
    df,fails,worked=process_dataset(ds,keys)
    df["participant_id"]=df.key.map(parse_subject)
    df=df.merge(meta,on="participant_id",how="left")
    df.to_csv(out/"scan_features.csv",index=False)
    (out/"worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    ex={"key","participant_id","group","age","sex"}|set(CONV)|{"tr","n_volumes","n_peaks","n_macroframes"}
    tma=feature_cols(df,ex)
    cons={}
    for g0,g1 in [("Control","PD-NC"),("Control","PD-MCI"),("PD-NC","PD-MCI")]:
        c=group_contrast(df,"group",g0,g1,tma);c.to_csv(out/f"{g1}_vs_{g0}.csv".replace("/","_"),index=False)
        cons[f"{g1}_vs_{g0}"]=c.head(12).to_dict(orient="records")
    # covariate adjusted: age, sex, conventional signal/QC measures
    sexmap={"M":1.0,"F":0.0};df["sex_num"]=df.sex.map(sexmap)
    adjusted={}
    cov=["age","sex_num","peak_interval_mean","peak_interval_cv","spectral_entropy","dvars_pct","tsnr"]
    for g0,g1 in [("Control","PD-NC"),("Control","PD-MCI"),("PD-NC","PD-MCI")]:
        rr=[]
        for c in tma:
            z=adjust_group(df,c,"group",g0,g1,cov)
            if z:rr.append(z)
        ad=pd.DataFrame(rr)
        if len(ad):ad["q_BH"]=bh_adjust(ad.p.values);ad=ad.sort_values("p")
        ad.to_csv(out/f"adjusted_{g1}_vs_{g0}.csv".replace("/","_"),index=False)
        adjusted[f"{g1}_vs_{g0}"]=ad.head(10).to_dict(orient="records")
    compact=compact_tma(df);inc={}
    base=[c for c in CONV+["age","sex_num"] if c in df]
    for g0,g1 in [("Control","PD-NC"),("Control","PD-MCI"),("PD-NC","PD-MCI")]:
        a0,n0=loocv_auc(df,base,"group",g0,g1);a1,n1=loocv_auc(df,base+compact,"group",g0,g1)
        inc[f"{g1}_vs_{g0}"]={"n":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0}
    summary={"dataset":ds,"description":"HC, PD normal cognition, PD mild cognitive impairment resting-state fMRI",
             "math_audit":math_audit(),"n_scans":len(df),"groups":df.groupby("group").size().to_dict(),
             "failures":fails,"top_contrasts":cons,"adjusted":adjusted,"incremental":inc,
             "max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))

def analyze_4392():
    ds="ds004392";out=ROOT/ds;out.mkdir(parents=True,exist_ok=True)
    parts=read_tsv(ds,"participants.tsv");dx=read_tsv(ds,"phenotype/dx_deidentified.tsv")
    hy=read_tsv(ds,"phenotype/hy_deidentified.tsv");cog=read_tsv(ds,"phenotype/cognitive_domains.tsv")
    keys=list_bold(ds);df,fails,worked=process_dataset(ds,keys);df["participant_id"]=df.key.map(parse_subject)
    df=df.merge(parts,on="participant_id",how="left").merge(dx,on="participant_id",how="left").merge(hy,on="participant_id",how="left").merge(cog,on="participant_id",how="left")
    df.to_csv(out/"scan_features.csv",index=False);(out/"worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    ex={"key","participant_id","Dx","age","sex","handedness","years_of_education","HY","attention","executive","global","language","memory","visuospatial"}|set(CONV)|{"tr","n_volumes","n_peaks","n_macroframes"}
    tma=feature_cols(df,ex)
    cons={}
    for g0,g1 in [("PD normal cognition","PD-MCI"),("PD-MCI","PDD"),("PD normal cognition","PDD")]:
        c=group_contrast(df,"Dx",g0,g1,tma);c.to_csv(out/f"{g1}_vs_{g0}.csv".replace(" ","_").replace("/","_"),index=False)
        cons[f"{g1}_vs_{g0}"]=c.head(12).to_dict(orient="records")
    # Ordinal diagnostic severity, cognitive global score, and Hoehn-Yahr.
    df["dx_order"]=df.Dx.map({"PD normal cognition":0,"PD-MCI":1,"PDD":2})
    corr={}
    for outcome in ["dx_order","global","HY"]:
        rr=[]
        for c in tma:
            q=df[[outcome,c]].dropna()
            if len(q)>=12 and q[outcome].nunique()>2:
                r=stats.spearmanr(q[outcome],q[c]);rr.append({"feature":c,"n":len(q),"rho":float(r.statistic),"p":float(r.pvalue)})
        rd=pd.DataFrame(rr)
        if len(rd):rd["q_BH"]=bh_adjust(rd.p.values);rd=rd.sort_values("p")
        rd.to_csv(out/f"{outcome}_correlations.csv",index=False);corr[outcome]=rd.head(12).to_dict(orient="records")
    # Adjust global cognitive and HY associations for age/education and basic BOLD/QC.
    partial={}
    cov=["age","years_of_education","peak_interval_mean","peak_interval_cv","spectral_entropy","dvars_pct","tsnr"]
    for outcome in ["global","HY"]:
        rr=[]
        for c in tma:
            z=partial_rank(df,outcome,c,cov)
            if z:rr.append(z)
        rd=pd.DataFrame(rr)
        if len(rd):rd["q_BH"]=bh_adjust(rd.p.values);rd=rd.sort_values("p")
        rd.to_csv(out/f"adjusted_{outcome}_correlations.csv",index=False);partial[outcome]=rd.head(12).to_dict(orient="records")
    compact=compact_tma(df);base=[c for c in CONV+["age","years_of_education"] if c in df]
    inc={}
    for g0,g1 in [("PD normal cognition","PD-MCI"),("PD normal cognition","PDD")]:
        a0,n0=loocv_auc(df,base,"Dx",g0,g1);a1,n1=loocv_auc(df,base+compact,"Dx",g0,g1)
        inc[f"{g1}_vs_{g0}"]={"n":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0}
    summary={"dataset":ds,"description":"PD cognition spectrum resting-state fMRI","math_audit":math_audit(),
             "n_scans":len(df),"groups":df.groupby("Dx").size().to_dict(),"failures":fails,
             "top_contrasts":cons,"correlations":corr,"adjusted_correlations":partial,"incremental":inc,
             "max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))

def analyze_5906():
    ds="ds005906";out=ROOT/ds;out.mkdir(parents=True,exist_ok=True)
    keys=list_bold(ds);df,fails,worked=process_dataset(ds,keys)
    df["subject"]=df.key.map(parse_subject)
    df["session"]=df.key.str.extract(r"/(ses-[^/]+)/",expand=False)
    df["condition"]=np.where(df.key.str.contains("task-restON"),"ON","OFF")
    df["run"]=df.key.str.extract(r"_run-(\d+)_",expand=False)
    df.to_csv(out/"scan_features.csv",index=False);(out/"worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    ex={"key","subject","session","condition","run"}|set(CONV)|{"tr","n_volumes","n_peaks","n_macroframes"}
    tma=feature_cols(df,ex)
    # Aggregate runs/sessions to independent subject-condition means for primary DBS test.
    nums=[c for c in df.select_dtypes(include=[np.number]).columns if c not in []]
    agg=df.groupby(["subject","condition"],as_index=False)[nums].mean()
    pc=__import__("research.tma_clinical_exact_utils",fromlist=["paired_contrast"]).paired_contrast(agg,"subject","condition","OFF","ON",tma)
    pc.to_csv(out/"dbs_on_vs_off.csv",index=False)
    # repeatability run 01 vs 02 within subject/session/condition
    repeat={}
    common=compact_tma(df)
    pairdf=df.dropna(subset=common).copy()
    A=pairdf[pairdf.run=="01"].set_index(["subject","session","condition"])
    B=pairdf[pairdf.run=="02"].set_index(["subject","session","condition"])
    ids=sorted(set(A.index)&set(B.index))
    if ids:
        both=pd.concat([A.loc[ids,common],B.loc[ids,common]])
        sd=both.std(ddof=1).replace(0,np.nan)
        same=[];different=[]
        for ident in ids:
            same.append(float(np.sqrt(np.nanmean(((A.loc[ident,common]-B.loc[ident,common])/sd)**2))))
        # compare to mismatched subjects under same condition when possible
        for ident in ids:
            for ident2 in ids:
                if ident2!=ident and ident2[2]==ident[2]:
                    different.append(float(np.sqrt(np.nanmean(((A.loc[ident,common]-B.loc[ident2,common])/sd)**2))))
        repeat={"n_pairs":len(same),"same_mean":float(np.mean(same)),"different_mean":float(np.mean(different)) if different else np.nan,
                "ratio":float(np.mean(same)/np.mean(different)) if different else np.nan}
    # LOSO condition discrimination using raw runs, conventional vs +TMA
    base=CONV
    a0,n0=loocv_auc(df,base,"condition","OFF","ON",subject_col="subject")
    a1,n1=loocv_auc(df,base+common,"condition","OFF","ON",subject_col="subject")
    inc={"n_predictions":n0,"auc_base":a0,"auc_plus_tma":a1,"delta_auc":a1-a0}
    summary={"dataset":ds,"description":"PD DBS ON/OFF repeated resting-state fMRI","math_audit":math_audit(),
             "n_scans":len(df),"n_subjects":int(df.subject.nunique()),"conditions":df.groupby("condition").size().to_dict(),
             "failures":fails,"top_dbs_effects":pc.head(12).to_dict(orient="records"),"repeatability":repeat,
             "incremental":inc,"max_projection_error_ms":float(df.projection_error_ms_max.max()) if len(df) else None}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True,choices=["ds005892","ds004392","ds005906"])
    ds=ap.parse_args().dataset
    if ds=="ds005892":analyze_5892()
    elif ds=="ds004392":analyze_4392()
    else:analyze_5906()

if __name__=="__main__":main()
