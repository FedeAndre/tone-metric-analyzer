#!/usr/bin/env python3
from __future__ import annotations

import json, math, time
from fractions import Fraction
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import wfdb
from scipy import stats, ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from tone_metric.engine import analyze
from tone_metric.models import Hit, MeasureInfo
from tone_metric.waves import build_wave_profile
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tone_metric.theory import (
    boundary_sequence, boundary_sequence_recurrence,
    pascal_binomial_mod, lucas_binomial_mod
)

SEED=20260925
RNG=np.random.default_rng(SEED)
BASE="https://physionet.org/files/gaitndd/1.0.0/"
CACHE=Path("/tmp/gaitndd_exact"); CACHE.mkdir(parents=True,exist_ok=True)
OUT=Path("research/tma_gait_ndd_exact_results"); OUT.mkdir(parents=True,exist_ok=True)
N_CYCLES=64
HALF=32
PROJ_TOL_S=0.005
EVENT_TYPES=("LHS","RTO","RHS","LTO")
PERM_B=5000

def dl(name):
    p=CACHE/name
    if p.exists() and p.stat().st_size>10:return p
    last=None
    for k in range(6):
        try:
            r=requests.get(BASE+name,timeout=120,headers={"User-Agent":"Mozilla/5.0 TMA-research"})
            r.raise_for_status(); p.write_bytes(r.content); return p
        except Exception as e:
            last=e; time.sleep(min(3*(k+1),15))
    raise RuntimeError(f"download failed {name}: {last}")

def get_records():
    p=dl("RECORDS")
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]

def get_metadata():
    p=dl("subject-description.txt")
    lines=p.read_text(errors="replace").splitlines()[1:]
    rows=[]; i=0
    while i<len(lines):
        toks=lines[i].split()
        if not toks:
            i+=1; continue
        rec=toks[0]
        if not (rec.startswith("control") or rec.startswith("park") or rec.startswith("hunt") or rec.startswith("als")):
            i+=1; continue
        if len(toks)==7 and i+1<len(lines):
            nxt=lines[i+1].split()
            if len(nxt)==1:
                toks=toks+nxt; i+=1
        if len(toks)<8:
            i+=1; continue
        grp="control" if rec.startswith("control") else "pd" if rec.startswith("park") else "hd" if rec.startswith("hunt") else "als"
        def num(s):
            try:return float(s)
            except:return np.nan
        rows.append({
            "record":rec,"group":grp,"age":num(toks[2]),"height_m":num(toks[3]),"weight_kg":num(toks[4]),
            "sex":toks[5],"gait_speed":num(toks[6]),"severity":num(toks[7])
        })
        i+=1
    return pd.DataFrame(rows)

def two_means_threshold(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    c=np.array(np.nanpercentile(x,[20,80]),float)
    for _ in range(50):
        d=np.abs(x[:,None]-c[None,:]); lab=np.argmin(d,axis=1)
        nc=np.array([x[lab==j].mean() if np.any(lab==j) else c[j] for j in range(2)])
        if np.max(np.abs(nc-c))<1e-10:break
        c=nc
    c=np.sort(c)
    return float(np.mean(c)),c.tolist()

def clean_contact(x,fs):
    thr,centers=two_means_threshold(x)
    b=np.asarray(x,float)>=thr
    # 23 ms median filter; preserve true gait edges, remove digital chatter.
    size=max(3,int(round(.023*fs))//2*2+1)
    b=ndimage.median_filter(b.astype(np.uint8),size=size,mode="nearest")>0
    # Remove residual contact/swing islands shorter than 40 ms.
    minrun=max(2,int(round(.04*fs)))
    y=b.copy(); changes=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    for a,z in zip(changes[:-1],changes[1:]):
        if z-a<minrun and a>0 and z<len(y) and y[a-1]==y[z]:
            y[a:z]=y[a-1]
    return y,thr,centers

def trans(b):
    hs=np.flatnonzero((~b[:-1])&b[1:])+1
    to=np.flatnonzero(b[:-1]&(~b[1:]))+1
    return hs.astype(int),to.astype(int)

def first_between(v,lo,hi):
    k=np.searchsorted(v,lo+1,side="left")
    return int(v[k]) if k<len(v) and v[k]<hi else None

def project(sample,start,end,fs):
    if sample==start:return Fraction(0),0,0.0,0.0
    raw=Fraction(2*(sample-start),end-start); rf=float(raw); cyc=(end-start)/fs
    for depth in range(21):
        den=2**depth; k=int(math.floor(rf*den+.5)); q=Fraction(k,den)
        err=abs(float(q-raw)*cyc/2)
        if err<=PROJ_TOL_S+1e-12:return q,depth,1000*err,rf
    return raw,-1,0.0,rf

def download_record(rec):
    for ext in ("hea","let","rit","ts"):
        dl(f"{rec}.{ext}")
    rr=wfdb.rdrecord(str(CACHE/rec),physical=True)
    if rr.p_signal.shape[1]!=2 or abs(float(rr.fs)-300)>1e-6:
        raise RuntimeError(f"{rec}: unexpected signal shape/fs {rr.p_signal.shape}/{rr.fs}")
    return rr,np.loadtxt(CACHE/f"{rec}.ts")

def extract_cycles(rec):
    rr,ts=download_record(rec); fs=float(rr.fs)
    left=np.asarray(rr.p_signal[:,0],float); right=np.asarray(rr.p_signal[:,1],float)
    lc,lthr,lcent=clean_contact(left,fs); rc,rthr,rcent=clean_contact(right,fs)
    lhs,lto=trans(lc); rhs,rto=trans(rc)
    # Dataset's derived time series starts after ~20 s; match that stable segment.
    lo=int(round(20*fs)); hi=int(round(298*fs))
    lhs=lhs[(lhs>=lo)&(lhs<=hi)]
    cycles=[]
    for a,b in zip(lhs[:-1],lhs[1:]):
        dur=(b-a)/fs
        if not (.5<=dur<=2.8):continue
        er=first_between(rto,a,b)
        if er is None:continue
        eh=first_between(rhs,er,b)
        if eh is None:continue
        el=first_between(lto,eh,b)
        if el is None:continue
        if not (a<er<eh<el<b):continue
        events=[]
        for label,s in (("LHS",a),("RTO",er),("RHS",eh),("LTO",el)):
            q,dep,err,raw=project(s,a,b,fs)
            events.append({"label":label,"sample":int(s),"raw_phase":raw,
                           "phase_num":q.numerator,"phase_den":q.denominator,
                           "projection_depth":dep,"projection_error_ms":err})
        cycles.append({"start_sample":int(a),"end_sample":int(b),"duration_s":dur,"events":events})
        if len(cycles)>=N_CYCLES:break
    if len(cycles)<N_CYCLES:
        raise RuntimeError(f"{rec}: only {len(cycles)} valid cycles")

    # Extraction validation against distributed derived time series.
    ex_stride=np.array([c["duration_s"] for c in cycles])
    ex_stance=[];ex_swing=[]
    for c in cycles:
        a=c["start_sample"]; b=c["end_sample"]
        lto_s=next(e["sample"] for e in c["events"] if e["label"]=="LTO")
        ex_stance.append((lto_s-a)/fs); ex_swing.append((b-lto_s)/fs)
    valid={
        "threshold_left":lthr,"threshold_right":rthr,
        "centers_left":lcent,"centers_right":rcent,
        "median_stride_extracted":float(np.median(ex_stride)),
        "median_stride_ts":float(np.nanmedian(ts[:,1])),
        "median_stride_abs_diff":float(abs(np.median(ex_stride)-np.nanmedian(ts[:,1]))),
        "median_left_stance_extracted":float(np.median(ex_stance)),
        "median_left_stance_ts":float(np.nanmedian(ts[:,7])),
        "median_left_stance_abs_diff":float(abs(np.median(ex_stance)-np.nanmedian(ts[:,7]))),
        "median_left_swing_extracted":float(np.median(ex_swing)),
        "median_left_swing_ts":float(np.nanmedian(ts[:,3])),
        "median_left_swing_abs_diff":float(abs(np.median(ex_swing)-np.nanmedian(ts[:,3]))),
        "valid_cycles":len(cycles)
    }
    return cycles,valid,ts

def build_hits(cycles):
    hits=[];labels=[];cis=[];measures=[]
    for i,c in enumerate(cycles):
        start=Fraction(2*i)
        measures.append(MeasureInfo(index=i,number=str(i+1),start=start,full_duration=Fraction(2),
            actual_duration=Fraction(2),pickup_shift=Fraction(0),numerator=2,denominator=4,
            implicit=False,opening_anacrusis=False))
        for e in c["events"]:
            q=Fraction(e["phase_num"],e["phase_den"])
            hits.append(Hit(onset=start+q,duration=Fraction(0),measure_index=i,measure_number=str(i+1),
                offset_in_measure=q,sources=[],canonical_recovered=False))
            labels.append(e["label"]);cis.append(i)
    order=sorted(range(len(hits)),key=lambda j:(hits[j].onset,EVENT_TYPES.index(labels[j])))
    return [hits[j] for j in order],measures,[labels[j] for j in order],[cis[j] for j in order]

def analyze_cycles(cycles):
    hits,measures,labels,cis=build_hits(cycles)
    result=analyze(hits,measures)
    erows=[e for seg in result["segments"] for e in seg["events"]]
    if len(erows)!=len(labels):raise RuntimeError("engine event mismatch")
    ev=pd.DataFrame({
        "label":labels,"cycle_index":cis,
        "H":[int(r["tone_metric_height"]) for r in erows],
        "D":[int(r["tone_metric_density"]) for r in erows],
        "lambda":[int(r["lowest_tone_metric_level"]) for r in erows],
        "levels":[";".join(map(str,r["tone_metric_levels"])) for r in erows],
        "onset":[r["onset_quarter"] for r in erows]
    })
    wave=build_wave_profile(result); piv=build_pivot_profile(wave); tree=build_tree_profile(wave)
    br=tree.get("branches",[])
    f={
        "n_cycles":len(cycles),"n_events":len(ev),
        "mean_H":float(ev.H.mean()),"mean_D":float(ev.D.mean()),"mean_lambda":float(ev["lambda"].mean()),
        "multilevel":float((ev.D>1).mean()),"pivot_rate":float(len(piv)/len(ev)),
        "compound_pivot_fraction":float(np.mean([p["compound"] for p in piv])) if piv else np.nan,
        "pivot_depth":float(np.mean([p["drop_depth"] for p in piv])) if piv else np.nan,
        "tree_span_events":float(np.mean([int(b["target_node_index"])-int(b["source_node_index"]) for b in br])) if br else np.nan,
        "tree_span_time":float(np.mean([float(Fraction(str(b["distance_quarter"]))) for b in br])) if br else np.nan,
        "tree_root_fraction":float(np.mean([bool(n["root"]) for n in tree.get("nodes",[])])) if tree.get("nodes") else np.nan,
    }
    for label in EVENT_TYPES:
        z=ev[ev.label==label]
        f[f"{label}_mean_H"]=float(z.H.mean()); f[f"{label}_mean_D"]=float(z.D.mean())
        f[f"{label}_mean_lambda"]=float(z["lambda"].mean()); f[f"{label}_multilevel"]=float((z.D>1).mean())
    depths=[e["projection_depth"] for c in cycles for e in c["events"] if e["label"]!="LHS"]
    errs=[e["projection_error_ms"] for c in cycles for e in c["events"] if e["label"]!="LHS"]
    f["projection_depth_mean"]=float(np.mean(depths));f["projection_error_ms_mean"]=float(np.mean(errs));f["projection_error_ms_max"]=float(np.max(errs))
    return f,ev,piv,tree

def hedges_g(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float); n0=len(a);n1=len(b)
    sp=((n0-1)*np.var(a,ddof=1)+(n1-1)*np.var(b,ddof=1))/(n0+n1-2)
    if sp<=0:return np.nan
    d=(np.mean(b)-np.mean(a))/math.sqrt(sp); J=1-3/(4*(n0+n1)-9)
    return float(J*d)

def bh_adjust(p):
    p=np.asarray(p,float);n=len(p);order=np.argsort(p);q=np.empty(n,float);last=1.0
    for rank,idx in reversed(list(enumerate(order,start=1))):
        last=min(last,p[idx]*n/rank);q[idx]=last
    return np.minimum(q,1)

def conventional_features(ts):
    # Distributed derived gait variables, summarized per subject.
    names=["L_stride","R_stride","L_swing","R_swing","L_swing_pct","R_swing_pct",
           "L_stance","R_stance","L_stance_pct","R_stance_pct","double_support","double_support_pct"]
    out={}
    for j,n in enumerate(names,start=1):
        x=np.asarray(ts[:,j],float)
        out[f"{n}_mean"]=float(np.nanmean(x));out[f"{n}_sd"]=float(np.nanstd(x,ddof=1))
    return out

def loocv_auc(df,features,label_col="is_pd"):
    use=df.dropna(subset=features+[label_col]).reset_index(drop=True)
    y=use[label_col].to_numpy(int)
    pred=np.zeros(len(use))
    for i in range(len(use)):
        tr=np.arange(len(use))!=i
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,class_weight="balanced"))
        model.fit(use.loc[tr,features],y[tr]);pred[i]=model.predict_proba(use.loc[[i],features])[0,1]
    return float(roc_auc_score(y,pred)),len(use),pred

def main():
    # Mathematical implementation audit: paper derivation and engine must agree.
    audit={
      "boundary_closed":boundary_sequence(2,12),
      "boundary_recurrence":boundary_sequence_recurrence(2,12),
      "boundary_match":boundary_sequence(2,12)==boundary_sequence_recurrence(2,12),
      "lucas_pascal_mismatches":0
    }
    for n in range(64):
        for k in range(n+1):
            if pascal_binomial_mod(n,k,2)!=lucas_binomial_mod(n,k,2):
                audit["lucas_pascal_mismatches"]+=1
    (OUT/"paper_math_audit.json").write_text(json.dumps(audit,indent=2))

    meta=get_metadata(); records=get_records()
    rows=[]; validations=[]; conventional=[]; failures={}
    first_events={}
    for i,rec in enumerate(records,1):
        print(f"[{i:02d}/{len(records)}] {rec}",flush=True)
        try:
            cycles,val,ts=extract_cycles(rec)
            feat,ev,piv,tree=analyze_cycles(cycles)
            m=meta[meta.record==rec]
            if len(m)!=1:raise RuntimeError("metadata missing")
            md=m.iloc[0].to_dict()
            rows.append({**md,**feat})
            validations.append({"record":rec,"group":md["group"],**val})
            conventional.append({"record":rec,**conventional_features(ts)})
            if rec in ("park1","control1","hunt1","als1"):
                first_events[rec]={
                    "first_cycle":cycles[0],
                    "first_16_events":ev.head(16).to_dict(orient="records"),
                    "n_pivots":len(piv),"n_tree_nodes":len(tree.get("nodes",[])),"n_tree_branches":len(tree.get("branches",[]))
                }
        except Exception as e:
            failures[rec]=repr(e); print("FAIL",rec,repr(e),flush=True)
    featdf=pd.DataFrame(rows);valdf=pd.DataFrame(validations);convdf=pd.DataFrame(conventional)
    featdf.to_csv(OUT/"subject_tma_features.csv",index=False);valdf.to_csv(OUT/"event_extraction_validation.csv",index=False)
    pd.DataFrame(conventional).to_csv(OUT/"conventional_features.csv",index=False)
    (OUT/"worked_event_examples.json").write_text(json.dumps(first_events,indent=2,default=str))

    # Four-group omnibus and pairwise PD contrasts.
    numeric=[c for c in featdf.columns if c not in {"record","group","sex"} and pd.api.types.is_numeric_dtype(featdf[c])]
    diagnostics={"projection_depth_mean","projection_error_ms_mean","projection_error_ms_max"}
    tma=[c for c in numeric if c not in {"age","height_m","weight_kg","gait_speed","severity","n_cycles","n_events"}|diagnostics]
    omni=[]
    for c in tma:
        groups=[featdf.loc[featdf.group==g,c].dropna().values for g in ["control","pd","hd","als"]]
        if min(map(len,groups))<3:continue
        h,p=stats.kruskal(*groups)
        omni.append({"feature":c,"H":float(h),"p":float(p)})
    omni=pd.DataFrame(omni)
    if len(omni):
        omni["q_BH"]=bh_adjust(omni.p.values);omni=omni.sort_values("p")
    omni.to_csv(OUT/"four_group_omnibus.csv",index=False)

    pairs=[]
    for other in ["control","hd","als"]:
        for c in tma:
            a=featdf.loc[featdf.group==other,c].dropna().values
            b=featdf.loc[featdf.group=="pd",c].dropna().values
            if len(a)<3 or len(b)<3:continue
            tt=stats.ttest_ind(a,b,equal_var=False)
            pairs.append({"contrast":f"pd-vs-{other}","feature":c,
                "other_mean":float(np.mean(a)),"pd_mean":float(np.mean(b)),
                "diff_pd_minus_other":float(np.mean(b)-np.mean(a)),
                "welch_p":float(tt.pvalue),"hedges_g":hedges_g(a,b)})
    pairdf=pd.DataFrame(pairs)
    if len(pairdf):
        pairdf["q_BH_within_contrast"]=np.nan
        for con in pairdf.contrast.unique():
            ix=pairdf.contrast==con
            pairdf.loc[ix,"q_BH_within_contrast"]=bh_adjust(pairdf.loc[ix,"welch_p"].values)
        pairdf=pairdf.sort_values(["contrast","welch_p"])
    pairdf.to_csv(OUT/"pd_pairwise_contrasts.csv",index=False)

    # Hoehn-Yahr severity within PD.
    sev=[]
    pdx=featdf[featdf.group=="pd"]
    for c in tma:
        z=pdx[["severity",c]].dropna()
        if len(z)<5 or z[c].nunique()<2:continue
        rr=stats.spearmanr(z.severity,z[c])
        sev.append({"feature":c,"n":len(z),"rho":float(rr.statistic),"p":float(rr.pvalue)})
    sevdf=pd.DataFrame(sev)
    if len(sevdf):
        sevdf["q_BH"]=bh_adjust(sevdf.p.values);sevdf=sevdf.sort_values("p")
    sevdf.to_csv(OUT/"pd_hoehn_yahr_correlations.csv",index=False)

    # Incremental PD-v-control test: conventional gait summaries + age + gait speed vs + compact TMA.
    merged=featdf.merge(convdf,on="record",how="left")
    pc=merged[merged.group.isin(["pd","control"])].copy();pc["is_pd"]=(pc.group=="pd").astype(int)
    base=["age","gait_speed","L_stride_mean","L_stride_sd","R_stride_mean","R_stride_sd",
          "L_swing_pct_mean","R_swing_pct_mean","double_support_pct_mean"]
    compact=["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","pivot_depth","tree_span_events",
             "RTO_mean_D","RHS_mean_D","LTO_mean_D"]
    base=[c for c in base if c in pc and pc[c].notna().sum()>=20]
    compact=[c for c in compact if c in pc and np.nanstd(pc[c])>1e-9]
    auc0,n0,p0=loocv_auc(pc,base);auc1,n1,p1=loocv_auc(pc,base+compact)
    inc={"base_features":base,"tma_features":compact,"n_base":n0,"n_plus_tma":n1,"auc_base":auc0,"auc_plus_tma":auc1,"delta_auc":auc1-auc0}
    (OUT/"pd_incremental_classification.json").write_text(json.dumps(inc,indent=2))

    # Multiclass nearest-centroid TMA classification with leave-one-subject-out.
    X=featdf[tma].replace([np.inf,-np.inf],np.nan)
    keep=[c for c in tma if X[c].notna().all() and X[c].std(ddof=1)>1e-9]
    mu=X[keep].mean();sd=X[keep].std(ddof=1);Z=(X[keep]-mu)/sd
    labels=featdf.group.to_numpy();pred=[];correct=[]
    for i in range(len(featdf)):
        tr=np.arange(len(featdf))!=i
        cents={g:Z.loc[tr & (labels==g)].mean().to_numpy() for g in np.unique(labels)}
        xi=Z.iloc[i].to_numpy();ds={g:float(np.sqrt(np.mean((xi-c)**2))) for g,c in cents.items()}
        pr=min(ds,key=ds.get);pred.append(pr);correct.append(pr==labels[i])
    acc=float(np.mean(correct));chance=max(np.mean(labels==g) for g in np.unique(labels))
    prng=np.random.default_rng(SEED+1);null=[]
    for _ in range(2000):
        lab=prng.permutation(labels);ok=0
        for i in range(len(featdf)):
            tr=np.arange(len(featdf))!=i;cents={g:Z.loc[tr & (lab==g)].mean().to_numpy() for g in np.unique(lab)}
            xi=Z.iloc[i].to_numpy();pr=min(cents,key=lambda g:np.sqrt(np.mean((xi-cents[g])**2)));ok+=pr==lab[i]
        null.append(ok/len(lab))
    pacc=(1+np.sum(np.asarray(null)>=acc))/(len(null)+1)
    preddf=pd.DataFrame({"record":featdf.record,"true_group":labels,"predicted_group":pred,"correct":correct})
    preddf.to_csv(OUT/"four_group_predictions.csv",index=False)

    summary={
      "paper_math_audit":audit,
      "n_completed":int(len(featdf)),"n_by_group":featdf.groupby("group").size().to_dict(),
      "failures":failures,
      "event_validation":{"median_stride_abs_diff_s":float(valdf.median_stride_abs_diff.median()),
                          "median_stance_abs_diff_s":float(valdf.median_left_stance_abs_diff.median()),
                          "median_swing_abs_diff_s":float(valdf.median_left_swing_abs_diff.median()),
                          "max_projection_error_ms":float(featdf.projection_error_ms_max.max())},
      "four_group_classification":{"accuracy":acc,"majority_chance":float(chance),"permutation_p":float(pacc),"features":keep},
      "pd_incremental":inc,
      "top_omnibus":omni.head(12).to_dict(orient="records"),
      "top_pd_vs_control":pairdf[pairdf.contrast=="pd-vs-control"].head(12).to_dict(orient="records"),
      "top_pd_vs_hd":pairdf[pairdf.contrast=="pd-vs-hd"].head(8).to_dict(orient="records"),
      "top_pd_vs_als":pairdf[pairdf.contrast=="pd-vs-als"].head(8).to_dict(orient="records"),
      "top_severity":sevdf.head(12).to_dict(orient="records"),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
