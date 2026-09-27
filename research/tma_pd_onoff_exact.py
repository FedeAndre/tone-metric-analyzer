#!/usr/bin/env python3
from __future__ import annotations
import math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests, ezc3d
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold

import tma_gait_gauge_validation as g
import tma_gait_fast_exact as fast

SEED=20260927
PRIMARY_N=24
SENSITIVITY_NS=(16,32)
OUT=Path("research/tma_pd_onoff_results")
OUT.mkdir(parents=True,exist_ok=True)
ZIP=Path("/tmp/C3Dfiles.zip")
INFO=Path("/tmp/PDGinfo.xlsx")
C3D_URL="https://ndownloader.figshare.com/files/28739484"
INFO_URL="https://ndownloader.figshare.com/files/37907001"

CORE_TMA_DIRECTIONS={
    "mean_D":+1,
    "multilevel":+1,
    "mean_lambda":-1,
    "RHS_mean_D":+1,
    "RHS_multilevel":+1,
    "RHS_mean_lambda":-1,
    "tree_span_time":-1,
}

CONV_KEYS=[
    "Speed","Stride_Length_Mean","Cycle_Time_Mean",
    "Double_Limb_Support_Time_Ave","Stride_Width_Mean",
    "Left_Step_Length_Mean","Right_Step_Length_Mean",
]

def download(url,path,min_size=1):
    if path.exists() and path.stat().st_size>=min_size:return
    with requests.get(url,stream=True,timeout=180) as r:
        r.raise_for_status()
        with open(path,"wb") as f:
            for ch in r.iter_content(4*1024*1024):
                if ch:f.write(ch)

def first_between(vals,lo,hi):
    vals=np.asarray(vals,float)
    i=np.searchsorted(vals,lo+1e-9)
    return float(vals[i]) if i<len(vals) and vals[i]<hi-1e-9 else None

def parse_temporal(raw):
    text=raw.decode("utf-8",errors="replace")
    lines=[l for l in text.splitlines() if l.strip()]
    if len(lines)<6:return{}
    names=lines[1].strip().split("\t")
    vals=lines[-1].strip().split("\t")
    if vals and vals[0]=="1":vals=vals[1:]
    out={}
    for k,v in zip(names,vals):
        try:out[k]=float(v)
        except:pass
    return out

def annotated_cycles(path,trial_number):
    c=ezc3d.c3d(str(path))
    rate=float(c["parameters"]["POINT"]["RATE"]["value"][0])
    if "EVENT" not in c["parameters"]:return[]
    ev=c["parameters"]["EVENT"]
    contexts=[str(x).strip().upper() for x in ev["CONTEXTS"]["value"]]
    times=np.asarray(ev["TIMES"]["value"],float)
    if times.ndim!=2 or times.shape[0]<2:return[]
    sec=times[1,:]
    by={k:np.array(sorted([float(t) for t,ctx in zip(sec,contexts) if ctx==k]),float)
        for k in g.EVENT_TYPES}
    lhs=by["LHS"]; out=[]
    for local_index,(a,b) in enumerate(zip(lhs[:-1],lhs[1:])):
        dur=b-a
        if not(0.55<=dur<=2.5):continue
        rto=first_between(by["RTO"],a,b)
        if rto is None:continue
        rhs=first_between(by["RHS"],rto,b)
        if rhs is None:continue
        lto=first_between(by["LTO"],rhs,b)
        if lto is None or not(a<rto<rhs<lto<b):continue
        sa=int(round(a*rate)); sb=int(round(b*rate))
        if sb<=sa:continue
        events=[]
        for label,t in (("LHS",a),("RTO",rto),("RHS",rhs),("LTO",lto)):
            s=int(round(t*rate))
            q,depth,err_ms,raw_phase=g.project_phase(s,sa,sb,rate)
            events.append({
                "label":label,"sample":s,"raw_phase":raw_phase,
                "phase_num":int(q.numerator),"phase_den":int(q.denominator),
                "projection_depth":int(depth),"projection_error_ms":float(err_ms),
            })
        out.append({
            "start_sample":sa,"end_sample":sb,"duration_s":float((sb-sa)/rate),
            "trial":int(trial_number),"trial_cycle":int(local_index),"events":events,
            "actual_phase":{
                "RTO":float((rto-a)/(b-a)),
                "RHS":float((rhs-a)/(b-a)),
                "LTO":float((lto-a)/(b-a)),
            },
        })
    return out

def bh(pvals):
    p=np.asarray(pvals,float)
    out=np.full(len(p),np.nan)
    idx=np.flatnonzero(np.isfinite(p))
    if not len(idx):return out
    order=idx[np.argsort(p[idx])]
    m=len(order); q=np.empty(m,float)
    vals=p[order]*m/np.arange(1,m+1)
    vals=np.minimum.accumulate(vals[::-1])[::-1]
    q=np.minimum(vals,1.0)
    out[order]=q
    return out

def paired_stat(off,on):
    off=np.asarray(off,float);on=np.asarray(on,float)
    mask=np.isfinite(off)&np.isfinite(on)
    off=off[mask];on=on[mask];d=on-off
    if len(d)<3:return {"n":len(d)}
    tt=stats.ttest_rel(on,off)
    try:
        wi=stats.wilcoxon(d,zero_method="wilcox",alternative="two-sided")
        wp=float(wi.pvalue)
    except Exception:wp=np.nan
    return {
        "n":len(d),"off_mean":float(off.mean()),"on_mean":float(on.mean()),
        "change_on_minus_off":float(d.mean()),
        "change_sd":float(d.std(ddof=1)),
        "cohen_dz":float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else np.nan,
        "paired_t":float(tt.statistic),"paired_p":float(tt.pvalue),"wilcoxon_p":wp,
    }

def spearman_perm(x,y,B=20000,seed=SEED):
    x=np.asarray(x,float);y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y);x=x[m];y=y[m]
    if len(x)<5 or np.std(x)==0 or np.std(y)==0:return (np.nan,np.nan,len(x))
    obs=float(stats.spearmanr(x,y).statistic)
    rng=np.random.default_rng(seed)
    ge=0
    for _ in range(B):
        rp=float(stats.spearmanr(x,rng.permutation(y)).statistic)
        if abs(rp)>=abs(obs)-1e-15:ge+=1
    return obs,float((ge+1)/(B+1)),len(x)

def partial_spearman(x,y,C):
    x=np.asarray(x,float);y=np.asarray(y,float);C=np.asarray(C,float)
    m=np.isfinite(x)&np.isfinite(y)&np.isfinite(C).all(axis=1)
    x=x[m];y=y[m];C=C[m]
    if len(x)<7:return (np.nan,np.nan,len(x))
    rx=stats.rankdata(x);ry=stats.rankdata(y)
    RC=np.column_stack([stats.rankdata(C[:,j]) for j in range(C.shape[1])])
    ex=rx-LinearRegression().fit(RC,rx).predict(RC)
    ey=ry-LinearRegression().fit(RC,ry).predict(RC)
    r=stats.pearsonr(ex,ey)
    return float(r.statistic),float(r.pvalue),len(x)

def nested_loocv(X,y,extra=None):
    X=np.asarray(X,float);y=np.asarray(y,float)
    if extra is not None:
        X=np.column_stack([X,np.asarray(extra,float)])
    mask=np.isfinite(y)
    X=X[mask];y=y[mask]
    n=len(y)
    pred=np.full(n,np.nan)
    alphas=np.logspace(-3,3,25)
    for i in range(n):
        tr=np.arange(n)!=i
        inner=KFold(n_splits=min(5,tr.sum()),shuffle=True,random_state=SEED+i)
        model=Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("ridge",RidgeCV(alphas=alphas,cv=inner,scoring="neg_mean_absolute_error")),
        ])
        model.fit(X[tr],y[tr]);pred[i]=model.predict(X[[i]])[0]
    mae=float(np.mean(np.abs(pred-y)))
    rmse=float(np.sqrt(np.mean((pred-y)**2)))
    sst=float(np.sum((y-y.mean())**2)); sse=float(np.sum((y-pred)**2))
    r2=float(1-sse/sst) if sst>0 else np.nan
    rho=float(stats.spearmanr(pred,y).statistic) if np.std(pred)>0 and np.std(y)>0 else np.nan
    return {"n":n,"mae":mae,"rmse":rmse,"cv_r2":r2,"spearman_pred_obs":rho,
            "pred":pred,"obs":y}

def signflip_error_p(err_a,err_b,B=50000):
    d=np.asarray(err_a,float)-np.asarray(err_b,float) # positive => B better
    obs=float(d.mean());rng=np.random.default_rng(SEED+77)
    vals=(rng.choice([-1.0,1.0],size=(B,len(d)))*d).mean(axis=1)
    p=float((1+np.sum(np.abs(vals)>=abs(obs)-1e-15))/(B+1))
    return obs,p

download(C3D_URL,ZIP,1_000_000_000)
download(INFO_URL,INFO,10_000)
info=pd.read_excel(INFO)
info["subject"]=info["ID"].astype(str).str.extract(r"(\d+)").astype(int)

cycles_by={}
conv_trials=[]
qc_errors={}
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    c3ds=[n for n in names if re.search(r"/SUB\d+_(?:on|off)_walk_\d+\.c3d$",n,re.I)]
    def key(n):
        m=re.search(r"SUB(\d+)_(on|off)_walk_(\d+)\.c3d$",n,re.I)
        return(int(m.group(1)),m.group(2).lower(),int(m.group(3)))
    c3ds=sorted(c3ds,key=key)
    tmp=Path("/tmp/walk.c3d")
    for ii,n in enumerate(c3ds,1):
        sid,cond,trial=key(n)
        with z.open(n) as src,open(tmp,"wb") as dst:
            while True:
                b=src.read(1024*1024)
                if not b:break
                dst.write(b)
        cs=annotated_cycles(tmp,trial)
        cycles_by.setdefault((sid,cond),[]).extend(cs)
        tname=n[:-4]+"_temporal_distance.txt"
        ref=parse_temporal(z.read(tname)) if tname in names else{}
        row={"subject":sid,"condition":cond,"trial":trial}
        for k in CONV_KEYS:
            row[k]=ref.get(k,np.nan)
        if np.isfinite(row.get("Left_Step_Length_Mean",np.nan)) and np.isfinite(row.get("Right_Step_Length_Mean",np.nan)):
            row["Step_Length_Asymmetry"]=abs(row["Left_Step_Length_Mean"]-row["Right_Step_Length_Mean"])
        else:row["Step_Length_Asymmetry"]=np.nan
        conv_trials.append(row)
        if cs and "Cycle_Time_Mean" in ref:
            mae=abs(np.mean([c["duration_s"] for c in cs])-ref["Cycle_Time_Mean"])
            qc_errors.setdefault((sid,cond),[]).append(mae)
        if ii%150==0:print("loaded",ii,"/",len(c3ds),flush=True)

convtr=pd.DataFrame(conv_trials)
conv_keys_model=["Speed","Stride_Length_Mean","Cycle_Time_Mean","Double_Limb_Support_Time_Ave","Stride_Width_Mean","Step_Length_Asymmetry"]
convagg=convtr.groupby(["subject","condition"])[conv_keys_model].mean().reset_index()

def analyze_N(N):
    eligible=[]
    for sid in sorted(set(s for s,c in cycles_by)):
        off=cycles_by.get((sid,"off"),[]);on=cycles_by.get((sid,"on"),[])
        qco=np.median(qc_errors.get((sid,"off"),[np.inf]))
        qcn=np.median(qc_errors.get((sid,"on"),[np.inf]))
        if len(off)>=N and len(on)>=N and qco<=0.05 and qcn<=0.05:
            eligible.append(sid)
    rows=[]
    # Exact repository-engine equivalence gate on four real ON/OFF sequences.
    checks=[]
    gate_ids=(eligible[:2]+eligible[-2:]) if len(eligible)>=4 else eligible
    for sid in gate_ids:
        for cond in ("off","on"):
            cy=cycles_by[(sid,cond)][:N]
            fast.assert_equivalent(cy,f"SUB{sid:02d}-{cond}-N{N}")
            checks.append(f"SUB{sid:02d}-{cond}-N{N}")
    for sid in eligible:
        for cond in ("off","on"):
            cy=cycles_by[(sid,cond)][:N]
            feat,*_=fast.fast_analyze_cycles(cy)
            feat["stride_mean_s"]=float(np.mean([c["duration_s"] for c in cy]))
            feat["stride_cv"]=float(np.std([c["duration_s"] for c in cy],ddof=1)/np.mean([c["duration_s"] for c in cy]))
            for lab in ("RTO","RHS","LTO"):
                vals=[c["actual_phase"][lab] for c in cy]
                feat[f"{lab}_phase_mean"]=float(np.mean(vals))
                feat[f"{lab}_phase_sd"]=float(np.std(vals,ddof=1))
            feat.update({"subject":sid,"condition":cond})
            rows.append(feat)
    wide=pd.DataFrame(rows)
    # merge paired feature rows
    off=wide[wide.condition=="off"].set_index("subject")
    on=wide[wide.condition=="on"].set_index("subject")
    numeric=[c for c in wide.columns if c not in("subject","condition") and pd.api.types.is_numeric_dtype(wide[c])]
    ch=pd.DataFrame({"subject":eligible})
    for c in numeric:
        ch[f"{c}_off"]=[off.loc[s,c] for s in eligible]
        ch[f"{c}_on"]=[on.loc[s,c] for s in eligible]
        ch[f"{c}_delta"]=[on.loc[s,c]-off.loc[s,c] for s in eligible]
    # conventional session metrics
    for cond in("off","on"):
        ca=convagg[convagg.condition==cond].set_index("subject")
        for c in conv_keys_model:
            ch[f"conv_{c}_{cond}"]=[ca.loc[s,c] if s in ca.index else np.nan for s in eligible]
    for c in conv_keys_model:
        ch[f"conv_{c}_delta"]=ch[f"conv_{c}_on"]-ch[f"conv_{c}_off"]

    # clinical data and improvement direction: positive means clinical improvement.
    imeta=info.set_index("subject")
    clin_specs={
        "UPDRS_II":("OFF - UPDRS-II","ON - UPDRS-II",-1),
        "UPDRS_II_walking":("OFF - UPDRS-II - walking","ON - UPDRS-II - walking",-1),
        "UPDRS_III":("OFF - UPDRS-III","ON - UPDRS-III",-1),
        "UPDRS_III_walking":("OFF - UPDRS-III - walking","ON - UPDRS-III - walking",-1),
        "miniBEST":("OFF - mini-BESTest","ON - mini-BESTest",+1),
        "FESI":("OFF - FES-I","ON - FES-I",-1),
    }
    for name,(oc,nc,higher_better) in clin_specs.items():
        ovo=[];onv=[];imp=[]
        for s in eligible:
            try:o=float(imeta.loc[s,oc]);nn=float(imeta.loc[s,nc])
            except: o=np.nan;nn=np.nan
            ovo.append(o);onv.append(nn)
            imp.append((nn-o) if higher_better>0 else (o-nn))
        ch[f"clinical_{name}_off"]=ovo;ch[f"clinical_{name}_on"]=onv;ch[f"clinical_{name}_improvement"]=imp

    # A priori TMA normalization score, scaled by OFF between-person SD so zero remains no change.
    components=[]
    for feat,direction in CORE_TMA_DIRECTIONS.items():
        sd=float(np.nanstd(ch[f"{feat}_off"],ddof=1))
        val=direction*ch[f"{feat}_delta"]/(sd if sd>0 else 1.0)
        ch[f"norm_{feat}"]=val
        components.append(ch[f"norm_{feat}"].to_numpy(float))
    ch["tma_normalization_score"]=np.nanmean(np.column_stack(components),axis=1)

    # Core paired stats with FDR.
    stat_rows=[]
    for feat,direction in CORE_TMA_DIRECTIONS.items():
        z=paired_stat(ch[f"{feat}_off"],ch[f"{feat}_on"])
        z.update({"feature":feat,"control_direction":direction,
                  "normalizing_change_mean":float(np.nanmean(direction*ch[f"{feat}_delta"]))})
        stat_rows.append(z)
    statdf=pd.DataFrame(stat_rows)
    statdf["fdr_q"]=bh(statdf["paired_p"].to_numpy(float))

    # Normalization composite.
    norm=ch["tma_normalization_score"].to_numpy(float)
    norm_t=stats.ttest_1samp(norm,0,nan_policy="omit")
    try:norm_w=stats.wilcoxon(norm,zero_method="wilcox").pvalue
    except:norm_w=np.nan
    norm_summary={
        "N_cycles":N,"n_subjects":len(eligible),"eligible_subjects":eligible,
        "equivalence_checks":checks,"equivalence_passed":True,
        "normalization_mean":float(np.nanmean(norm)),
        "normalization_sd":float(np.nanstd(norm,ddof=1)),
        "normalization_t":float(norm_t.statistic),"normalization_p":float(norm_t.pvalue),
        "normalization_wilcoxon_p":float(norm_w),
    }

    # Correlations with clinical improvement and partial correlations controlling conventional change.
    corrrows=[]
    C=np.column_stack([
        ch["conv_Speed_delta"],ch["conv_Stride_Length_Mean_delta"],ch["conv_Cycle_Time_Mean_delta"]
    ])
    for name in clin_specs:
        y=ch[f"clinical_{name}_improvement"].to_numpy(float)
        rho,pp,n=spearman_perm(norm,y,seed=SEED+N+len(corrrows))
        pr,prp,pn=partial_spearman(norm,y,C)
        corrrows.append({"outcome":name,"n":n,"spearman_rho":rho,"permutation_p":pp,
                         "partial_spearman_r":pr,"partial_p":prp,"partial_n":pn})
    corrdf=pd.DataFrame(corrrows)
    corrdf["fdr_q"]=bh(corrdf["permutation_p"].to_numpy(float))
    corrdf["partial_fdr_q"]=bh(corrdf["partial_p"].to_numpy(float))

    # Exploratory incremental prediction: fixed conventional block vs same block + one TMA composite.
    predrows=[]
    X=np.column_stack([ch[f"conv_{c}_delta"].to_numpy(float) for c in conv_keys_model]+[ch["stride_cv_delta"].to_numpy(float)])
    for name in ("UPDRS_II","UPDRS_III","UPDRS_III_walking","FESI"):
        y=ch[f"clinical_{name}_improvement"].to_numpy(float)
        mask=np.isfinite(y)
        a=nested_loocv(X[mask],y[mask])
        b=nested_loocv(X[mask],y[mask],extra=norm[mask])
        obs,p=signflip_error_p(np.abs(a["pred"]-a["obs"]),np.abs(b["pred"]-b["obs"]))
        predrows.append({
            "outcome":name,"n":a["n"],
            "conv_mae":a["mae"],"conv_r2":a["cv_r2"],"conv_rho":a["spearman_pred_obs"],
            "conv_tma_mae":b["mae"],"conv_tma_r2":b["cv_r2"],"conv_tma_rho":b["spearman_pred_obs"],
            "mae_improvement_conv_minus_plusTMA":obs,"signflip_p":p,
        })
    preddf=pd.DataFrame(predrows)

    wide.to_csv(OUT/f"onoff_tma_features_N{N}.csv",index=False)
    ch.to_csv(OUT/f"onoff_paired_changes_N{N}.csv",index=False)
    statdf.to_csv(OUT/f"onoff_tma_paired_stats_N{N}.csv",index=False)
    corrdf.to_csv(OUT/f"onoff_tma_clinical_correlations_N{N}.csv",index=False)
    preddf.to_csv(OUT/f"onoff_incremental_prediction_N{N}.csv",index=False)
    return norm_summary,statdf,corrdf,preddf,ch

summaries={}
for N in (PRIMARY_N,*SENSITIVITY_NS):
    print("ANALYZE N",N,flush=True)
    summaries[N]=analyze_N(N)

# Build final report from primary plus sensitivity.
ps,st,co,pr,ch=summaries[PRIMARY_N]
lines=["# Exact TMA ON/OFF levodopa clinical validation","",
       "Dataset: Shida et al. public ON/OFF overground walking dataset (Figshare 14896881).",
       f"Primary fixed sequence: {PRIMARY_N} complete annotated gait cycles per medication state.",
       f"Primary paired cohort: {ps['n_subjects']} patients.",
       f"Repository-engine equivalence gate: {len(ps['equivalence_checks'])}/{len(ps['equivalence_checks'])} passed.",
       "",
       "## Criterion 2: Does TMA move with medication?",
       f"A-priori TMA normalization composite mean = {ps['normalization_mean']:.4f} OFF-SD units (positive = movement toward the independent control direction).",
       f"One-sample t={ps['normalization_t']:.3f}, p={ps['normalization_p']:.5g}; Wilcoxon p={ps['normalization_wilcoxon_p']:.5g}.",
       "",
       "Core TMA features:"]
for _,r in st.iterrows():
    lines.append(f"- {r.feature}: OFF={r.off_mean:.4f}, ON={r.on_mean:.4f}, ON-OFF={r.change_on_minus_off:+.4f}, dz={r.cohen_dz:+.3f}, p={r.paired_p:.5g}, FDR q={r.fdr_q:.5g}; normalizing-direction change={r.normalizing_change_mean:+.4f}.")
lines += ["","## Does TMA change track clinical improvement?"]
for _,r in co.iterrows():
    lines.append(f"- {r.outcome}: Spearman rho={r.spearman_rho:+.3f}, permutation p={r.permutation_p:.5g}, FDR q={r.fdr_q:.5g}; partial rank r={r.partial_spearman_r:+.3f} controlling Δspeed, Δstride length, Δcycle time (p={r.partial_p:.5g}).")
lines += ["","## Criterion 3: incremental prediction beyond conventional gait change"]
for _,r in pr.iterrows():
    lines.append(f"- {r.outcome}: conventional LOOCV MAE={r.conv_mae:.3f}, +TMA={r.conv_tma_mae:.3f}; MAE improvement={r.mae_improvement_conv_minus_plusTMA:+.3f}, sign-flip p={r.signflip_p:.5g}; CV R² {r.conv_r2:+.3f} -> {r.conv_tma_r2:+.3f}.")
lines += ["","## Sensitivity to sequence length"]
for N,(s,_,c,_,_) in summaries.items():
    lines.append(f"- N={N}: n={s['n_subjects']}, TMA normalization mean={s['normalization_mean']:+.4f}, p={s['normalization_p']:.5g}; UPDRS-III rho={float(c.loc[c.outcome=='UPDRS_III','spearman_rho'].iloc[0]):+.3f}.")
lines += ["","## Interpretation rule",
          "A positive normalization score means ON-medication moved the TMA profile in the direction previously observed in healthy controls: higher D/multilevel occupancy and/or lower lambda/tree span. The direction was fixed before examining this ON/OFF dataset.",
          "",
          "These analyses test medication responsiveness within the same patients. They do not establish diagnostic specificity or clinical utility unless the effect is reproducible and incremental beyond conventional gait measures."]
(OUT/"ONOFF_EXACT_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"ONOFF_EXACT_RESULTS.md").read_text(encoding="utf-8"))
