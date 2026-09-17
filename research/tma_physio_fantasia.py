#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260917
OUT = Path("research/tma_physio_results")
OUT.mkdir(parents=True, exist_ok=True)
YOUNG = [f"f1y{i:02d}" for i in range(1,11)] + [f"f2y{i:02d}" for i in range(1,11)]
OLD = [f"f1o{i:02d}" for i in range(1,11)] + [f"f2o{i:02d}" for i in range(1,11)]
RECORDS = YOUNG + OLD
DEPTHS = (4,5,6)

def expected_trace(n:int, d:int)->float:
    if n <= 0: return 0.0
    return 1.0 + sum((2**l) * (1.0 - (1.0 - 1.0/(2**l))**n) for l in range(1,d+1))

def trace_complexity(phases: np.ndarray, d:int)->int:
    phases=np.asarray(phases,float); phases=phases[np.isfinite(phases)]
    if phases.size==0: return 0
    phases=np.mod(phases,1.0); c=1
    for l in range(1,d+1):
        bins=np.floor(phases*(2**l)).astype(int); bins=np.clip(bins,0,2**l-1)
        c += np.unique(bins).size
    return int(c)

def tma_ratio(phases,d):
    n=len(phases)
    return np.nan if n<=0 else trace_complexity(phases,d)/expected_trace(n,d)

def entropy8(phases):
    phases=np.asarray(phases,float)
    if phases.size==0:return np.nan
    h=np.bincount(np.clip(np.floor(np.mod(phases,1)*8).astype(int),0,7),minlength=8).astype(float)
    p=h/h.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/np.log(8))

def circ_stats(phases):
    phases=np.asarray(phases,float)
    if phases.size==0:return np.nan,np.nan,np.nan
    a=2*np.pi*np.mod(phases,1); s=float(np.mean(np.sin(a))); c=float(np.mean(np.cos(a)))
    return s,c,float(np.hypot(s,c))

def parse_age_sex(comments):
    text=" ".join(comments or []); ma=re.search(r"Age:\s*(\d+)",text,re.I); ms=re.search(r"Sex:\s*([MF])",text,re.I)
    return (int(ma.group(1)) if ma else np.nan, ms.group(1).upper() if ms else "")

def bandpass_resp(x,fs):
    x=np.asarray(x,float)
    if np.any(~np.isfinite(x)):
        idx=np.arange(len(x));good=np.isfinite(x);x=np.interp(idx,idx[good],x[good])
    x=signal.detrend(x,type="linear")
    sos=signal.butter(3,[0.05,0.7],btype="bandpass",fs=fs,output="sos")
    return signal.sosfiltfilt(sos,x)

def respiratory_boundaries(y,fs,polarity=1):
    f,p=signal.welch(y,fs=fs,nperseg=min(len(y),int(fs*120)),noverlap=min(int(fs*60),max(0,min(len(y),int(fs*120))//2)))
    mask=(f>=0.08)&(f<=0.5); fdom=float(f[mask][np.argmax(p[mask])]) if np.any(mask) else 0.2
    period=1.0/max(fdom,0.05); distance=int(fs*max(1.25,0.45*period)); yy=polarity*y
    prom=max(0.18*np.std(yy),0.03*(np.nanpercentile(yy,95)-np.nanpercentile(yy,5)))
    peaks,_=signal.find_peaks(yy,distance=distance,prominence=prom)
    if len(peaks)<10: peaks,_=signal.find_peaks(yy,distance=max(1,int(fs*1.25)),prominence=max(0.1*np.std(yy),1e-9))
    return peaks,fdom

def annotation_samples(rec):
    ann=wfdb.rdann(rec,"ecg",pn_dir="fantasia")
    return np.asarray(ann.sample,dtype=int),np.asarray(ann.symbol,dtype=object)

def cycles_from_boundaries(bounds,r_samples,y,fs):
    rows=[]; phase_lists=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        dur=(b-a)/fs
        if not (1.5<=dur<=12.0):continue
        ia=np.searchsorted(r_samples,a,side="left");ib=np.searchsorted(r_samples,b,side="left");beats=r_samples[ia:ib];n=len(beats)
        if n<2 or n>10:continue
        phi=(beats-a)/(b-a);rr=np.diff(beats)/fs
        if rr.size and (np.any(rr<0.3) or np.any(rr>2.0)):continue
        s,c,R=circ_stats(phi);amp=float(np.nanmax(y[a:b])-np.nanmin(y[a:b]))
        row={"duration_s":float(dur),"n_beats":int(n),"mean_rr_s":float(np.mean(rr)) if rr.size else np.nan,
             "sd_rr_s":float(np.std(rr,ddof=1)) if rr.size>1 else 0.0,"cv_rr":float(np.std(rr,ddof=1)/np.mean(rr)) if rr.size>1 and np.mean(rr)>0 else 0.0,
             "phase_sin":s,"phase_cos":c,"phase_R":R,"phase_entropy8":entropy8(phi),"resp_amp":amp}
        for d in DEPTHS:
            tr=trace_complexity(phi,d);row[f"trace_d{d}"]=tr;row[f"tma_ratio_d{d}"]=tr/expected_trace(n,d)
        rows.append(row);phase_lists.append(np.asarray(phi,float))
    return pd.DataFrame(rows),phase_lists

def global_rr_features(r_samples,fs):
    rr=np.diff(r_samples)/fs;rr=rr[(rr>=0.3)&(rr<=2.0)]
    if len(rr)<3:return dict(mean_rr_ms=np.nan,sdnn_ms=np.nan,rmssd_ms=np.nan)
    return dict(mean_rr_ms=float(np.mean(rr)*1000),sdnn_ms=float(np.std(rr,ddof=1)*1000),rmssd_ms=float(np.sqrt(np.mean(np.diff(rr)**2))*1000))

def subject_features(rec,orientation):
    hdr=wfdb.rdheader(rec,pn_dir="fantasia");age,sex=parse_age_sex(hdr.comments)
    resp_i=[i for i,n in enumerate(hdr.sig_name) if str(n).upper()=="RESP"][0]
    rr=wfdb.rdrecord(rec,pn_dir="fantasia",channels=[resp_i]);x=np.asarray(rr.p_signal[:,0],float);fs=float(rr.fs);y=bandpass_resp(x,fs)
    r_samples,symbols=annotation_samples(rec);bounds,fdom=respiratory_boundaries(y,fs,1 if orientation=="peak" else -1)
    cyc,phase_lists=cycles_from_boundaries(bounds,r_samples,y,fs)
    if len(cyc)<50:raise RuntimeError(f"{rec}: only {len(cyc)} valid respiratory cycles")
    all_phi=np.concatenate(phase_lists);psin,pcos,pR=circ_stats(all_phi);g=global_rr_features(r_samples,fs)
    feat={"record":rec,"group":"young" if "y" in rec else "old","age":age,"sex":sex,"fs":fs,"n_samples":len(x),"n_annotations":len(r_samples),
          "normal_annotation_fraction":float(np.mean(symbols=="N")) if len(symbols) else np.nan,"n_resp_boundaries":len(bounds),"n_cycles":len(cyc),
          "dominant_resp_hz":float(fdom),**g,"resp_period_mean_s":float(cyc.duration_s.mean()),"resp_period_sd_s":float(cyc.duration_s.std(ddof=1)),
          "beats_per_breath_mean":float(cyc.n_beats.mean()),"beats_per_breath_sd":float(cyc.n_beats.std(ddof=1)),"phase_sin":psin,"phase_cos":pcos,"phase_R":pR,
          "phase_entropy8":entropy8(all_phi),"resp_amp_mean":float(cyc.resp_amp.mean())}
    for d in DEPTHS:
        feat[f"tma_ratio_mean_d{d}"]=float(cyc[f"tma_ratio_d{d}"].mean());feat[f"tma_ratio_sd_d{d}"]=float(cyc[f"tma_ratio_d{d}"].std(ddof=1));feat[f"trace_mean_d{d}"]=float(cyc[f"trace_d{d}"].mean())
    return feat,cyc,phase_lists

BASE_COLS=["mean_rr_ms","sdnn_ms","rmssd_ms","resp_period_mean_s","resp_period_sd_s","beats_per_breath_mean","beats_per_breath_sd","phase_sin","phase_cos","phase_R","phase_entropy8"]

def make_classifier():
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("model",LogisticRegression(max_iter=5000,solver="liblinear",penalty="l2"))])

def loo_classification(df,cols):
    X=df[cols].to_numpy(float);y=(df.group=="old").astype(int).to_numpy();pred=np.zeros(len(df));chosen=[];Cs=[0.01,0.1,1,10,100]
    for tr,te in LeaveOneOut().split(X):
        inner=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED);gs=GridSearchCV(make_classifier(),{"model__C":Cs},cv=inner,scoring="neg_log_loss",n_jobs=1)
        gs.fit(X[tr],y[tr]);pred[te]=gs.predict_proba(X[te])[:,1];chosen.append(gs.best_params_["model__C"])
    return pred,chosen

def class_metrics(y,p):
    return {"log_loss":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p)),"accuracy":float(accuracy_score(y,p>=0.5)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=0.5))}

def bootstrap_delta(y,p0,p1,B=5000):
    y=np.asarray(y);p0=np.asarray(p0);p1=np.asarray(p1);ids0=np.flatnonzero(y==0);ids1=np.flatnonzero(y==1);vals=[];rng=np.random.default_rng(SEED+1)
    for _ in range(B):
        idx=np.r_[rng.choice(ids0,len(ids0),replace=True),rng.choice(ids1,len(ids1),replace=True)];vals.append(log_loss(y[idx],p1[idx],labels=[0,1])-log_loss(y[idx],p0[idx],labels=[0,1]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],float(np.mean(vals))

def make_regressor():return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("model",Ridge())])

def loo_age(df,cols):
    X=df[cols].to_numpy(float);y=df.age.to_numpy(float);pred=np.zeros(len(df));alphas=[0.01,0.1,1,10,100]
    for tr,te in LeaveOneOut().split(X):
        inner=KFold(n_splits=5,shuffle=True,random_state=SEED);gs=GridSearchCV(make_regressor(),{"model__alpha":alphas},cv=inner,scoring="neg_mean_squared_error",n_jobs=1)
        gs.fit(X[tr],y[tr]);pred[te]=gs.predict(X[te])
    return pred

def age_metrics(y,p):return {"rmse":float(mean_squared_error(y,p)**0.5),"mae":float(mean_absolute_error(y,p)),"r2":float(r2_score(y,p))}

def perm_group_p(x,y,B=20000):
    x=np.asarray(x,float);y=np.asarray(y,int);obs=float(np.mean(x[y==1])-np.mean(x[y==0]));rng=np.random.default_rng(SEED+2);vals=np.empty(B)
    for b in range(B):
        yp=rng.permutation(y);vals[b]=np.mean(x[yp==1])-np.mean(x[yp==0])
    return obs,float((1+np.sum(np.abs(vals)>=abs(obs)))/(B+1))

def cohend(x0,x1):
    x0=np.asarray(x0,float);x1=np.asarray(x1,float);sp=math.sqrt(((len(x0)-1)*np.var(x0,ddof=1)+(len(x1)-1)*np.var(x1,ddof=1))/(len(x0)+len(x1)-2))
    return float((np.mean(x1)-np.mean(x0))/sp) if sp>0 else np.nan

def surrogate_phase_rotation(subject_phases,groups,depth=5,B=500):
    rng=np.random.default_rng(SEED+3);recs=list(subject_phases);y=np.array([1 if groups[r]=="old" else 0 for r in recs]);obs=np.array([np.mean([tma_ratio(ph,depth) for ph in subject_phases[r]]) for r in recs]);d_obs=cohend(obs[y==0],obs[y==1]);ds=[]
    for _ in range(B):
        vals=[]
        for r in recs:
            rs=[tma_ratio(np.mod(ph+rng.random(),1.0),depth) for ph in subject_phases[r]];vals.append(np.mean(rs))
        vals=np.asarray(vals);ds.append(cohend(vals[y==0],vals[y==1]))
    ds=np.asarray(ds,float);p=float((1+np.sum(np.abs(ds)>=abs(d_obs)))/(B+1))
    return {"observed_cohens_d_old_minus_young":float(d_obs),"surrogate_abs_tail_p":p,"surrogate_d_mean":float(np.nanmean(ds)),"surrogate_d_sd":float(np.nanstd(ds,ddof=1)),"surrogate_d_q025":float(np.nanquantile(ds,.025)),"surrogate_d_q975":float(np.nanquantile(ds,.975))}

def run_orientation(orientation):
    feats=[];cycles={};phases={}
    for j,rec in enumerate(RECORDS,1):
        print(f"[{orientation}] {j:02d}/40 {rec}",flush=True);f,c,p=subject_features(rec,orientation);feats.append(f);cycles[rec]=c;phases[rec]=p
    df=pd.DataFrame(feats).sort_values("record").reset_index(drop=True);df.to_csv(OUT/f"subject_features_{orientation}.csv",index=False)
    pd.DataFrame([{"record":r,"n_cycles":len(cycles[r]),"mean_duration":cycles[r].duration_s.mean(),"mean_n_beats":cycles[r].n_beats.mean(),**{f"mean_tma_d{d}":cycles[r][f"tma_ratio_d{d}"].mean() for d in DEPTHS}} for r in sorted(cycles)]).to_csv(OUT/f"cycle_summary_{orientation}.csv",index=False)
    y=(df.group=="old").astype(int).to_numpy();result={"orientation":orientation,"n_subjects":len(df),"n_cycles_total":int(df.n_cycles.sum()),"group_counts":df.group.value_counts().to_dict(),"depths":{}}
    p0,_=loo_classification(df,BASE_COLS);m0=class_metrics(y,p0);a0=loo_age(df,BASE_COLS);am0=age_metrics(df.age.to_numpy(),a0);result["baseline_classification"]=m0;result["baseline_age_regression"]=am0
    predout=pd.DataFrame({"record":df.record,"group":df.group,"age":df.age,"p_old_baseline":p0,"agehat_baseline":a0})
    for d in DEPTHS:
        tcols=[f"tma_ratio_mean_d{d}",f"tma_ratio_sd_d{d}"];p1,_=loo_classification(df,BASE_COLS+tcols);m1=class_metrics(y,p1);ci,bootmean=bootstrap_delta(y,p0,p1);a1=loo_age(df,BASE_COLS+tcols);am1=age_metrics(df.age.to_numpy(),a1);x=df[f"tma_ratio_mean_d{d}"].to_numpy();diff,pperm=perm_group_p(x,y);dcohen=cohend(x[y==0],x[y==1])
        result["depths"][str(d)]={"classification_extended":m1,"delta_logloss_extended_minus_baseline":float(m1["log_loss"]-m0["log_loss"]),"delta_auc_extended_minus_baseline":float(m1["auc"]-m0["auc"]),"bootstrap_delta_logloss_95ci":ci,"bootstrap_delta_logloss_mean":bootmean,"age_regression_extended":am1,"delta_age_rmse_extended_minus_baseline":float(am1["rmse"]-am0["rmse"]),"tma_mean_young":float(x[y==0].mean()),"tma_mean_old":float(x[y==1].mean()),"tma_group_diff_old_minus_young":diff,"tma_group_permutation_p":pperm,"tma_group_cohens_d":dcohen}
        predout[f"p_old_tma_d{d}"]=p1;predout[f"agehat_tma_d{d}"]=a1
    predout.to_csv(OUT/f"predictions_{orientation}.csv",index=False)
    if orientation=="peak":result["phase_rotation_surrogate_d5"]=surrogate_phase_rotation(phases,{r:("young" if "y" in r else "old") for r in phases},depth=5,B=500)
    return df,result

def main():
    t=time.time();allres={"seed":SEED,"dataset":"PhysioNet Fantasia v1.0.0","hypothesis":"Respiration-anchored recursive topology of heartbeats adds information beyond conventional HRV, respiratory, and phase-coupling features."}
    _,peak_res=run_orientation("peak");_,trough_res=run_orientation("trough");allres["peak"]=peak_res;allres["trough"]=trough_res;allres["runtime_s"]=time.time()-t
    with open(OUT/"summary.json","w") as f:json.dump(allres,f,indent=2)
    lines=["# TMA cardiorespiratory analysis — Fantasia","",f"Subjects: {peak_res['n_subjects']} (20 young, 20 old).",f"Valid peak-to-peak respiratory cycles: {peak_res['n_cycles_total']}.",f"Valid trough-to-trough respiratory cycles: {trough_res['n_cycles_total']}.","","## Peak-anchored primary analysis",f"Baseline log loss: {peak_res['baseline_classification']['log_loss']:.6f}; AUC: {peak_res['baseline_classification']['auc']:.4f}.",""]
    for d in DEPTHS:
        z=peak_res["depths"][str(d)];lines += [f"- Depth {d}: extended log loss {z['classification_extended']['log_loss']:.6f}; delta {z['delta_logloss_extended_minus_baseline']:+.6f}; AUC {z['classification_extended']['auc']:.4f}; bootstrap 95% CI for delta log loss {z['bootstrap_delta_logloss_95ci']}; TMA group difference p={z['tma_group_permutation_p']:.5g}, Cohen d={z['tma_group_cohens_d']:+.3f}."]
    s=peak_res["phase_rotation_surrogate_d5"];lines += ["",f"Depth-5 phase-rotation surrogate: observed Cohen d={s['observed_cohens_d_old_minus_young']:+.3f}, surrogate tail p={s['surrogate_abs_tail_p']:.5g}, surrogate 95% range [{s['surrogate_d_q025']:+.3f}, {s['surrogate_d_q975']:+.3f}].","","## Trough-anchored sensitivity",f"Baseline log loss: {trough_res['baseline_classification']['log_loss']:.6f}; AUC: {trough_res['baseline_classification']['auc']:.4f}."]
    for d in DEPTHS:
        z=trough_res["depths"][str(d)];lines += [f"- Depth {d}: delta log loss {z['delta_logloss_extended_minus_baseline']:+.6f}; delta AUC {z['delta_auc_extended_minus_baseline']:+.4f}; TMA group p={z['tma_group_permutation_p']:.5g}, d={z['tma_group_cohens_d']:+.3f}."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n");print("\n".join(lines),flush=True)

if __name__=="__main__":main()
