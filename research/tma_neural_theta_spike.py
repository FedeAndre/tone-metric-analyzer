#!/usr/bin/env python3
from __future__ import annotations

import json, math, os, subprocess, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260917
DEPTHS = (4,5,6)
WINDOW_S = 10.0
POP_BIN_S = 0.001
OUT = Path("research/tma_neural_results")
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [
("M01","M01_20240312","https://dandiarchive.s3.amazonaws.com/blobs/b1d/d65/b1dd65cb-80a8-4829-a7ce-8645ec2269a9"),
("M01","M01_20240313","https://dandiarchive.s3.amazonaws.com/blobs/686/eb4/686eb43e-7bd2-4bde-adcd-0de5a76b7702"),
("M01","M01_20240314","https://dandiarchive.s3.amazonaws.com/blobs/74b/015/74b01582-786c-4e36-b1a4-86dd4987f5df"),
("M01","M01_20240318","https://dandiarchive.s3.amazonaws.com/blobs/3df/0d2/3df0d257-d382-4466-bb90-938765c0774f"),
("M02","M02_20240312","https://dandiarchive.s3.amazonaws.com/blobs/e42/58a/e4258ab4-a9cb-45ff-b92e-6aec93d44aca"),
("M02","M02_20240313","https://dandiarchive.s3.amazonaws.com/blobs/539/100/539100fc-2aeb-43eb-a0ba-209ad70ed4f3"),
("M02","M02_20240314","https://dandiarchive.s3.amazonaws.com/blobs/97c/508/97c508d4-5c3f-44b3-b551-5483f7b685b2"),
("M02","M02_20240318","https://dandiarchive.s3.amazonaws.com/blobs/2db/2f0/2db2f006-a622-44a4-9043-2921e146ef48"),
("M03","M03_20240621","https://dandiarchive.s3.amazonaws.com/blobs/f99/2af/f992af02-e8ce-4e97-96be-c64a5c55bf0f"),
("M03","M03_20240622","https://dandiarchive.s3.amazonaws.com/blobs/ab7/0ad/ab70adaf-af98-4fbb-8429-a8eb66c07e2e"),
("M03","M03_20240623","https://dandiarchive.s3.amazonaws.com/blobs/516/f25/516f25d9-ed23-4b59-9ee8-0a841016f080"),
("M03","M03_20240624","https://dandiarchive.s3.amazonaws.com/blobs/02c/8e9/02c8e995-d570-42c6-a1ed-c825581348b4"),
("M05","M05_20240729","https://dandiarchive.s3.amazonaws.com/blobs/ad2/92c/ad292cc1-44be-48ed-abcf-5d62fd8954ac"),
("M05","M05_20240730","https://dandiarchive.s3.amazonaws.com/blobs/616/bcb/616bcb92-b31f-4b4a-864d-8910e8205e99"),
("M05","M05_20240731","https://dandiarchive.s3.amazonaws.com/blobs/f00/380/f0038008-5e87-4517-9418-9f927d6dfab7"),
]

BASE_COLS = [
    "theta_power","theta_frequency","theta_amp_cv",
    "raw_firing_rate","population_event_rate","active_units",
    "events_per_cycle_mean","events_per_cycle_sd","frac_cycles_ge2",
    "phase_sin","phase_cos","phase_R","phase_entropy8","population_isi_cv",
]

RICH_COLS = BASE_COLS + [
    "theta_cycle_duration_sd",
    "events_per_cycle_q25","events_per_cycle_median","events_per_cycle_q75",
    "cycle_phase_R_mean","cycle_phase_R_sd",
    "cycle_phase_entropy8_mean","cycle_phase_entropy8_sd",
    "cycle_pairdist_mean","cycle_pairdist_sd",
    "cycle_gap_cv_mean","cycle_gap_cv_sd",
    "cycle_occ32_mean","cycle_occ32_sd",
]

def dec(v):
    if isinstance(v, bytes): return v.decode("utf-8","replace")
    return str(v)

def expected_trace(n:int,d:int)->float:
    if n<=0: return np.nan
    return 1.0 + sum((2**l)*(1-(1-1/(2**l))**n) for l in range(1,d+1))

def trace_complexity(ph,d):
    ph=np.asarray(ph,float)
    ph=ph[np.isfinite(ph)]
    if len(ph)==0:return 0
    ph=np.mod(ph,1.0); c=1
    for l in range(1,d+1):
        bins=np.floor(ph*(2**l)).astype(np.int16)
        bins=np.clip(bins,0,2**l-1)
        c += np.unique(bins).size
    return int(c)

def tma_ratio(ph,d):
    n=len(ph)
    return np.nan if n<2 else trace_complexity(ph,d)/expected_trace(n,d)

def rotated_tma_batch(ph, shifts, depth=5):
    ph=np.asarray(ph,float); shifts=np.asarray(shifts,float)
    if len(ph)<2:return np.full(len(shifts),np.nan)
    q=np.mod(ph[None,:]+shifts[:,None],1.0)
    tr=np.ones(len(shifts),float)
    for l in range(1,depth+1):
        bins=np.floor(q*(2**l)).astype(np.int16)
        bins.sort(axis=1)
        tr += 1 + np.sum(np.diff(bins,axis=1)!=0,axis=1)
    return tr/expected_trace(len(ph),depth)

def uniform_tma_batch(n, B, depth, rng):
    if n < 2:
        return np.full(B, np.nan)
    q=rng.random((B,n))
    tr=np.ones(B,float)
    for l in range(1,depth+1):
        bins=np.floor(q*(2**l)).astype(np.int16)
        bins.sort(axis=1)
        tr += 1 + np.sum(np.diff(bins,axis=1)!=0,axis=1)
    return tr/expected_trace(n,depth)

def self_test():
    rng=np.random.default_rng(123)
    for n in range(2,15):
        ph=np.sort(rng.random(n)); sh=rng.random(30)
        a=np.array([tma_ratio(np.mod(ph+s,1),5) for s in sh])
        b=rotated_tma_batch(ph,sh,5)
        if not np.allclose(a,b,atol=1e-12,rtol=0):
            raise AssertionError("rotated TMA vectorization mismatch")

def entropy8(ph):
    ph=np.asarray(ph,float)
    if len(ph)==0:return np.nan
    h=np.bincount(np.clip(np.floor(np.mod(ph,1)*8).astype(int),0,7),minlength=8).astype(float)
    p=h/h.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/np.log(8))

def circ_stats(ph):
    ph=np.asarray(ph,float)
    if len(ph)==0:return np.nan,np.nan,np.nan
    a=2*np.pi*np.mod(ph,1)
    s=float(np.mean(np.sin(a))); c=float(np.mean(np.cos(a)))
    return s,c,float(np.hypot(s,c))

def cycle_geometry_features(ph):
    ph=np.mod(np.asarray(ph,float),1.0)
    n=len(ph)
    if n<2:
        return (np.nan,np.nan,np.nan,np.nan)
    _,_,R=circ_stats(ph)
    ent=entropy8(ph)
    d=np.abs(ph[:,None]-ph[None,:])
    d=np.minimum(d,1.0-d)
    iu=np.triu_indices(n,1)
    pair=float(np.mean(d[iu])) if len(iu[0]) else np.nan
    s=np.sort(ph)
    gaps=np.diff(np.r_[s,s[0]+1.0])
    gapcv=float(np.std(gaps,ddof=1)/np.mean(gaps)) if n>2 and np.mean(gaps)>0 else 0.0
    occ=float(len(np.unique(np.floor(ph*32).astype(int)))/n)
    return R,ent,pair,gapcv,occ

def summarize_cycle_geometry(phase_lists):
    vals=[cycle_geometry_features(ph) for ph in phase_lists if len(ph)>=2]
    if not vals:
        return {k:np.nan for k in [
            "cycle_phase_R_mean","cycle_phase_R_sd","cycle_phase_entropy8_mean","cycle_phase_entropy8_sd",
            "cycle_pairdist_mean","cycle_pairdist_sd","cycle_gap_cv_mean","cycle_gap_cv_sd",
            "cycle_occ32_mean","cycle_occ32_sd"]}
    a=np.asarray(vals,float)
    names=["cycle_phase_R","cycle_phase_entropy8","cycle_pairdist","cycle_gap_cv","cycle_occ32"]
    out={}
    for j,nm in enumerate(names):
        z=a[:,j];z=z[np.isfinite(z)]
        out[nm+"_mean"]=float(np.mean(z)) if len(z) else np.nan
        out[nm+"_sd"]=float(np.std(z,ddof=1)) if len(z)>1 else 0.0
    return out

def download(url,path):
    for attempt in range(3):
        try:
            subprocess.run(["wget","-q","--show-progress","-O",str(path),url],check=True)
            return
        except subprocess.CalledProcessError:
            if attempt==2: raise
            time.sleep(2)

def choose_lfp_group(f):
    root=f["processing/ecephys/LFP"]
    names=list(root.keys())
    ca1=[n for n in names if "CA1" in n.upper()]
    name=ca1[0] if ca1 else names[0]
    return root[name], name

def read_ca1_pyramidal_spikes(f):
    u=f["units"]
    areas=np.array([dec(x) for x in u["cell_area"][:]],dtype=object)
    types=np.array([dec(x) for x in u["cell_type"][:]],dtype=object)
    keep=np.array([("CA1" in a.upper()) and ("PYRAM" in t.upper()) for a,t in zip(areas,types)])
    ids=np.flatnonzero(keep)
    st=u["spike_times"]
    ends=u["spike_times_index"][:].astype(np.int64)
    starts=np.r_[0,ends[:-1]]
    times=[]; unit_ids=[]
    for uid in ids:
        x=np.asarray(st[starts[uid]:ends[uid]],float)
        if len(x):
            times.append(x)
            unit_ids.append(np.full(len(x),uid,dtype=np.int32))
    if not times:
        return np.array([]),np.array([],dtype=np.int32),0
    t=np.concatenate(times); q=np.concatenate(unit_ids)
    order=np.argsort(t)
    return t[order],q[order],len(ids)

def collapse_population_events(raw_t):
    if len(raw_t)==0:return raw_t
    bins=np.floor(raw_t/POP_BIN_S).astype(np.int64)
    _,idx=np.unique(bins,return_index=True)
    return raw_t[idx]

def phase_map_and_cycles(pop_t, peak_t):
    if len(peak_t)<2:
        return None
    dur=np.diff(peak_t)
    valid_cycle=(dur>=0.08)&(dur<=0.22)
    cycle_mid=(peak_t[:-1]+peak_t[1:])/2
    idx=np.searchsorted(peak_t,pop_t,side="right")-1
    ok=(idx>=0)&(idx<len(dur))
    idx0=idx[ok]; evt=pop_t[ok]
    ok2=valid_cycle[idx0]
    idx0=idx0[ok2]; evt=evt[ok2]
    phase=(evt-peak_t[idx0])/dur[idx0]
    order=np.argsort(idx0)
    idx0=idx0[order]; evt=evt[order]; phase=phase[order]
    counts=np.bincount(idx0,minlength=len(dur))
    tmas={d:np.full(len(dur),np.nan) for d in DEPTHS}
    phase_map={}
    if len(idx0):
        uniq,starts=np.unique(idx0,return_index=True)
        ends=np.r_[starts[1:],len(idx0)]
        for ci,a,b in zip(uniq,starts,ends):
            ph=phase[a:b]
            if len(ph)>=2:
                phase_map[int(ci)]=ph.astype(np.float32)
                for d in DEPTHS:tmas[d][ci]=tma_ratio(ph,d)
    return dict(
        duration=dur,valid=valid_cycle,mid=cycle_mid,counts=counts,
        event_time=evt,event_phase=phase,event_cycle=idx0,
        tmas=tmas,phase_map=phase_map
    )

def process_session(subject,session,url,tmp):
    print(f"[session] {session} download",flush=True)
    download(url,tmp)
    rows=[]; candidate_cycles=[]
    try:
        with h5py.File(tmp,"r") as f:
            lfp_g,lfp_name=choose_lfp_group(f)
            ds=lfp_g["data"]
            rate=float(lfp_g["starting_time"].attrs["rate"])
            lfp_start=float(lfp_g["starting_time"][()])
            conv=float(ds.attrs.get("conversion",1.0))
            speed=np.asarray(f["processing/behavior/Speed/data"][:],float)
            speed_t=np.asarray(f["processing/behavior/Speed/timestamps"][:],float)
            good=np.isfinite(speed)&np.isfinite(speed_t)
            speed=speed[good];speed_t=speed_t[good]
            if len(speed_t)<100: raise RuntimeError("insufficient speed samples")
            raw_t,raw_uid,n_units=read_ca1_pyramidal_spikes(f)
            if n_units<2: raise RuntimeError("fewer than 2 CA1 pyramidal units")
            pop_t=collapse_population_events(raw_t)
            t0=max(float(speed_t[0]),lfp_start+2)
            t1=min(float(speed_t[-1]),lfp_start+(ds.shape[0]-1)/rate-2)
            i0=max(0,int((t0-lfp_start)*rate))
            i1=min(ds.shape[0],int((t1-lfp_start)*rate))
            if i1-i0 < rate*60: raise RuntimeError("less than 60s overlapping LFP/behavior")
            if ds.ndim==2: raw=np.asarray(ds[i0:i1,0],dtype=np.float32)
            else: raw=np.asarray(ds[i0:i1],dtype=np.float32)
            raw*=conv
            seg_start=lfp_start+i0/rate
            q=max(1,int(round(rate/250.0)))
            lfp=signal.resample_poly(raw,1,q).astype(np.float32)
            fs=rate/q
            sos=signal.butter(4,[6,10],btype="bandpass",fs=fs,output="sos")
            theta=signal.sosfiltfilt(sos,lfp).astype(np.float32)
            amp=np.abs(signal.hilbert(theta)).astype(np.float32)
            peaks,_=signal.find_peaks(theta,distance=max(1,int(fs/12.0)),prominence=max(float(np.std(theta))*0.03,1e-12))
            peak_t=seg_start+peaks/fs
            cm=phase_map_and_cycles(pop_t,peak_t)
            if cm is None: raise RuntimeError("no theta cycles")
            # event phases aligned to valid theta cycles
            evt_t=cm["event_time"]; evt_ph=cm["event_phase"]
            valid_cycle_idx=np.flatnonzero(cm["valid"])
            # non-overlapping 10s windows
            start=math.ceil(t0/WINDOW_S)*WINDOW_S
            stop=t1-WINDOW_S
            wi=0
            while start+wi*WINDOW_S<=stop:
                a=start+wi*WINDOW_S;b=a+WINDOW_S
                sm=(speed_t>=a)&(speed_t<b)
                if np.sum(sm)<5:
                    wi+=1;continue
                y=float(np.nanmean(speed[sm]))
                cyc=np.flatnonzero(cm["valid"]&(cm["mid"]>=a)&(cm["mid"]<b))
                if len(cyc)<30:
                    wi+=1;continue
                # TMA requires enough cycles with >=2 population events
                finite5=cyc[np.isfinite(cm["tmas"][5][cyc])]
                if len(finite5)<5:
                    wi+=1;continue
                # raw spikes
                ra=np.searchsorted(raw_t,a,"left"); rb=np.searchsorted(raw_t,b,"left")
                rt=raw_t[ra:rb]; ru=raw_uid[ra:rb]
                pa=np.searchsorted(pop_t,a,"left"); pb=np.searchsorted(pop_t,b,"left")
                pt=pop_t[pa:pb]
                if len(pt)<20:
                    wi+=1;continue
                ea=np.searchsorted(evt_t,a,"left"); eb=np.searchsorted(evt_t,b,"left")
                ph=evt_ph[ea:eb]
                ss,cc,RR=circ_stats(ph)
                isi=np.diff(pt)
                isi_cv=float(np.std(isi,ddof=1)/np.mean(isi)) if len(isi)>2 and np.mean(isi)>0 else np.nan
                # theta segment on resampled timeline
                la=max(0,int((a-seg_start)*fs)); lb=min(len(theta),int((b-seg_start)*fs))
                th=theta[la:lb]; am=amp[la:lb]
                if len(th)<fs*5:
                    wi+=1;continue
                cycle_counts=cm["counts"][cyc].astype(float)
                cycle_phases=[cm["phase_map"][int(ci)] for ci in cyc if int(ci) in cm["phase_map"]]
                geom=summarize_cycle_geometry(cycle_phases)
                row=dict(
                    row_id=f"{session}:{wi}",subject=subject,session=session,
                    speed_mean=y,
                    theta_power=float(np.mean(th.astype(float)**2)),
                    theta_frequency=float(1/np.mean(cm["duration"][cyc])),
                    theta_amp_cv=float(np.std(am,ddof=1)/np.mean(am)) if np.mean(am)>0 else np.nan,
                    theta_cycle_duration_sd=float(np.std(cm["duration"][cyc],ddof=1)),
                    raw_firing_rate=float(len(rt)/WINDOW_S),
                    population_event_rate=float(len(pt)/WINDOW_S),
                    active_units=int(len(np.unique(ru))),
                    events_per_cycle_mean=float(np.mean(cycle_counts)),
                    events_per_cycle_sd=float(np.std(cycle_counts,ddof=1)),
                    events_per_cycle_q25=float(np.quantile(cycle_counts,.25)),
                    events_per_cycle_median=float(np.median(cycle_counts)),
                    events_per_cycle_q75=float(np.quantile(cycle_counts,.75)),
                    frac_cycles_ge2=float(np.mean(cycle_counts>=2)),
                    phase_sin=ss,phase_cos=cc,phase_R=RR,
                    phase_entropy8=entropy8(ph),
                    population_isi_cv=isi_cv,
                    **geom,
                    n_ca1_pyramidal=n_units,n_theta_cycles=len(cyc),n_phase_events=len(ph),
                    lfp_source=lfp_name
                )
                for d in DEPTHS:
                    z=cm["tmas"][d][cyc];z=z[np.isfinite(z)]
                    row[f"tma_mean_d{d}"]=float(np.mean(z)) if len(z) else np.nan
                    row[f"tma_sd_d{d}"]=float(np.std(z,ddof=1)) if len(z)>1 else 0.0
                rows.append(row)
                candidate_cycles.append(cyc.copy())
                wi+=1

            # choose at most 20 surrogate windows/session, stratified over speed rank
            selected=[]
            if rows:
                order=np.argsort([r["speed_mean"] for r in rows])
                rng=np.random.default_rng(SEED+sum(map(ord,session)))
                for block in np.array_split(order,4):
                    if len(block):
                        k=min(5,len(block))
                        selected.extend(rng.choice(block,size=k,replace=False).tolist())
            surrogate=[]
            for j in selected:
                phases=[cm["phase_map"][int(ci)] for ci in candidate_cycles[j] if int(ci) in cm["phase_map"]]
                if len(phases)>=5:
                    surrogate.append((rows[j]["row_id"],subject,phases))
            print(f"[session] {session}: windows={len(rows)} surrogate_windows={len(surrogate)} units={n_units}",flush=True)
            return rows,surrogate
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass

def reg_pipeline():
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("model",Ridge())])

def clf_pipeline():
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("model",LogisticRegression(max_iter=5000,solver="liblinear"))])

def nested_group_regression(df,cols):
    X=df[cols].to_numpy(float);y=df.speed_mean.to_numpy(float);g=df.subject.to_numpy()
    pred=np.full(len(df),np.nan)
    for held in np.unique(g):
        te=g==held;tr=~te;gt=g[tr]
        inner=GroupKFold(n_splits=len(np.unique(gt)))
        gs=GridSearchCV(reg_pipeline(),{"model__alpha":[0.01,0.1,1,10,100]},cv=inner,scoring="neg_mean_squared_error",n_jobs=-1)
        gs.fit(X[tr],y[tr],groups=gt)
        pred[te]=gs.predict(X[te])
    return pred

def nested_group_classifier(df,cols):
    use=(df.speed_mean<=2.0)|(df.speed_mean>=10.0)
    d=df.loc[use].reset_index()
    X=d[cols].to_numpy(float);y=(d.speed_mean>=10.0).astype(int).to_numpy();g=d.subject.to_numpy()
    pred=np.full(len(d),np.nan)
    for held in np.unique(g):
        te=g==held;tr=~te;gt=g[tr]
        inner=GroupKFold(n_splits=len(np.unique(gt)))
        gs=GridSearchCV(clf_pipeline(),{"model__C":[0.01,0.1,1,10,100]},cv=inner,scoring="neg_log_loss",n_jobs=-1)
        gs.fit(X[tr],y[tr],groups=gt)
        pred[te]=gs.predict_proba(X[te])[:,1]
    return d,pred

def reg_metrics(y,p):
    return {"rmse":float(np.sqrt(mean_squared_error(y,p))),"mae":float(mean_absolute_error(y,p)),"r2":float(r2_score(y,p))}

def clf_metrics(y,p):
    return {"log_loss":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=0.5))}

def cluster_boot_delta_rmse(df,p0,p1,B=5000):
    y=df.speed_mean.to_numpy(float);subs=df.subject.to_numpy();us=np.unique(subs);rng=np.random.default_rng(SEED+11)
    vals=[]
    for _ in range(B):
        pick=rng.choice(us,size=len(us),replace=True)
        idx=np.concatenate([np.flatnonzero(subs==s) for s in pick])
        vals.append(np.sqrt(np.mean((y[idx]-p1[idx])**2))-np.sqrt(np.mean((y[idx]-p0[idx])**2)))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],float(np.mean(vals))

def center_by_group(x,g):
    x=np.asarray(x,float).copy();g=np.asarray(g)
    for u in np.unique(g):
        m=g==u;x[m]-=np.nanmean(x[m])
    return x

def surrogate_control(df,p0,surr_records,B=200):
    idx_by_id={r:i for i,r in enumerate(df.row_id)}
    keep=[r for r in surr_records if r[0] in idx_by_id]
    resid=df.speed_mean.to_numpy(float)-p0
    obs=[];res=[];groups=[]
    for rid,sub,phases in keep:
        i=idx_by_id[rid]
        if np.isfinite(df.loc[i,"tma_mean_d5"]):
            obs.append(float(df.loc[i,"tma_mean_d5"]));res.append(float(resid[i]));groups.append(sub)
    obs=np.asarray(obs);res=np.asarray(res);groups=np.asarray(groups)
    oc=center_by_group(obs,groups);rc=center_by_group(res,groups)
    r_obs=float(np.corrcoef(oc,rc)[0,1])
    rng=np.random.default_rng(SEED+12)
    vals=np.empty((len(keep),B),float)
    valid_rows=[]
    for j,(rid,sub,phases) in enumerate(keep):
        acc=np.zeros(B);n=0
        for ph in phases:
            shifts=rng.random(B)
            z=rotated_tma_batch(ph,shifts,5)
            if np.all(np.isfinite(z)):
                acc+=z;n+=1
        vals[j]=acc/n if n else np.nan
        valid_rows.append(idx_by_id[rid])
        if (j+1)%50==0:print(f"[surrogate] windows {j+1}/{len(keep)}",flush=True)
    # align exactly to keep ordering
    rr=np.array([resid[idx_by_id[rid]] for rid,_,_ in keep])
    gg=np.array([sub for _,sub,_ in keep])
    rr=center_by_group(rr,gg)
    rs=np.empty(B)
    for b in range(B):
        x=center_by_group(vals[:,b],gg)
        rs[b]=np.corrcoef(x,rr)[0,1]
    p=float((1+np.sum(np.abs(rs)>=abs(r_obs)))/(B+1))

    # Stronger null: preserve the number of population events in every theta
    # cycle, but replace their phases by independent Uniform(0,1) draws.
    # This destroys within-cycle event geometry as well as absolute theta phase.
    rng_u=np.random.default_rng(SEED+13)
    uvals=np.empty((len(keep),B),float)
    for j,(rid,sub,phases) in enumerate(keep):
        acc=np.zeros(B);ncy=0
        for ph in phases:
            z=uniform_tma_batch(len(ph),B,5,rng_u)
            if np.all(np.isfinite(z)):
                acc+=z;ncy+=1
        uvals[j]=acc/ncy if ncy else np.nan
        if (j+1)%50==0:print(f"[uniform-null] windows {j+1}/{len(keep)}",flush=True)
    ur=np.empty(B)
    for b in range(B):
        x=center_by_group(uvals[:,b],gg)
        ur[b]=np.corrcoef(x,rr)[0,1]
    pu=float((1+np.sum(np.abs(ur)>=abs(r_obs)))/(B+1))
    return {"n_windows":len(keep),"observed_within_subject_residual_r":r_obs,
            "phase_rotation_abs_tail_p":p,"phase_rotation_r_mean":float(np.nanmean(rs)),
            "phase_rotation_r_sd":float(np.nanstd(rs,ddof=1)),
            "phase_rotation_r_q025":float(np.nanquantile(rs,.025)),
            "phase_rotation_r_q975":float(np.nanquantile(rs,.975)),
            "count_preserving_uniform_abs_tail_p":pu,
            "count_preserving_uniform_r_mean":float(np.nanmean(ur)),
            "count_preserving_uniform_r_sd":float(np.nanstd(ur,ddof=1)),
            "count_preserving_uniform_r_q025":float(np.nanquantile(ur,.025)),
            "count_preserving_uniform_r_q975":float(np.nanquantile(ur,.975))}

def main():
    t=time.time();self_test();print("[self-test] passed",flush=True)
    all_rows=[];surr=[]
    tmp=Path("/tmp/tma_neural_session.nwb")
    for k,(sub,ses,url) in enumerate(ASSETS,1):
        print(f"[progress] {k}/{len(ASSETS)}",flush=True)
        rows,s=process_session(sub,ses,url,tmp)
        all_rows.extend(rows);surr.extend(s)
    df=pd.DataFrame(all_rows)
    if df.empty: raise RuntimeError("no usable windows")
    if set(df.subject.unique()) != {"M01","M02","M03","M05"}:
        raise RuntimeError(f"unexpected subjects {df.subject.unique()}")
    df.to_csv(OUT/"window_features.csv",index=False)
    result={"dataset":"DANDI:001695 draft","n_sessions":int(df.session.nunique()),"n_subjects":int(df.subject.nunique()),
            "n_windows":int(len(df)),"window_s":WINDOW_S,"population_simultaneity_bin_s":POP_BIN_S,
            "mapping":"CA1 pyramidal population spikes within peak-to-peak 6-10 Hz CA1 LFP theta cycles",
            "primary_outcome":"10-second mean running speed (cm/s)","depths":{}}
    print("[model] baseline regression",flush=True)
    p0=nested_group_regression(df,BASE_COLS);m0=reg_metrics(df.speed_mean,p0)
    result["baseline_regression"]=m0
    dc,p0c=nested_group_classifier(df,BASE_COLS)
    yc=(dc.speed_mean>=10).astype(int).to_numpy()
    result["baseline_fast_slow_classification"]=clf_metrics(yc,p0c)

    print("[model] adversarial rich-geometry baseline",flush=True)
    pr0=nested_group_regression(df,RICH_COLS);mr0=reg_metrics(df.speed_mean,pr0)
    drc,p0rc=nested_group_classifier(df,RICH_COLS)
    if not np.array_equal(drc["index"].to_numpy(),dc["index"].to_numpy()):
        raise RuntimeError("rich classification row mismatch")
    mrc0=clf_metrics(yc,p0rc)
    result["rich_geometry_baseline_regression"]=mr0
    result["rich_geometry_baseline_fast_slow_classification"]=mrc0

    pred=pd.DataFrame({"row_id":df.row_id,"subject":df.subject,"session":df.session,"speed_mean":df.speed_mean,
                       "speedhat_baseline":p0,"speedhat_rich_geometry_baseline":pr0})
    for d in DEPTHS:
        print(f"[model] TMA depth {d}",flush=True)
        tc=[f"tma_mean_d{d}",f"tma_sd_d{d}"]
        p1=nested_group_regression(df,BASE_COLS+tc);m1=reg_metrics(df.speed_mean,p1)
        ci,bm=cluster_boot_delta_rmse(df,p0,p1)
        dcc,p1c=nested_group_classifier(df,BASE_COLS+tc)
        if not np.array_equal(dcc["index"].to_numpy(),dc["index"].to_numpy()):
            raise RuntimeError("classification row mismatch")
        cm1=clf_metrics(yc,p1c)
        pr1=nested_group_regression(df,RICH_COLS+tc);mr1=reg_metrics(df.speed_mean,pr1)
        rci,rbm=cluster_boot_delta_rmse(df,pr0,pr1)
        drc1,p1rc=nested_group_classifier(df,RICH_COLS+tc)
        if not np.array_equal(drc1["index"].to_numpy(),dc["index"].to_numpy()):
            raise RuntimeError("rich+TMA classification row mismatch")
        mrc1=clf_metrics(yc,p1rc)
        result["depths"][str(d)]={
            "regression_extended":m1,
            "delta_rmse_extended_minus_baseline":float(m1["rmse"]-m0["rmse"]),
            "delta_r2_extended_minus_baseline":float(m1["r2"]-m0["r2"]),
            "cluster_bootstrap_delta_rmse_95ci":ci,
            "cluster_bootstrap_delta_rmse_mean":bm,
            "fast_slow_extended":cm1,
            "delta_logloss_extended_minus_baseline":float(cm1["log_loss"]-result["baseline_fast_slow_classification"]["log_loss"]),
            "delta_auc_extended_minus_baseline":float(cm1["auc"]-result["baseline_fast_slow_classification"]["auc"]),
            "rich_geometry_plus_tma_regression":mr1,
            "delta_rmse_tma_beyond_rich_geometry":float(mr1["rmse"]-mr0["rmse"]),
            "delta_r2_tma_beyond_rich_geometry":float(mr1["r2"]-mr0["r2"]),
            "rich_geometry_cluster_bootstrap_delta_rmse_95ci":rci,
            "rich_geometry_cluster_bootstrap_delta_rmse_mean":rbm,
            "rich_geometry_plus_tma_fast_slow":mrc1,
            "delta_logloss_tma_beyond_rich_geometry":float(mrc1["log_loss"]-mrc0["log_loss"]),
            "delta_auc_tma_beyond_rich_geometry":float(mrc1["auc"]-mrc0["auc"]),
        }
        pred[f"speedhat_tma_d{d}"]=p1
        pred[f"speedhat_rich_plus_tma_d{d}"]=pr1
    pred.to_csv(OUT/"predictions.csv",index=False)
    print("[surrogate] standard baseline residuals depth 5",flush=True)
    result["phase_rotation_surrogate_d5"]=surrogate_control(df,p0,surr,B=200)
    print("[surrogate] rich-geometry baseline residuals depth 5",flush=True)
    result["rich_geometry_surrogate_d5"]=surrogate_control(df,pr0,surr,B=200)
    result["runtime_s"]=float(time.time()-t)
    with open(OUT/"summary.json","w") as f:json.dump(result,f,indent=2)
    lines=[
        "# TMA neural theta–spike analysis — DANDI 001695","",
        f"Subjects: {result['n_subjects']}; sessions: {result['n_sessions']}; usable 10-s windows: {result['n_windows']}.",
        f"Baseline speed prediction: RMSE {m0['rmse']:.4f} cm/s; R2 {m0['r2']:.4f}.",
        f"Baseline fast-vs-slow: log loss {result['baseline_fast_slow_classification']['log_loss']:.6f}; AUC {result['baseline_fast_slow_classification']['auc']:.4f}.",
        f"Rich non-TMA geometry baseline speed: RMSE {mr0['rmse']:.4f}; R2 {mr0['r2']:.4f}.",
        f"Rich non-TMA geometry fast-vs-slow: log loss {mrc0['log_loss']:.6f}; AUC {mrc0['auc']:.4f}.",""
    ]
    for d in DEPTHS:
        z=result["depths"][str(d)]
        lines.append(f"- Depth {d}: original baseline -> TMA speed delta RMSE {z['delta_rmse_extended_minus_baseline']:+.4f}, delta AUC {z['delta_auc_extended_minus_baseline']:+.4f}; RICH geometry baseline -> +TMA delta RMSE {z['delta_rmse_tma_beyond_rich_geometry']:+.4f}, bootstrap 95% CI {z['rich_geometry_cluster_bootstrap_delta_rmse_95ci']}, delta fast/slow log loss {z['delta_logloss_tma_beyond_rich_geometry']:+.6f}, delta AUC {z['delta_auc_tma_beyond_rich_geometry']:+.4f}.")
    s=result["phase_rotation_surrogate_d5"]
    sr=result["rich_geometry_surrogate_d5"]
    lines += ["",f"Depth-5 phase-rotation control (original baseline residuals): observed r={s['observed_within_subject_residual_r']:+.4f}; p={s['phase_rotation_abs_tail_p']:.5g}; 95% range [{s['phase_rotation_r_q025']:+.4f}, {s['phase_rotation_r_q975']:+.4f}].",f"Depth-5 count-preserving uniform-phase control (original baseline): p={s['count_preserving_uniform_abs_tail_p']:.5g}; 95% range [{s['count_preserving_uniform_r_q025']:+.4f}, {s['count_preserving_uniform_r_q975']:+.4f}].",f"Depth-5 controls after rich non-TMA geometry baseline: observed residual r={sr['observed_within_subject_residual_r']:+.4f}; phase-rotation p={sr['phase_rotation_abs_tail_p']:.5g}; count-preserving uniform-phase p={sr['count_preserving_uniform_abs_tail_p']:.5g}."]
    (OUT/"RESULTS.md").write_text("\\n".join(lines)+"\\n")
    print("\\n".join(lines),flush=True)

if __name__=="__main__":
    main()
