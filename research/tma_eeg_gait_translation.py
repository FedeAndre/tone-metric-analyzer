#!/usr/bin/env python3
from __future__ import annotations

"""
Same-subject EEG<->gait TMA translational-transform test for OpenNeuro/NEMAR ds004475.

Design:
- Shared biological frame: one LHS-to-LHS stride.
- Gait events: RTO, RHS, LTO.
- EEG events: the three strongest separated peaks of beta-band (13-30 Hz)
  robust global field power within the *same* stride.
- Universal TMA core: 8 frames x 3 events, 256-bin dyadic grid, using the
  frozen tma_cross_domain_frozen.py implementation and its universal-null scaling.
- Primary translational tests:
  (1) matched-subject EEG/gait distances vs mismatched subjects,
  (2) cross-modal preservation of subject-distance geometry (Mantel/Spearman),
  (3) matched change-vectors baseline -> adaptation,
  (4) same-feature change correlations,
  (5) circular phase-rotation EEG control.
"""

import io
import json
import math
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy import signal, stats

from tma_cross_domain_frozen import analyze_frames, FEATURES

SEED = 20260926
RNG = np.random.default_rng(SEED)
PERM_B = 5000
PHASE_B = 300
OUT = Path("research/tma_eeg_gait_translation_results")
OUT.mkdir(parents=True, exist_ok=True)

REPO_RAW = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds004475/main"
S3_ROOT = "https://s3.amazonaws.com/openneuro.org/ds004475"
SUBJECTS = [
    "S18","S19","S20","S21","S22","S23","S24","S25","S26","S27","S28","S29",
    "S30","S31","S32","S33","S34","S35","S36","S37","S38","S39","S40","S41",
    "S42","S43","S44","S46","S47","S48"
]
CONDITIONS = ("baseline","early_abrupt","late_abrupt","early_gradual","late_gradual")
PRIMARY_CHANGE = "late_abrupt"

LOG = []
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"TMA-cross-domain-research/2026-09-26"})

def log(x):
    s = str(x)
    print(s, flush=True)
    LOG.append(s)

def get_text(url, timeout=120):
    last = None
    for a in range(5):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(2*(a+1))
    raise RuntimeError(f"GET failed {url}: {last}")

def get_range(url, first_byte, last_byte, timeout=180):
    if last_byte < first_byte:
        raise ValueError("empty byte range")
    expected = last_byte-first_byte+1
    last = None
    for a in range(6):
        try:
            with SESSION.get(
                url,
                headers={"Range":f"bytes={first_byte}-{last_byte}"},
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            ) as r:
                # Refuse a range-ignoring origin rather than accidentally downloading ~1.7 GB.
                if r.status_code != 206:
                    raise RuntimeError(f"range request returned HTTP {r.status_code}")
                cr = r.headers.get("Content-Range","")
                if not cr.lower().startswith("bytes "):
                    raise RuntimeError(f"missing Content-Range: {cr!r}")
                data = r.content
            if len(data) != expected:
                raise RuntimeError(f"range size {len(data)} != expected {expected}")
            return data
        except Exception as e:
            last = e
            time.sleep(min(3*(a+1),15))
    raise RuntimeError(f"Range GET failed {url}: {last}")

def parse_events(text):
    df = pd.read_csv(io.StringIO(text), sep="\t")
    df["value"] = df["value"].astype(str)
    return df

def marker_time(df, label):
    q = df.loc[df.value == label, "onset"]
    return float(q.iloc[0]) if len(q) else None

def eligible_cycles(df, lo, hi):
    lhs = np.asarray(df.loc[df.value=="LHS","onset"], float)
    rhs = np.asarray(df.loc[df.value=="RHS","onset"], float)
    rto = np.asarray(df.loc[df.value=="RTO","onset"], float)
    lto = np.asarray(df.loc[df.value=="LTO","onset"], float)
    out = []
    for i in range(len(lhs)-1):
        a,b = float(lhs[i]),float(lhs[i+1])
        if a < lo or b > hi or b <= a:
            continue
        er = rto[(rto>a)&(rto<b)]
        hr = rhs[(rhs>a)&(rhs<b)]
        el = lto[(lto>a)&(lto<b)]
        if not len(er) or not len(hr) or not len(el):
            continue
        er,hr,el = float(er[0]),float(hr[0]),float(el[0])
        if not (er < hr < el):
            continue
        ph = np.array([(er-a)/(b-a),(hr-a)/(b-a),(el-a)/(b-a)],float)
        if np.any(~np.isfinite(ph)) or np.any(ph<=0) or np.any(ph>=1):
            continue
        out.append({"a":a,"b":b,"gait":ph})
    return out

def select_eight(cycles, mode):
    if mode == "last":
        return cycles[-8:] if len(cycles) >= 8 else None
    if mode == "post10":
        return cycles[10:18] if len(cycles) >= 18 else None
    if mode == "first":
        return cycles[:8] if len(cycles) >= 8 else None
    raise ValueError(mode)

def condition_cycles(df):
    b3 = marker_time(df,"B3")
    sb1 = marker_time(df,"SB1")
    p1 = marker_time(df,"P1")
    endp1 = marker_time(df,"End P1")
    sb2 = marker_time(df,"SB2")
    p2 = marker_time(df,"P2")
    if None in (b3,sb1,p1,sb2,p2):
        return {}
    specs = {
        "baseline": (b3,sb1,"last"),
        "early_abrupt": (sb1,p1,"post10"),
        "late_abrupt": (sb1,p1,"last"),
        "early_gradual": (sb2,p2,"post10"),
        "late_gradual": (sb2,p2,"last"),
    }
    out = {}
    for name,(lo,hi,mode) in specs.items():
        c = eligible_cycles(df,lo,hi)
        q = select_eight(c,mode)
        if q is not None and len(q)==8:
            out[name]=q
    return out

def fdt_window(subject, t0, t1, fs, n_total, duration_s):
    # EEGLAB .fdt stores float32 with channels as the fastest varying dimension.
    start = max(0, int(math.floor(t0*fs)))
    stop = min(int(math.floor(duration_s*fs)), int(math.ceil(t1*fs)))
    if stop <= start:
        raise RuntimeError("invalid EEG time window")
    byte0 = start*n_total*4
    byte1 = stop*n_total*4 - 1
    url = f"{S3_ROOT}/sub-{subject}/eeg/sub-{subject}_task-task_eeg.fdt"
    raw = get_range(url,byte0,byte1)
    x = np.frombuffer(raw,dtype="<f4")
    if x.size != (stop-start)*n_total:
        raise RuntimeError(f"{subject}: decoded float count mismatch")
    x = x.reshape(stop-start,n_total)
    return x,start/fs

def robust_beta_gfp(raw, n_eeg, fs):
    x = np.asarray(raw[:,:n_eeg],dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
    # Robust instantaneous common reference to attenuate common movement/electrode offsets.
    x -= np.median(x,axis=1,keepdims=True)
    sos = signal.butter(4,[13.0,30.0],btype="bandpass",fs=fs,output="sos")
    y = signal.sosfiltfilt(sos,x,axis=0)
    med = np.median(y,axis=0)
    mad = 1.4826*np.median(np.abs(y-med),axis=0)
    finite = np.isfinite(mad) & (mad > np.nanmedian(mad)*1e-4) & (mad>1e-12)
    if finite.sum() < max(16,n_eeg//4):
        raise RuntimeError(f"too few usable EEG channels: {finite.sum()}")
    z = (y[:,finite]-med[finite])/mad[finite]
    z = np.clip(z,-8,8)
    env = np.abs(signal.hilbert(z,axis=0))
    score = np.sqrt(np.mean(env*env,axis=1))
    return score

def top3_score_phases(score):
    score = np.asarray(score,float)
    if len(score)<10:
        raise RuntimeError("EEG stride too short")
    order = np.argsort(np.nan_to_num(score,nan=-np.inf))[::-1]
    chosen=[]
    minsep=max(1,len(score)//6)
    for idx in order:
        idx=int(idx)
        if all(abs(idx-j)>=minsep for j in chosen):
            chosen.append(idx)
            if len(chosen)==3:
                break
    if len(chosen)<3:
        raise RuntimeError("could not select three EEG events")
    return np.sort((np.asarray(chosen)+0.5)/len(score))

def eeg_frames(subject, cycles, fs, n_total, n_eeg, duration_s):
    lo = cycles[0]["a"]-1.5
    hi = cycles[-1]["b"]+1.5
    raw, raw_t0 = fdt_window(subject,lo,hi,fs,n_total,duration_s)
    score = robust_beta_gfp(raw,n_eeg,fs)
    frames=[]
    for c in cycles:
        i0=max(0,int(round((c["a"]-raw_t0)*fs)))
        i1=min(len(score),int(round((c["b"]-raw_t0)*fs)))
        if i1-i0<20:
            raise RuntimeError("short EEG gait cycle")
        frames.append(top3_score_phases(score[i0:i1]))
    return frames

def gait_frames(cycles):
    return [np.asarray(c["gait"],float) for c in cycles]

def bh_qvalues(p):
    p=np.asarray(p,float)
    q=np.full(len(p),np.nan)
    good=np.flatnonzero(np.isfinite(p))
    if not len(good):
        return q
    pv=p[good]; order=np.argsort(pv); ranked=pv[order]; m=len(ranked)
    qq=np.minimum.accumulate((ranked*m/np.arange(1,m+1))[::-1])[::-1]
    qq=np.minimum(qq,1.0)
    q[good[order]]=qq
    return q

def rmsdist(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2)))

def pdist_matrix(X):
    X=np.asarray(X,float)
    return np.sqrt(np.mean((X[:,None,:]-X[None,:,:])**2,axis=2))

def cross_modal_identity(zdf, condition, features, B=PERM_B):
    g=zdf[(zdf.modality=="gait")&(zdf.condition==condition)].set_index("subject")
    e=zdf[(zdf.modality=="eeg")&(zdf.condition==condition)].set_index("subject")
    common=sorted(set(g.index)&set(e.index))
    G=g.loc[common,features].to_numpy(float)
    E=e.loc[common,features].to_numpy(float)
    D=np.sqrt(np.mean((G[:,None,:]-E[None,:,:])**2,axis=2))
    diag=np.diag(D)
    off=D[~np.eye(len(common),dtype=bool)]
    top1=float(np.mean(np.argmin(D,axis=1)==np.arange(len(common))))
    ranks=[]
    for i in range(len(common)):
        ranks.append(int(np.where(np.argsort(D[i])==i)[0][0])+1)
    rng=np.random.default_rng(SEED+sum(map(ord,condition)))
    perm_diag=[];perm_top=[]
    for _ in range(B):
        p=rng.permutation(len(common))
        perm_diag.append(float(np.mean(D[np.arange(len(common)),p])))
        perm_top.append(float(np.mean(np.argmin(D[:,p],axis=1)==np.arange(len(common)))))
    p_dist=(1+np.sum(np.asarray(perm_diag)<=np.mean(diag)))/(B+1)
    p_top=(1+np.sum(np.asarray(perm_top)>=top1))/(B+1)

    # Mantel-style preservation of within-modality subject geometry.
    DG=pdist_matrix(G);DE=pdist_matrix(E)
    iu=np.triu_indices(len(common),1)
    obs=float(stats.spearmanr(DG[iu],DE[iu]).statistic)
    null=[]
    for _ in range(B):
        p=rng.permutation(len(common))
        q=DE[p][:,p]
        null.append(float(stats.spearmanr(DG[iu],q[iu]).statistic))
    p_mantel=(1+np.sum(np.asarray(null)>=obs))/(B+1)
    return {
        "condition":condition,"n":len(common),
        "matched_distance_mean":float(np.mean(diag)),
        "mismatched_distance_mean":float(np.mean(off)),
        "matched_over_mismatched":float(np.mean(diag)/np.mean(off)),
        "matched_distance_permutation_p":float(p_dist),
        "top1_retrieval":top1,"chance_top1":1/len(common),"top1_permutation_p":float(p_top),
        "mean_true_match_rank":float(np.mean(ranks)),
        "mantel_spearman":obs,"mantel_permutation_p":float(p_mantel),
    }

def change_analysis(zdf, condition, features, B=PERM_B):
    common=None; mats={}
    for mod in ("gait","eeg"):
        b=zdf[(zdf.modality==mod)&(zdf.condition=="baseline")].set_index("subject")
        c=zdf[(zdf.modality==mod)&(zdf.condition==condition)].set_index("subject")
        ids=sorted(set(b.index)&set(c.index))
        common=set(ids) if common is None else common&set(ids)
        mats[mod]=(b,c)
    common=sorted(common)
    G=mats["gait"][1].loc[common,features].to_numpy(float)-mats["gait"][0].loc[common,features].to_numpy(float)
    E=mats["eeg"][1].loc[common,features].to_numpy(float)-mats["eeg"][0].loc[common,features].to_numpy(float)
    D=np.sqrt(np.mean((G[:,None,:]-E[None,:,:])**2,axis=2))
    diag=np.diag(D);off=D[~np.eye(len(common),dtype=bool)]
    def cosine(a,b):
        den=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/den) if den>0 else np.nan
    C=np.array([[cosine(G[i],E[j]) for j in range(len(common))] for i in range(len(common))])
    cdiag=np.diag(C);coff=C[~np.eye(len(common),dtype=bool)]
    top1=float(np.mean(np.argmin(D,axis=1)==np.arange(len(common))))
    rng=np.random.default_rng(SEED+7000+sum(map(ord,condition)))
    nd=[];nc=[];nt=[]
    for _ in range(B):
        p=rng.permutation(len(common))
        nd.append(float(np.mean(D[np.arange(len(common)),p])))
        nc.append(float(np.nanmean(C[np.arange(len(common)),p])))
        nt.append(float(np.mean(np.argmin(D[:,p],axis=1)==np.arange(len(common)))))
    return {
        "condition":condition,"n":len(common),
        "matched_change_distance":float(np.mean(diag)),
        "mismatched_change_distance":float(np.mean(off)),
        "distance_ratio":float(np.mean(diag)/np.mean(off)),
        "distance_permutation_p":float((1+np.sum(np.asarray(nd)<=np.mean(diag)))/(B+1)),
        "matched_change_cosine":float(np.nanmean(cdiag)),
        "mismatched_change_cosine":float(np.nanmean(coff)),
        "cosine_permutation_p":float((1+np.sum(np.asarray(nc)>=np.nanmean(cdiag)))/(B+1)),
        "top1_change_retrieval":top1,
        "chance_top1":1/len(common),
        "top1_permutation_p":float((1+np.sum(np.asarray(nt)>=top1))/(B+1)),
    }, common, G, E

def feature_change_correlations(zdf, condition, features):
    rows=[]
    for f in features:
        gb=zdf[(zdf.modality=="gait")&(zdf.condition=="baseline")].set_index("subject")
        gc=zdf[(zdf.modality=="gait")&(zdf.condition==condition)].set_index("subject")
        eb=zdf[(zdf.modality=="eeg")&(zdf.condition=="baseline")].set_index("subject")
        ec=zdf[(zdf.modality=="eeg")&(zdf.condition==condition)].set_index("subject")
        ids=sorted(set(gb.index)&set(gc.index)&set(eb.index)&set(ec.index))
        dg=gc.loc[ids,f].to_numpy(float)-gb.loc[ids,f].to_numpy(float)
        de=ec.loc[ids,f].to_numpy(float)-eb.loc[ids,f].to_numpy(float)
        if np.nanstd(dg)<1e-12 or np.nanstd(de)<1e-12:
            rho,p=np.nan,np.nan
        else:
            s=stats.spearmanr(dg,de,nan_policy="omit")
            rho,p=float(s.statistic),float(s.pvalue)
        rows.append({"condition":condition,"feature":f,"n":len(ids),"spearman_rho":rho,"p":p})
    q=bh_qvalues([r["p"] for r in rows])
    for r,qq in zip(rows,q):r["q_fdr"]=float(qq) if np.isfinite(qq) else np.nan
    return rows

def modality_condition_effects(rawdf):
    rows=[]
    # Raw TMA features: descriptive paired effects, not needed for common-space scaling.
    for mod in ("gait","eeg"):
        for cond in CONDITIONS:
            if cond=="baseline":continue
            b=rawdf[(rawdf.modality==mod)&(rawdf.condition=="baseline")].set_index("subject")
            c=rawdf[(rawdf.modality==mod)&(rawdf.condition==cond)].set_index("subject")
            ids=sorted(set(b.index)&set(c.index))
            for f in FEATURES:
                if f not in b.columns or f not in c.columns:continue
                d=c.loc[ids,f].to_numpy(float)-b.loc[ids,f].to_numpy(float)
                if len(d)<3 or np.std(d,ddof=1)<1e-12:
                    t,p,dz=np.nan,np.nan,np.nan
                else:
                    tt=stats.ttest_1samp(d,0,nan_policy="omit")
                    t,p=float(tt.statistic),float(tt.pvalue)
                    dz=float(np.mean(d)/np.std(d,ddof=1))
                rows.append({"modality":mod,"condition":cond,"feature":f,"n":len(d),
                             "mean_change":float(np.nanmean(d)),"t":t,"p":p,"paired_dz":dz})
    # FDR within modality across all condition-feature tests.
    for mod in ("gait","eeg"):
        ids=[i for i,r in enumerate(rows) if r["modality"]==mod]
        q=bh_qvalues([rows[i]["p"] for i in ids])
        for i,qq in zip(ids,q):rows[i]["q_fdr"]=float(qq) if np.isfinite(qq) else np.nan
    return rows

def rotate_frames(frames,rng):
    out=[]
    for ph in frames:
        shift=float(rng.random())
        out.append(np.sort(np.mod(np.asarray(ph,float)+shift,1.0)))
    return out

def phase_rotation_control(frame_store, zdf, mu,sd,features, condition=PRIMARY_CHANGE, B=PHASE_B):
    # Observed cross-modal change cosine.
    obs,common,G,_=change_analysis(zdf,condition,features,B=1000)
    obs_cos=obs["matched_change_cosine"]
    gait_by={s:G[i] for i,s in enumerate(common)}
    rng=np.random.default_rng(SEED+991)
    null=[]
    for b in range(B):
        vals=[]
        for s in common:
            fb=rotate_frames(frame_store[(s,"baseline","eeg")],rng)
            fc=rotate_frames(frame_store[(s,condition,"eeg")],rng)
            vb=analyze_frames(fb);vc=analyze_frames(fc)
            de=np.array([(vc[f]-mu[f])/sd[f]-(vb[f]-mu[f])/sd[f] for f in features],float)
            dg=gait_by[s]
            den=np.linalg.norm(dg)*np.linalg.norm(de)
            if den>0: vals.append(float(np.dot(dg,de)/den))
        null.append(float(np.mean(vals)) if vals else np.nan)
        if (b+1)%50==0:log(f"phase null {b+1}/{B}")
    null=np.asarray(null,float)
    p=float((1+np.sum(null>=obs_cos))/(1+np.sum(np.isfinite(null))))
    return {"condition":condition,"B":B,"observed_matched_change_cosine":obs_cos,
            "phase_rotation_null_mean":float(np.nanmean(null)),
            "phase_rotation_null_sd":float(np.nanstd(null,ddof=1)),
            "phase_rotation_p":p}

def main():
    t0=time.time()
    null=pd.read_csv("research/tma_cross_domain_frozen_results/universal_null_features.csv")
    mu=null[FEATURES].mean()
    sd=null[FEATURES].std(ddof=1)
    features=[f for f in FEATURES if np.isfinite(sd[f]) and sd[f]>1e-8]
    log(f"Universal core features retained: {len(features)} = {features}")

    rows=[]
    frame_store={}
    failed={}
    for si,s in enumerate(SUBJECTS,1):
        try:
            evtxt=get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_events.tsv")
            meta=json.loads(get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_eeg.json"))
            ch=pd.read_csv(io.StringIO(get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_channels.tsv")),sep="\t")
            fs=float(meta["SamplingFrequency"])
            duration=float(meta["RecordingDuration"])
            n_eeg=int(np.sum(ch["type"].astype(str).str.upper()=="EEG"))
            n_total=len(ch)
            cc=condition_cycles(parse_events(evtxt))
            missing=[c for c in CONDITIONS if c not in cc]
            if missing:raise RuntimeError(f"missing condition cycle samples {missing}")
            for cond in CONDITIONS:
                cyc=cc[cond]
                gf=gait_frames(cyc)
                ef=eeg_frames(s,cyc,fs,n_total,n_eeg,duration)
                frame_store[(s,cond,"gait")]=gf
                frame_store[(s,cond,"eeg")]=ef
                for mod,fr in (("gait",gf),("eeg",ef)):
                    feat=analyze_frames(fr)
                    rows.append({"subject":s,"condition":cond,"modality":mod,**feat})
            log(f"{si}/{len(SUBJECTS)} {s}: OK")
        except Exception as e:
            failed[s]=repr(e)
            log(f"{si}/{len(SUBJECTS)} {s}: FAIL {e!r}")

    rawdf=pd.DataFrame(rows)
    rawdf.to_csv(OUT/"sample_features_raw.csv",index=False)
    good=sorted(set(rawdf[rawdf.modality=="gait"].subject)&set(rawdf[rawdf.modality=="eeg"].subject))
    complete=[s for s in good if all(((rawdf.subject==s)&(rawdf.condition==c)&(rawdf.modality==m)).any()
                                     for c in CONDITIONS for m in ("gait","eeg"))]
    rawdf=rawdf[rawdf.subject.isin(complete)].copy()
    if len(complete)<15:
        raise RuntimeError(f"Only {len(complete)} complete subjects; failures={failed}")

    zdf=rawdf.copy()
    for f in features:zdf[f]=(zdf[f]-mu[f])/sd[f]
    zdf.to_csv(OUT/"sample_signatures_z.csv",index=False)

    identities=[cross_modal_identity(zdf,c,features) for c in CONDITIONS]
    pd.DataFrame(identities).to_csv(OUT/"cross_modal_identity.csv",index=False)

    changes=[];corrs=[]
    for c in CONDITIONS:
        if c=="baseline":continue
        row,_,_,_=change_analysis(zdf,c,features)
        changes.append(row)
        corrs.extend(feature_change_correlations(zdf,c,features))
    pd.DataFrame(changes).to_csv(OUT/"cross_modal_changes.csv",index=False)
    pd.DataFrame(corrs).to_csv(OUT/"feature_change_correlations.csv",index=False)

    effects=modality_condition_effects(rawdf)
    pd.DataFrame(effects).to_csv(OUT/"modality_condition_effects.csv",index=False)

    phase=phase_rotation_control(frame_store,zdf,mu,sd,features)
    pd.DataFrame([phase]).to_csv(OUT/"phase_rotation_control.csv",index=False)

    summary={
        "dataset":"OpenNeuro ds004475 / NEMAR on004475",
        "seed":SEED,
        "n_complete_subjects":len(complete),
        "subjects":complete,
        "failed_subjects":failed,
        "eeg_front_end":{
            "shared_frame":"LHS-to-next-LHS stride; exactly same frames as gait",
            "channels":"all BIDS EEG channels only",
            "reference":"instantaneous spatial median",
            "band_hz":[13.0,30.0],
            "event_score":"RMS analytic envelope of robust-z beta-band EEG across usable channels",
            "events_per_frame":3,
            "minimum_peak_separation":"one sixth of stride samples",
        },
        "tma_core":{
            "frames":8,"events_per_frame":3,"grid":256,"arity":2,
            "implementation":"research/tma_cross_domain_frozen.py::analyze_frames",
            "universal_null":"existing frozen 1500-sample universal null",
            "features":features,
        },
        "identity_tests":identities,
        "change_tests":changes,
        "phase_rotation_control":phase,
        "runtime_s":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
    log(json.dumps(summary,indent=2))
    log(f"runtime_s={time.time()-t0:.1f}")

if __name__=="__main__":
    main()
