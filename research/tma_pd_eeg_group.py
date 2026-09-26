#!/usr/bin/env python3
from __future__ import annotations

"""
TMA comparison of Parkinson's disease vs healthy-control EEG in OpenNeuro ds007526.

Primary goal:
    Compare PD and HC after transforming EEG event timing into the same frozen
    Universal TMA Core used in the cross-domain work.

Conditions:
    rest (primary; minimizes gait/motion artifact)
    walk (secondary)
    walk-rest within-subject delta (secondary)

Prespecified EEG front ends:
    theta_global   4-8 Hz, scalp-global
    alpha_global   8-13 Hz, scalp-global
    beta_global    13-30 Hz, scalp-global
    mu_central     8-13 Hz, C3/C4
    beta_central   13-30 Hz, C3/C4

Universal representation:
    fixed 120 s segment (30-150 s), divided into 8 x 15 s frames;
    3 separated band-envelope events per frame;
    positions normalized to each frame, then passed unchanged to
    tma_cross_domain_frozen.analyze_frames (256-bin dyadic grid).

Inference:
    - Welch group comparison + Hedges g
    - label-permutation p for raw group difference
    - age/sex-adjusted HC3 OLS group coefficient
    - age/sex exact-sex nearest-age 1:1 matched sensitivity with sign-flip p
    - rest->walk delta group comparison
    - exploratory PD-only UPDRS correlations
"""

import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import signal, stats, optimize
from scipy.io import loadmat
import statsmodels.api as sm

from tma_cross_domain_frozen import analyze_frames, FEATURES

SEED = 20260926
PERM_B = 10000
MATCH_B = 20000
OUT = Path("research/tma_pd_eeg_group_results")
OUT.mkdir(parents=True, exist_ok=True)

REPO_RAW = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds007526/main"
S3_ROOT = "https://s3.amazonaws.com/openneuro.org/ds007526"
CONDITIONS = ("rest", "walk")
FRONTENDS = {
    "theta_global": {"band": (4.0, 8.0), "scope": "global"},
    "alpha_global": {"band": (8.0, 13.0), "scope": "global"},
    "beta_global": {"band": (13.0, 30.0), "scope": "global"},
    "mu_central": {"band": (8.0, 13.0), "scope": "central"},
    "beta_central": {"band": (13.0, 30.0), "scope": "central"},
}
SEGMENT_START = 30.0
SEGMENT_DURATION = 120.0
N_FRAMES = 8
EVENTS_PER_FRAME = 3

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "TMA-PD-EEG-research/2026-09-26"})
LOG = []

def log(x):
    s = str(x)
    print(s, flush=True)
    LOG.append(s)

def get_text(url, timeout=120):
    last = None
    for a in range(6):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(min(2*(a+1), 12))
    raise RuntimeError(f"GET failed {url}: {last}")

def get_bytes(url, timeout=240):
    last = None
    for a in range(6):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(min(3*(a+1), 15))
    raise RuntimeError(f"GET failed {url}: {last}")

def parse_participants():
    txt = get_text(f"{REPO_RAW}/participants.tsv")
    df = pd.read_csv(io.StringIO(txt), sep="\t", na_values=["n/a", "N/A", "na"])
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df

def read_eeg_set(subject, condition):
    url = f"{S3_ROOT}/{subject}/eeg/{subject}_task-{condition}_eeg.set"
    blob = get_bytes(url)
    mat = loadmat(io.BytesIO(blob), squeeze_me=True, struct_as_record=False)
    if "EEG" in mat:
        eeg = mat["EEG"]
        data = np.asarray(eeg.data)
        srate = float(np.asarray(eeg.srate).squeeze())
    elif "data" in mat and "srate" in mat:
        # Some EEGLAB .set files save EEG fields directly at MAT-file top level.
        data = np.asarray(mat["data"])
        srate = float(np.asarray(mat["srate"]).squeeze())
    else:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise RuntimeError(f"EEG data fields not found; mat keys={keys[:30]}")
    if data.dtype.kind in ("U", "S", "O") and data.ndim == 0:
        raise RuntimeError("external .fdt data not expected in this dataset")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError(f"unexpected EEG data shape {data.shape}")
    # EEGLAB normally channels x samples. If reversed, fix.
    if data.shape[0] > data.shape[1] and data.shape[1] <= 256:
        data = data.T
    return data, srate

def channel_table(subject, condition):
    txt = get_text(f"{REPO_RAW}/{subject}/eeg/{subject}_task-{condition}_channels.tsv")
    df = pd.read_csv(io.StringIO(txt), sep="\t")
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df

def scalp_indices(ch):
    names = ch["name"].astype(str).to_numpy()
    idx = []
    for i, n in enumerate(names):
        u = n.upper()
        if u.startswith("EOG") or u == "VREF":
            continue
        idx.append(i)
    return np.asarray(idx, int)

def central_indices(ch, scalp_abs):
    names = ch["name"].astype(str).to_numpy()
    rel = []
    for j, abs_i in enumerate(scalp_abs):
        u = names[abs_i].upper()
        if u in ("C3", "C4"):
            rel.append(j)
    return np.asarray(rel, int)

def robust_band_envelopes(x, fs, bands):
    # x samples x scalp channels
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # instantaneous common-median reference
    x = x - np.median(x, axis=1, keepdims=True)
    out = {}
    for band in bands:
        sos = signal.butter(4, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, x, axis=0)
        med = np.median(y, axis=0)
        mad = 1.4826*np.median(np.abs(y-med), axis=0)
        finite = np.isfinite(mad) & (mad > max(1e-12, np.nanmedian(mad)*1e-4))
        if finite.sum() < 12:
            raise RuntimeError(f"too few usable channels in band {band}: {finite.sum()}")
        z = np.zeros_like(y)
        z[:, finite] = (y[:, finite]-med[finite])/mad[finite]
        z = np.clip(z, -8, 8)
        env = np.abs(signal.hilbert(z, axis=0))
        out[band] = (env, finite)
    return out

def top3_phases(score):
    score = np.asarray(score, float)
    if len(score) < 30:
        raise RuntimeError("frame too short")
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    chosen = []
    minsep = max(1, len(score)//6)
    for idx in order:
        idx = int(idx)
        if all(abs(idx-j) >= minsep for j in chosen):
            chosen.append(idx)
            if len(chosen) == EVENTS_PER_FRAME:
                break
    if len(chosen) < EVENTS_PER_FRAME:
        raise RuntimeError("could not select three separated EEG events")
    return np.sort((np.asarray(chosen, float)+0.5)/len(score))

def frontend_frames(data, fs, ch):
    i0 = int(round(SEGMENT_START*fs))
    i1 = int(round((SEGMENT_START+SEGMENT_DURATION)*fs))
    if data.shape[1] < i1:
        raise RuntimeError(f"recording too short: {data.shape[1]/fs:.1f}s")
    scalp_abs = scalp_indices(ch)
    if np.max(scalp_abs) >= data.shape[0]:
        raise RuntimeError("channel table/data mismatch")
    central_rel = central_indices(ch, scalp_abs)
    if len(central_rel) < 2:
        raise RuntimeError("C3/C4 missing")
    x = data[scalp_abs, i0:i1].T
    bands = sorted(set(tuple(v["band"]) for v in FRONTENDS.values()))
    envs = robust_band_envelopes(x, fs, bands)
    frame_n = int(round((SEGMENT_DURATION/N_FRAMES)*fs))
    results = {}
    for fe, cfg in FRONTENDS.items():
        env, finite = envs[tuple(cfg["band"])]
        if cfg["scope"] == "global":
            idx = np.flatnonzero(finite)
        else:
            idx = np.asarray([j for j in central_rel if finite[j]], int)
            if len(idx) < 2:
                raise RuntimeError(f"{fe}: unusable C3/C4")
        score = np.sqrt(np.mean(env[:, idx]*env[:, idx], axis=1))
        frames = []
        for k in range(N_FRAMES):
            a, b = k*frame_n, (k+1)*frame_n
            frames.append(top3_phases(score[a:b]))
        results[fe] = frames
    return results

def bh_qvalues(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if not len(good):
        return q
    pv = p[good]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    qq = np.minimum.accumulate((ranked*m/np.arange(1,m+1))[::-1])[::-1]
    qq = np.minimum(qq, 1.0)
    q[good[order]] = qq
    return q

def hedges_g(x_pd, x_hc):
    x1 = np.asarray(x_pd, float); x0 = np.asarray(x_hc, float)
    n1, n0 = len(x1), len(x0)
    s1, s0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    sp = math.sqrt(((n1-1)*s1+(n0-1)*s0)/(n1+n0-2))
    if sp <= 0 or not np.isfinite(sp):
        return np.nan
    d = (np.mean(x1)-np.mean(x0))/sp
    J = 1 - 3/(4*(n1+n0)-9)
    return float(J*d)

def perm_p(x_pd, x_hc, B=PERM_B, seed=0):
    x1=np.asarray(x_pd,float);x0=np.asarray(x_hc,float)
    obs=float(np.mean(x1)-np.mean(x0))
    vals=np.r_[x1,x0];n1=len(x1)
    rng=np.random.default_rng(SEED+seed)
    ge=0
    for _ in range(B):
        p=rng.permutation(len(vals))
        d=float(np.mean(vals[p[:n1]])-np.mean(vals[p[n1:]]))
        if abs(d) >= abs(obs)-1e-15: ge+=1
    return (ge+1)/(B+1)

def adjusted_group_test(df, feature):
    q=df[["group","age","sex",feature]].dropna().copy()
    if len(q)<20 or q["group"].nunique()<2:
        return np.nan,np.nan
    y=q[feature].to_numpy(float)
    X=pd.DataFrame({
        "const":1.0,
        "PD":(q["group"].astype(str)=="PD").astype(float),
        "age":q["age"].astype(float),
        "male":(q["sex"].astype(str).str.upper()=="M").astype(float),
    })
    fit=sm.OLS(y,X).fit(cov_type="HC3")
    return float(fit.params["PD"]),float(fit.pvalues["PD"])

def group_tests(featdf, participant_meta):
    df=featdf.merge(participant_meta,on="participant_id",how="left")
    rows=[]
    for cond in CONDITIONS:
      for fe in FRONTENDS:
        z=df[(df.condition==cond)&(df.frontend==fe)]
        for f in FEATURES:
            pdv=z.loc[z.group=="PD",f].dropna().to_numpy(float)
            hcv=z.loc[z.group=="HC",f].dropna().to_numpy(float)
            if len(pdv)<5 or len(hcv)<5:continue
            wt=stats.ttest_ind(pdv,hcv,equal_var=False)
            beta,padj=adjusted_group_test(z,f)
            rows.append({
                "condition":cond,"frontend":fe,"feature":f,
                "n_PD":len(pdv),"n_HC":len(hcv),
                "mean_PD":float(np.mean(pdv)),"sd_PD":float(np.std(pdv,ddof=1)),
                "mean_HC":float(np.mean(hcv)),"sd_HC":float(np.std(hcv,ddof=1)),
                "diff_PD_minus_HC":float(np.mean(pdv)-np.mean(hcv)),
                "hedges_g":hedges_g(pdv,hcv),
                "welch_p":float(wt.pvalue),
                "perm_p":float(perm_p(pdv,hcv,seed=sum(map(ord,cond+fe+f)))),
                "adjusted_PD_beta":beta,"adjusted_p":padj,
            })
    # conservative global FDR across the whole prespecified family
    for key in ("welch_p","perm_p","adjusted_p"):
        q=bh_qvalues([r[key] for r in rows])
        for r,qq in zip(rows,q):r[key.replace("_p","_q")]=float(qq) if np.isfinite(qq) else np.nan
    return rows

def exact_sex_age_match(z, meta):
    ids=z["participant_id"].unique().tolist()
    m=meta[meta.participant_id.isin(ids)][["participant_id","group","age","sex"]].dropna()
    hc=m[m.group=="HC"].reset_index(drop=True)
    pdm=m[m.group=="PD"].reset_index(drop=True)
    if len(hc)==0 or len(pdm)<len(hc):return []
    cost=np.zeros((len(hc),len(pdm)),float)
    for i,a in hc.iterrows():
        for j,b in pdm.iterrows():
            penalty=10000.0 if str(a.sex).upper()!=str(b.sex).upper() else 0.0
            cost[i,j]=penalty+abs(float(a.age)-float(b.age))
    ri,cj=optimize.linear_sum_assignment(cost)
    pairs=[]
    for i,j in zip(ri,cj):
        if cost[i,j]>=10000:continue
        pairs.append((hc.loc[i,"participant_id"],pdm.loc[j,"participant_id"],
                      float(hc.loc[i,"age"]),float(pdm.loc[j,"age"]),str(hc.loc[i,"sex"])))
    return pairs

def signflip_p(diff,B=MATCH_B,seed=0):
    diff=np.asarray(diff,float)
    obs=float(np.mean(diff))
    rng=np.random.default_rng(SEED+seed)
    ge=0
    for _ in range(B):
        signs=rng.choice(np.array([-1.0,1.0]),size=len(diff))
        v=float(np.mean(diff*signs))
        if abs(v)>=abs(obs)-1e-15:ge+=1
    return (ge+1)/(B+1)

def matched_tests(featdf, meta):
    rows=[]
    for cond in CONDITIONS:
      for fe in FRONTENDS:
        z=featdf[(featdf.condition==cond)&(featdf.frontend==fe)]
        pairs=exact_sex_age_match(z,meta)
        if len(pairs)<10:continue
        zz=z.set_index("participant_id")
        for f in FEATURES:
            dif=[]
            for hc,pd,_,_,_ in pairs:
                if hc in zz.index and pd in zz.index:
                    a=float(zz.loc[hc,f]);b=float(zz.loc[pd,f])
                    if np.isfinite(a) and np.isfinite(b):dif.append(b-a)
            if len(dif)<10:continue
            dif=np.asarray(dif,float)
            sd=np.std(dif,ddof=1)
            rows.append({
                "condition":cond,"frontend":fe,"feature":f,"n_pairs":len(dif),
                "mean_PD_minus_HC":float(np.mean(dif)),
                "paired_dz":float(np.mean(dif)/sd) if sd>0 else np.nan,
                "signflip_p":float(signflip_p(dif,seed=sum(map(ord,cond+fe+f)))),
                "mean_abs_age_difference":float(np.mean([abs(a-b) for _,_,a,b,_ in pairs])),
                "max_abs_age_difference":float(np.max([abs(a-b) for _,_,a,b,_ in pairs])),
            })
    q=bh_qvalues([r["signflip_p"] for r in rows])
    for r,qq in zip(rows,q):r["signflip_q"]=float(qq) if np.isfinite(qq) else np.nan
    return rows

def task_delta_tests(featdf, meta):
    rows=[]
    for fe in FRONTENDS:
      a=featdf[(featdf.condition=="rest")&(featdf.frontend==fe)].set_index("participant_id")
      b=featdf[(featdf.condition=="walk")&(featdf.frontend==fe)].set_index("participant_id")
      common=sorted(set(a.index)&set(b.index))
      dm=meta[meta.participant_id.isin(common)].copy()
      for f in FEATURES:
        vals=[]
        for s in common:
            vals.append({"participant_id":s,"delta":float(b.loc[s,f]-a.loc[s,f])})
        q=pd.DataFrame(vals).merge(dm,on="participant_id",how="left")
        pdv=q.loc[q.group=="PD","delta"].dropna().to_numpy(float)
        hcv=q.loc[q.group=="HC","delta"].dropna().to_numpy(float)
        if len(pdv)<5 or len(hcv)<5:continue
        wt=stats.ttest_ind(pdv,hcv,equal_var=False)
        beta,padj=adjusted_group_test(q.rename(columns={"delta":f}),f)
        rows.append({
            "frontend":fe,"feature":f,"n_PD":len(pdv),"n_HC":len(hcv),
            "mean_delta_PD":float(np.mean(pdv)),"mean_delta_HC":float(np.mean(hcv)),
            "delta_diff_PD_minus_HC":float(np.mean(pdv)-np.mean(hcv)),
            "hedges_g":hedges_g(pdv,hcv),"welch_p":float(wt.pvalue),
            "perm_p":float(perm_p(pdv,hcv,seed=30000+sum(map(ord,fe+f)))),
            "adjusted_PD_beta":beta,"adjusted_p":padj,
        })
    for key in ("welch_p","perm_p","adjusted_p"):
        q=bh_qvalues([r[key] for r in rows])
        for r,qq in zip(rows,q):r[key.replace("_p","_q")]=float(qq) if np.isfinite(qq) else np.nan
    return rows

def pd_clinical_correlations(featdf, meta):
    cols=["updrs_part_iii","updrs_total","moca","disease_duration","ledd","pigd_score","td_score","ctt"]
    rows=[]
    d=featdf.merge(meta,on="participant_id",how="left")
    d=d[d.group=="PD"]
    for cond in CONDITIONS:
      for fe in FRONTENDS:
        z=d[(d.condition==cond)&(d.frontend==fe)]
        for f in FEATURES:
          for c in cols:
            q=z[[f,c]].dropna()
            if len(q)<20 or q[f].nunique()<3 or q[c].nunique()<3:continue
            s=stats.spearmanr(q[f].to_numpy(float),q[c].to_numpy(float))
            rows.append({"condition":cond,"frontend":fe,"feature":f,"clinical":c,
                         "n":len(q),"rho":float(s.statistic),"p":float(s.pvalue)})
    q=bh_qvalues([r["p"] for r in rows])
    for r,qq in zip(rows,q):r["q_global"]=float(qq) if np.isfinite(qq) else np.nan
    return rows

def demographics(meta, featdf):
    out=[]
    for cond in CONDITIONS:
        ids=set(featdf.loc[featdf.condition==cond,"participant_id"])
        m=meta[meta.participant_id.isin(ids)]
        for g in ("HC","PD"):
            q=m[m.group==g]
            out.append({
                "condition":cond,"group":g,"n":len(q),
                "age_mean":float(q.age.mean()),"age_sd":float(q.age.std(ddof=1)),
                "male_n":int((q.sex.astype(str).str.upper()=="M").sum()),
                "female_n":int((q.sex.astype(str).str.upper()=="F").sum()),
            })
    return out

def main():
    t0=time.time()
    meta=parse_participants()
    rows=[]; failures={}
    # rest is present for all 144; walking is absent for 11 and handled by 404/failure.
    for si,r in meta.iterrows():
        s=str(r["participant_id"])
        for cond in CONDITIONS:
            try:
                data,fs=read_eeg_set(s,cond)
                ch=channel_table(s,cond)
                fronts=frontend_frames(data,fs,ch)
                for fe,frames in fronts.items():
                    feat=analyze_frames(frames)
                    rows.append({"participant_id":s,"condition":cond,"frontend":fe,**feat})
                log(f"{s} {cond}: OK")
            except Exception as e:
                failures[f"{s}:{cond}"]=repr(e)
                log(f"{s} {cond}: FAIL {e!r}")
    featdf=pd.DataFrame(rows)
    featdf.to_csv(OUT/"subject_tma_features.csv",index=False)

    # Require complete five-front-end result for a condition.
    good=[]
    for (s,c),q in featdf.groupby(["participant_id","condition"]):
        if set(q.frontend)==set(FRONTENDS):
            good.append((s,c))
    goodset=set(good)
    featdf=featdf[[ (s,c) in goodset for s,c in zip(featdf.participant_id,featdf.condition)]].copy()

    dem=demographics(meta,featdf)
    pd.DataFrame(dem).to_csv(OUT/"demographics.csv",index=False)

    gt=group_tests(featdf,meta)
    pd.DataFrame(gt).to_csv(OUT/"group_tests.csv",index=False)

    mt=matched_tests(featdf,meta)
    pd.DataFrame(mt).to_csv(OUT/"age_sex_matched_tests.csv",index=False)

    dt=task_delta_tests(featdf,meta)
    pd.DataFrame(dt).to_csv(OUT/"rest_to_walk_delta_tests.csv",index=False)

    cc=pd_clinical_correlations(featdf,meta)
    pd.DataFrame(cc).to_csv(OUT/"pd_clinical_correlations.csv",index=False)

    def best(rows,key,n=20):
        return sorted([x for x in rows if np.isfinite(x.get(key,np.nan))],key=lambda x:x[key])[:n]

    summary={
        "dataset":"OpenNeuro ds007526 / PD-EEG",
        "seed":SEED,
        "design":{
            "groups":{"HC":int((meta.group=="HC").sum()),"PD":int((meta.group=="PD").sum())},
            "segment_seconds":[SEGMENT_START,SEGMENT_START+SEGMENT_DURATION],
            "frames":N_FRAMES,"events_per_frame":EVENTS_PER_FRAME,
            "frontends":FRONTENDS,
            "tma_core":"frozen Universal TMA Core: research/tma_cross_domain_frozen.py::analyze_frames",
        },
        "demographics":dem,
        "failures":failures,
        "n_feature_rows":len(featdf),
        "best_adjusted_group_tests":best(gt,"adjusted_p",30),
        "best_group_permutation_tests":best(gt,"perm_p",30),
        "best_matched_tests":best(mt,"signflip_p",30),
        "best_task_delta_adjusted_tests":best(dt,"adjusted_p",30),
        "best_pd_clinical_correlations":best(cc,"p",30),
        "runtime_s":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
    log(json.dumps(summary,indent=2))
    log(f"runtime_s={time.time()-t0:.1f}")

if __name__=="__main__":
    main()
