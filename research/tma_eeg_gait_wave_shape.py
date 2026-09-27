#!/usr/bin/env python3
from __future__ import annotations

"""
EEG<->gait TMA wave-shape correlation test for OpenNeuro ds004475.

Purpose
-------
Tests whether the *event-wise TMA trajectories* themselves correspond between
simultaneously recorded gait and EEG, even when subject-level summary features
do not.

The frozen TMA mathematics is unchanged.  We derive the 24-event wave
(H, D, lambda), pivot positions, and predecessor-tree branch spans from exactly
the same 8-frame x 3-event representation already used in the cross-domain work.

Primary tests
-------------
For each of 5 EEG front ends and 5 walking conditions:
  1. matched-subject aligned multivariate wave-shape cosine
  2. matched-subject maximum lagged wave-shape cosine, lags -3..+3 event slots
  3. matched-subject multivariate DTW distance
  4. matched-subject tree-branch-span wave cosine
Each is compared against subject-mismatched label permutations.

Secondary tests
---------------
  * aligned H, D, and lambda correlations
  * pivot-position Jaccard
  * group-average EEG/gait TMA wave similarity
  * baseline->adaptation delta-wave similarity
  * within-frame circular phase-rotation nulls for group-average wave similarity

No TMA rule is changed to improve a result.
"""

import io
import json
import math
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tone_metric.engine import recursive_integer_structure, _next_seq_anchor
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tma_cross_domain_frozen import quantize_phases
from tma_eeg_gait_multifrontend import (
    SUBJECTS, CONDITIONS, FRONTENDS, REPO_RAW,
    get_text, parse_events, condition_cycles, fdt_window,
    eeg_preprocess, frames_from_score, gait_frames
)

SEED = 20260927
PERM_B = 5000
PHASE_B = 200
F = 8
GRID = 256
LAGS = tuple(range(-3, 4))
OUT = Path("research/tma_eeg_gait_wave_shape_results")
OUT.mkdir(parents=True, exist_ok=True)

LOG=[]

def log(x):
    s=str(x)
    print(s, flush=True)
    LOG.append(s)

def analyze_wave(frames):
    """Exact frozen universal TMA wave, plus event-aligned derivatives."""
    if len(frames) != F:
        raise ValueError(f"need {F} frames")
    qframes=[quantize_phases(x) for x in frames]
    rows=[]
    for fi,phs in enumerate(qframes):
        for q in phs:
            rows.append((Fraction(fi)+q,fi,q))
    rows.sort(key=lambda z:z[0])
    unique=sorted({t for t,_,_ in rows})
    actual=set(unique)
    levels={t:set() for t in unique}

    max_pos=F+1
    top_end=_next_seq_anchor(2,max_pos)
    point_layers={};interval_levels={}
    recursive_integer_structure(1,top_end,2,1,point_layers,interval_levels)
    point_layers={p:ls for p,ls in point_layers.items() if p<=max_pos}
    interval_levels={p:l for p,l in interval_levels.items() if p<max_pos}

    for pos,ls in point_layers.items():
        t=Fraction(pos-1)
        if t in actual:
            levels[t].update(int(x) for x in ls)

    by_interval=defaultdict(list)
    for t in unique:
        k=int(math.floor(float(t)))
        if 0<=k<F:
            by_interval[k].append(t)

    def refine(a,b,ev,parent,guard=0):
        if guard>20:
            raise RuntimeError("recursive guard")
        interior=[t for t in ev if a<t<b]
        if not interior:
            return
        m=(a+b)/2
        lvl=int(parent)+1
        for t in (a,m,b):
            if t in actual:
                levels[t].add(lvl)
        left=[t for t in ev if a<=t<=m]
        right=[t for t in ev if m<=t<=b]
        if any(a<t<m for t in left):
            refine(a,m,left,lvl,guard+1)
        if any(m<t<b for t in right):
            refine(m,b,right,lvl,guard+1)

    for i in range(F):
        a,b=Fraction(i),Fraction(i+1)
        ev=by_interval.get(i,[])
        if any(a<t<b for t in ev):
            refine(a,b,ev,int(interval_levels.get(i+1,1)))

    wave=[]
    for onset,fi,q in rows:
        Kset=sorted(int(x) for x in levels[onset] if int(x)>0)
        if not Kset:
            raise RuntimeError(f"unresolved event {onset}")
        wave.append({
            "segment_index":0,
            "measure_index":fi,
            "measure_number":str(fi+1),
            "onset_quarter":str(onset),
            "offset_in_measure_quarter":str(q),
            "levels":Kset,
            "height":max(Kset),
            "density":len(Kset),
            "lowest_level":min(Kset),
            "attack":True,
            "parenthetical":False,
        })

    H=np.asarray([w["height"] for w in wave],float)
    D=np.asarray([w["density"] for w in wave],float)
    L=np.asarray([w["lowest_level"] for w in wave],float)
    onset=np.asarray([float(Fraction(w["onset_quarter"])) for w in wave],float)

    piv=build_pivot_profile(wave)
    onset_to_idx={str(w["onset_quarter"]):i for i,w in enumerate(wave)}
    pivot_depth=np.zeros(len(wave),float)
    pivot_binary=np.zeros(len(wave),float)
    for p in piv:
        key=str(p.get("pivot_onset_quarter"))
        idx=onset_to_idx.get(key)
        if idx is not None:
            pivot_binary[idx]=1.0
            pivot_depth[idx]=float(p.get("drop_depth",0))

    tree=build_tree_profile(wave)
    tree_span=np.zeros(len(wave),float)
    tree_time=np.zeros(len(wave),float)
    for b in tree.get("branches",[]) or []:
        j=int(b["target_node_index"])
        i=int(b["source_node_index"])
        if 0<=j<len(wave):
            tree_span[j]=float(j-i)
            tree_time[j]=float(Fraction(str(b["distance_quarter"])))

    return {
        "H":H,"D":D,"L":L,
        "onset":onset,
        "pivot_binary":pivot_binary,
        "pivot_depth":pivot_depth,
        "tree_span":tree_span,
        "tree_time":tree_time,
    }

def safe_pearson(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return np.nan
    return float(stats.pearsonr(a,b).statistic)

def safe_spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return np.nan
    return float(stats.spearmanr(a,b).statistic)

def zshape(x):
    x=np.asarray(x,float)
    s=np.std(x)
    if s<1e-12:
        return np.zeros_like(x)
    return (x-np.mean(x))/s

def combined_shape(w):
    return np.column_stack([zshape(w["H"]),zshape(w["D"]),zshape(w["L"])])

def cosine_flat(a,b):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else np.nan

def tree_cosine(a,b):
    return cosine_flat(zshape(a["tree_span"]),zshape(b["tree_span"]))

def pivot_jaccard(a,b):
    x=np.asarray(a["pivot_binary"])>0
    y=np.asarray(b["pivot_binary"])>0
    u=np.sum(x|y)
    return float(np.sum(x&y)/u) if u>0 else np.nan

def aligned_metrics(g,e):
    G=combined_shape(g); E=combined_shape(e)
    return {
        "shape_cosine":cosine_flat(G,E),
        "H_pearson":safe_pearson(g["H"],e["H"]),
        "H_spearman":safe_spearman(g["H"],e["H"]),
        "D_pearson":safe_pearson(g["D"],e["D"]),
        "D_spearman":safe_spearman(g["D"],e["D"]),
        "L_pearson":safe_pearson(g["L"],e["L"]),
        "L_spearman":safe_spearman(g["L"],e["L"]),
        "tree_cosine":tree_cosine(g,e),
        "pivot_jaccard":pivot_jaccard(g,e),
    }

def lagged_shape(g,e):
    G=combined_shape(g); E=combined_shape(e)
    best=-np.inf; best_lag=0
    for lag in LAGS:
        if lag<0:
            a=G[-lag:]; b=E[:len(E)+lag]
        elif lag>0:
            a=G[:-lag]; b=E[lag:]
        else:
            a=G; b=E
        if len(a)<12:
            continue
        c=cosine_flat(a,b)
        if np.isfinite(c) and c>best:
            best=c; best_lag=lag
    return float(best) if np.isfinite(best) else np.nan, int(best_lag)

def dtw_distance(g,e,band=3):
    G=combined_shape(g); E=combined_shape(e)
    n,m=len(G),len(E)
    inf=1e100
    dp=np.full((n+1,m+1),inf,float)
    steps=np.zeros((n+1,m+1),int)
    dp[0,0]=0.0
    for i in range(1,n+1):
        j0=max(1,i-band); j1=min(m,i+band)
        for j in range(j0,j1+1):
            cost=float(np.linalg.norm(G[i-1]-E[j-1]))
            opts=[(dp[i-1,j],steps[i-1,j]),(dp[i,j-1],steps[i,j-1]),(dp[i-1,j-1],steps[i-1,j-1])]
            k=int(np.argmin([x[0] for x in opts]))
            dp[i,j]=cost+opts[k][0]
            steps[i,j]=opts[k][1]+1
    if not np.isfinite(dp[n,m]) or steps[n,m]==0:
        return np.nan
    return float(dp[n,m]/steps[n,m])

def pair_metric(g,e,metric):
    if metric=="shape_cosine":
        return cosine_flat(combined_shape(g),combined_shape(e))
    if metric=="maxlag_cosine":
        return lagged_shape(g,e)[0]
    if metric=="dtw_distance":
        return dtw_distance(g,e)
    if metric=="tree_cosine":
        return tree_cosine(g,e)
    raise KeyError(metric)

def permutation_match_test(waves, subjects, cond, frontend, metric, B=PERM_B):
    n=len(subjects)
    M=np.full((n,n),np.nan,float)
    for i,s1 in enumerate(subjects):
        g=waves[(s1,cond,"gait")]
        for j,s2 in enumerate(subjects):
            e=waves[(s2,cond,frontend)]
            M[i,j]=pair_metric(g,e,metric)
    obs=float(np.nanmean(np.diag(M)))
    off=M[~np.eye(n,dtype=bool)]
    mismatch=float(np.nanmean(off))
    rng=np.random.default_rng(SEED+sum(map(ord,cond+frontend+metric)))
    null=[]
    for _ in range(B):
        p=rng.permutation(n)
        null.append(float(np.nanmean(M[np.arange(n),p])))
    null=np.asarray(null,float)
    if metric=="dtw_distance":
        pval=float((1+np.sum(null<=obs))/(B+1))
    else:
        pval=float((1+np.sum(null>=obs))/(B+1))
    return {
        "condition":cond,"frontend":frontend,"metric":metric,"n":n,
        "matched_mean":obs,"mismatched_mean":mismatch,
        "matched_minus_mismatched":obs-mismatch,
        "permutation_p":pval,
    }

def group_mean_wave(waves,subjects,cond,mod):
    keys=("H","D","L","tree_span")
    return {k:np.mean(np.stack([waves[(s,cond,mod)][k] for s in subjects]),axis=0) for k in keys}

def group_shape_metrics(g,e):
    out={
        "shape_cosine":cosine_flat(
            np.column_stack([zshape(g["H"]),zshape(g["D"]),zshape(g["L"])]),
            np.column_stack([zshape(e["H"]),zshape(e["D"]),zshape(e["L"])])
        ),
        "H_pearson":safe_pearson(g["H"],e["H"]),
        "D_pearson":safe_pearson(g["D"],e["D"]),
        "L_pearson":safe_pearson(g["L"],e["L"]),
        "tree_cosine":cosine_flat(zshape(g["tree_span"]),zshape(e["tree_span"])),
    }
    return out

def rotate_frames(frames,rng):
    out=[]
    for ph in frames:
        shift=float(rng.random())
        out.append(np.sort(np.mod(np.asarray(ph,float)+shift,1.0)))
    return out

def phase_group_null(waves,frame_store,subjects,cond,frontend,B=PHASE_B):
    gmean=group_mean_wave(waves,subjects,cond,"gait")
    emean=group_mean_wave(waves,subjects,cond,frontend)
    obs=group_shape_metrics(gmean,emean)
    rng=np.random.default_rng(SEED+500000+sum(map(ord,cond+frontend)))
    null={k:[] for k in obs}
    for b in range(B):
        ew=[]
        for s in subjects:
            fr=rotate_frames(frame_store[(s,cond,frontend)],rng)
            ew.append(analyze_wave(fr))
        em={
            k:np.mean(np.stack([w[k] for w in ew]),axis=0)
            for k in ("H","D","L","tree_span")
        }
        mm=group_shape_metrics(gmean,em)
        for k,v in mm.items():
            null[k].append(v)
    out={"condition":cond,"frontend":frontend,"n":len(subjects),"B":B}
    for k,v in obs.items():
        arr=np.asarray(null[k],float)
        arr=arr[np.isfinite(arr)]
        out[k]=float(v) if np.isfinite(v) else np.nan
        out[k+"_null_mean"]=float(np.nanmean(arr)) if len(arr) else np.nan
        out[k+"_null_sd"]=float(np.nanstd(arr,ddof=1)) if len(arr)>1 else np.nan
        out[k+"_phase_p"]=float((1+np.sum(arr>=v))/(len(arr)+1)) if np.isfinite(v) and len(arr) else np.nan
    return out

def delta_wave(w1,w0):
    return {k:np.asarray(w1[k],float)-np.asarray(w0[k],float) for k in ("H","D","L","tree_span")}

def scale_delta_pair_sets(Gs,Es):
    # Equalize H/D/lambda units using a pooled SD over all subjects and event slots.
    allv=[]
    for w in Gs+Es:
        allv.append(np.column_stack([w["H"],w["D"],w["L"]]))
    A=np.concatenate(allv,axis=0)
    sd=np.std(A,axis=0)
    sd=np.where(sd<1e-12,1.0,sd)
    def tr(w):
        return np.column_stack([w["H"]/sd[0],w["D"]/sd[1],w["L"]/sd[2]])
    return [tr(w) for w in Gs],[tr(w) for w in Es],sd

def delta_match_test(waves,subjects,cond,frontend,B=PERM_B):
    Gs=[delta_wave(waves[(s,cond,"gait")],waves[(s,"baseline","gait")]) for s in subjects]
    Es=[delta_wave(waves[(s,cond,frontend)],waves[(s,"baseline",frontend)]) for s in subjects]
    GA,EA,sd=scale_delta_pair_sets(Gs,Es)
    n=len(subjects)
    M=np.full((n,n),np.nan,float)
    for i in range(n):
        for j in range(n):
            M[i,j]=cosine_flat(GA[i],EA[j])
    obs=float(np.nanmean(np.diag(M)))
    mismatch=float(np.nanmean(M[~np.eye(n,dtype=bool)]))
    rng=np.random.default_rng(SEED+700000+sum(map(ord,cond+frontend)))
    null=[]
    for _ in range(B):
        p=rng.permutation(n)
        null.append(float(np.nanmean(M[np.arange(n),p])))
    arr=np.asarray(null,float)
    p=float((1+np.sum(arr>=obs))/(B+1))
    return {
        "condition":cond,"frontend":frontend,"n":n,
        "matched_delta_cosine":obs,
        "mismatched_delta_cosine":mismatch,
        "difference":obs-mismatch,
        "permutation_p":p,
        "scale_H":float(sd[0]),"scale_D":float(sd[1]),"scale_L":float(sd[2]),
    }

def bh(rows,pkey,qkey):
    vals=np.asarray([r.get(pkey,np.nan) for r in rows],float)
    q=np.full(len(vals),np.nan)
    good=np.flatnonzero(np.isfinite(vals))
    if len(good):
        pv=vals[good]; order=np.argsort(pv); ranked=pv[order]; m=len(ranked)
        qq=np.minimum.accumulate((ranked*m/np.arange(1,m+1))[::-1])[::-1]
        qq=np.minimum(qq,1.0)
        q[good[order]]=qq
    for r,v in zip(rows,q):
        r[qkey]=float(v) if np.isfinite(v) else np.nan

def main():
    t0=time.time()
    waves={}
    frame_store={}
    failures={}
    event_rows=[]

    for si,s in enumerate(SUBJECTS,1):
        try:
            evtxt=get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_events.tsv")
            meta=json.loads(get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_eeg.json"))
            ch=pd.read_csv(io.StringIO(get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_channels.tsv")),sep="\t")
            fs=float(meta["SamplingFrequency"])
            dur=float(meta["RecordingDuration"])
            n_total=len(ch)
            cc=condition_cycles(parse_events(evtxt))
            if any(c not in cc for c in CONDITIONS):
                raise RuntimeError("missing condition cycle sample")

            for cond in CONDITIONS:
                cyc=cc[cond]
                gf=gait_frames(cyc)
                frame_store[(s,cond,"gait")]=gf
                gw=analyze_wave(gf)
                waves[(s,cond,"gait")]=gw

                lo=cyc[0]["a"]-1.5; hi=cyc[-1]["b"]+1.5
                raw,tbase=fdt_window(s,lo,hi,fs,n_total,dur)
                scores=eeg_preprocess(raw,ch,fs)
                for fe,score in scores.items():
                    fr=frames_from_score(score,cyc,tbase,fs)
                    frame_store[(s,cond,fe)]=fr
                    waves[(s,cond,fe)]=analyze_wave(fr)

                for mod in ("gait",*FRONTENDS.keys()):
                    w=waves[(s,cond,mod)]
                    for i in range(len(w["H"])):
                        event_rows.append({
                            "subject":s,"condition":cond,"modality":mod,"event_index":i,
                            "H":w["H"][i],"D":w["D"][i],"lambda":w["L"][i],
                            "pivot":w["pivot_binary"][i],"pivot_depth":w["pivot_depth"][i],
                            "tree_span":w["tree_span"][i],"tree_time":w["tree_time"][i],
                        })
            log(f"{si}/{len(SUBJECTS)} {s}: OK")
        except Exception as e:
            failures[s]=repr(e)
            log(f"{si}/{len(SUBJECTS)} {s}: FAIL {e!r}")

    complete=[
        s for s in SUBJECTS
        if all((s,c,m) in waves for c in CONDITIONS for m in ("gait",*FRONTENDS.keys()))
    ]
    if len(complete)<20:
        raise RuntimeError(f"only {len(complete)} complete subjects: {failures}")
    pd.DataFrame(event_rows).to_csv(OUT/"event_level_waves.csv",index=False)

    # Per-subject aligned descriptive correlations.
    subj=[]
    for c in CONDITIONS:
        for fe in FRONTENDS:
            for s in complete:
                m=aligned_metrics(waves[(s,c,"gait")],waves[(s,c,fe)])
                ml,lag=lagged_shape(waves[(s,c,"gait")],waves[(s,c,fe)])
                m.update({"subject":s,"condition":c,"frontend":fe,
                          "maxlag_cosine":ml,"best_lag":lag,
                          "dtw_distance":dtw_distance(waves[(s,c,"gait")],waves[(s,c,fe)])})
                subj.append(m)
    pd.DataFrame(subj).to_csv(OUT/"subject_wave_correlations.csv",index=False)

    # Primary matched-vs-mismatched tests.
    primary=[]
    for c in CONDITIONS:
        for fe in FRONTENDS:
            for metric in ("shape_cosine","maxlag_cosine","dtw_distance","tree_cosine"):
                primary.append(permutation_match_test(waves,complete,c,fe,metric))
    # FDR by metric family across 25 condition x frontend tests.
    for metric in ("shape_cosine","maxlag_cosine","dtw_distance","tree_cosine"):
        rows=[r for r in primary if r["metric"]==metric]
        bh(rows,"permutation_p","q_fdr")
    pd.DataFrame(primary).to_csv(OUT/"matched_vs_mismatched_primary.csv",index=False)

    # Secondary aligned H/D/lambda and pivot Jaccard matched-vs-mismatched.
    secondary=[]
    secmetrics=("H_pearson","D_pearson","L_pearson","pivot_jaccard")
    for c in CONDITIONS:
        for fe in FRONTENDS:
            n=len(complete)
            for met in secmetrics:
                M=np.full((n,n),np.nan)
                for i,s1 in enumerate(complete):
                    g=waves[(s1,c,"gait")]
                    for j,s2 in enumerate(complete):
                        e=waves[(s2,c,fe)]
                        am=aligned_metrics(g,e)
                        M[i,j]=am[met]
                obs=float(np.nanmean(np.diag(M)))
                mis=float(np.nanmean(M[~np.eye(n,dtype=bool)]))
                rng=np.random.default_rng(SEED+900000+sum(map(ord,c+fe+met)))
                null=[]
                for _ in range(PERM_B):
                    p=rng.permutation(n)
                    null.append(float(np.nanmean(M[np.arange(n),p])))
                arr=np.asarray(null,float)
                secondary.append({
                    "condition":c,"frontend":fe,"metric":met,"n":n,
                    "matched_mean":obs,"mismatched_mean":mis,
                    "matched_minus_mismatched":obs-mis,
                    "permutation_p":float((1+np.sum(arr>=obs))/(PERM_B+1)),
                })
    for met in secmetrics:
        rows=[r for r in secondary if r["metric"]==met]
        bh(rows,"permutation_p","q_fdr")
    pd.DataFrame(secondary).to_csv(OUT/"matched_vs_mismatched_secondary.csv",index=False)

    # Group-average wave shape and phase-rotation null.
    group=[]
    for c in CONDITIONS:
        for fe in FRONTENDS:
            row=phase_group_null(waves,frame_store,complete,c,fe)
            group.append(row)
            log(f"phase null {c} {fe}: shape={row['shape_cosine']:.4f} p={row['shape_cosine_phase_p']:.4g}")
    # FDR for each group-wave metric across 25 tests.
    for met in ("shape_cosine","H_pearson","D_pearson","L_pearson","tree_cosine"):
        bh(group,met+"_phase_p",met+"_phase_q")
    pd.DataFrame(group).to_csv(OUT/"group_mean_wave_phase_null.csv",index=False)

    # Baseline -> adaptation delta wave correspondence.
    delta=[]
    for c in CONDITIONS:
        if c=="baseline":
            continue
        for fe in FRONTENDS:
            delta.append(delta_match_test(waves,complete,c,fe))
    bh(delta,"permutation_p","q_fdr")
    pd.DataFrame(delta).to_csv(OUT/"delta_wave_correspondence.csv",index=False)

    # Aggregate condition/frontend means from per-subject aligned metrics.
    sdf=pd.DataFrame(subj)
    agg=(sdf.groupby(["condition","frontend"],as_index=False)
         .agg(
             n=("subject","nunique"),
             shape_cosine_mean=("shape_cosine","mean"),
             shape_cosine_sd=("shape_cosine","std"),
             maxlag_cosine_mean=("maxlag_cosine","mean"),
             maxlag_cosine_sd=("maxlag_cosine","std"),
             H_pearson_mean=("H_pearson","mean"),
             D_pearson_mean=("D_pearson","mean"),
             L_pearson_mean=("L_pearson","mean"),
             tree_cosine_mean=("tree_cosine","mean"),
             pivot_jaccard_mean=("pivot_jaccard","mean"),
             dtw_mean=("dtw_distance","mean"),
         ))
    agg.to_csv(OUT/"descriptive_means.csv",index=False)

    def best(rows,pkey,n=20):
        return sorted([r for r in rows if np.isfinite(r.get(pkey,np.nan))],key=lambda r:r[pkey])[:n]

    summary={
        "dataset":"OpenNeuro ds004475 / NEMAR on004475",
        "seed":SEED,
        "n_complete_subjects":len(complete),
        "subjects":complete,
        "failures":failures,
        "design":{
            "same_physical_strides":True,
            "frames":8,
            "events_per_frame":3,
            "wave_points":24,
            "frontends":FRONTENDS,
            "conditions":list(CONDITIONS),
            "lags":list(LAGS),
            "primary_metrics":["shape_cosine","maxlag_cosine","dtw_distance","tree_cosine"],
            "subject_permutations":PERM_B,
            "phase_rotations":PHASE_B,
        },
        "best_primary":best(primary,"permutation_p",30),
        "best_group_phase_null":best(group,"shape_cosine_phase_p",25),
        "best_delta_wave":best(delta,"permutation_p",20),
        "runtime_s":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
    log(json.dumps(summary,indent=2))
    log(f"runtime_s={time.time()-t0:.1f}")

if __name__=="__main__":
    main()
