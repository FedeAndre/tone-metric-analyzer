#!/usr/bin/env python3
from __future__ import annotations

"""
Universal-scaffold residual audit for the ds004475 EEG<->gait TMA wave analysis.

Removes the expected event-index wave imposed by the common 8-frame dyadic TMA
scaffold.  This directly tests whether gait and EEG share *deviations from* the
generic TMA scaffold rather than merely sharing the scaffold itself.
"""

from pathlib import Path
import json, math
import numpy as np
import pandas as pd

from tma_eeg_gait_wave_shape import analyze_wave, cosine_flat

SEED=20260927
NULL_B=5000
PERM_B=10000
OUT=Path("research/tma_eeg_gait_wave_shape_results")
FRONTENDS=("theta_global","alpha_global","beta_global","mu_central","beta_central")
CONDITIONS=("baseline","early_abrupt","late_abrupt","early_gradual","late_gradual")
VARS=("H","D","lambda","tree_span")

def bh(rows):
    p=np.asarray([r["p"] for r in rows],float);q=np.full(len(p),np.nan)
    good=np.flatnonzero(np.isfinite(p))
    if len(good):
        pv=p[good];order=np.argsort(pv);ranked=pv[order];m=len(ranked)
        qq=np.minimum.accumulate((ranked*m/np.arange(1,m+1))[::-1])[::-1]
        q[good[order]]=np.minimum(qq,1.0)
    for r,v in zip(rows,q):r["q_fdr"]=float(v) if np.isfinite(v) else np.nan

def main():
    df=pd.read_csv(OUT/"event_level_waves.csv")
    subjects=sorted(df.subject.unique())
    rng=np.random.default_rng(SEED)

    # Generic universal scaffold: 3 uniform events in each of 8 frames.
    null={k:[] for k in VARS}
    for _ in range(NULL_B):
        frames=[np.sort(rng.uniform(1/256,255/256,3)) for _ in range(8)]
        w=analyze_wave(frames)
        null["H"].append(w["H"]);null["D"].append(w["D"])
        null["lambda"].append(w["L"]);null["tree_span"].append(w["tree_span"])
    mu={k:np.mean(np.stack(v),axis=0) for k,v in null.items()}
    sd={k:np.std(np.stack(v),axis=0,ddof=1) for k,v in null.items()}
    for k in VARS:sd[k]=np.where(sd[k]<1e-8,1.0,sd[k])

    nullrows=[]
    for k in VARS:
        for i in range(24):
            nullrows.append({"variable":k,"event_index":i,"null_mean":mu[k][i],"null_sd":sd[k][i]})
    pd.DataFrame(nullrows).to_csv(OUT/"universal_scaffold_wave_null.csv",index=False)

    # Subject residual vectors.
    vec={}
    for (s,c,m),q in df.groupby(["subject","condition","modality"]):
        q=q.sort_values("event_index")
        if len(q)!=24:continue
        parts=[]
        for k in ("H","D","lambda"):
            x=q[k].to_numpy(float)
            parts.append((x-mu[k])/sd[k])
        vec[(s,c,m)]=np.column_stack(parts)

    rows=[]
    group=[]
    for c in CONDITIONS:
        for fe in FRONTENDS:
            ids=[s for s in subjects if (s,c,"gait") in vec and (s,c,fe) in vec]
            n=len(ids)
            M=np.full((n,n),np.nan)
            for i,s1 in enumerate(ids):
                for j,s2 in enumerate(ids):
                    M[i,j]=cosine_flat(vec[(s1,c,"gait")],vec[(s2,c,fe)])
            obs=float(np.mean(np.diag(M)));mis=float(np.mean(M[~np.eye(n,dtype=bool)]))
            rr=np.random.default_rng(SEED+sum(map(ord,c+fe)))
            nullp=[]
            for _ in range(PERM_B):
                p=rr.permutation(n);nullp.append(float(np.mean(M[np.arange(n),p])))
            arr=np.asarray(nullp)
            pval=float((1+np.sum(arr>=obs))/(PERM_B+1))
            rows.append({"condition":c,"frontend":fe,"n":n,"matched_residual_cosine":obs,
                         "mismatched_residual_cosine":mis,"difference":obs-mis,"p":pval})

            G=np.mean(np.stack([vec[(s,c,"gait")] for s in ids]),axis=0)
            E=np.mean(np.stack([vec[(s,c,fe)] for s in ids]),axis=0)
            gcos=cosine_flat(G,E)
            # Bootstrap/permutation subject labels cannot change group means. Use
            # sign-flipped subject residual contributions as a zero-coupling null.
            pair=np.stack([np.sum(vec[(s,c,"gait")]*vec[(s,c,fe)]) for s in ids])
            den=np.linalg.norm(G)*np.linalg.norm(E)
            rr2=np.random.default_rng(SEED+100000+sum(map(ord,c+fe)))
            # More direct null: independently sign-flip each subject's EEG residual wave
            # before averaging, preserving its shape/amplitude but destroying common direction.
            nullg=[]
            Esubs=np.stack([vec[(s,c,fe)] for s in ids])
            for _ in range(PERM_B):
                signs=rr2.choice(np.array([-1.0,1.0]),size=n)[:,None,None]
                Er=np.mean(Esubs*signs,axis=0)
                nullg.append(cosine_flat(G,Er))
            aa=np.asarray(nullg)
            gp=float((1+np.sum(aa>=gcos))/(PERM_B+1))
            group.append({"condition":c,"frontend":fe,"n":n,
                          "group_residual_cosine":gcos,
                          "signflip_p":gp,
                          "null_mean":float(np.mean(aa)),"null_sd":float(np.std(aa,ddof=1))})

    bh(rows)
    # FDR group separately
    p=np.asarray([r["signflip_p"] for r in group],float);order=np.argsort(p);m=len(p)
    qq=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    q=np.empty(m);q[order]=np.minimum(qq,1)
    for r,v in zip(group,q):r["q_fdr"]=float(v)

    pd.DataFrame(rows).to_csv(OUT/"scaffold_residual_matched_tests.csv",index=False)
    pd.DataFrame(group).to_csv(OUT/"scaffold_residual_group_tests.csv",index=False)

    summary={
        "null_samples":NULL_B,"subject_permutations":PERM_B,
        "best_matched":sorted(rows,key=lambda r:r["p"])[:10],
        "best_group":sorted(group,key=lambda r:r["signflip_p"])[:10],
        "significant_matched":[r for r in rows if r["q_fdr"]<0.05],
        "significant_group":[r for r in group if r["q_fdr"]<0.05],
    }
    (OUT/"scaffold_residual_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
