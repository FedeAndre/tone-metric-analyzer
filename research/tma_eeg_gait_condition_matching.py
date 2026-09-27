#!/usr/bin/env python3
from __future__ import annotations

"""
Condition-matching audit for scaffold-residual EEG<->gait TMA waves.

Asks whether the group-average residual TMA wave for a gait condition is most
similar to the EEG residual TMA wave from the *same* experimental condition.
Uses exact permutation over all 5! condition relabelings.
"""

from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd

OUT=Path("research/tma_eeg_gait_wave_shape_results")
CONDS=("baseline","early_abrupt","late_abrupt","early_gradual","late_gradual")
FRONTENDS=("theta_global","alpha_global","beta_global","mu_central","beta_central")

def cosine(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else np.nan

def main():
    df=pd.read_csv(OUT/"event_level_waves.csv")
    nd=pd.read_csv(OUT/"universal_scaffold_wave_null.csv")
    mu={k:nd[nd.variable==k].sort_values("event_index").null_mean.to_numpy(float) for k in ("H","D","lambda")}
    sd={k:nd[nd.variable==k].sort_values("event_index").null_sd.to_numpy(float) for k in ("H","D","lambda")}
    for k in sd:sd[k]=np.where(sd[k]<1e-8,1,sd[k])

    def group_vec(cond,mod):
        q=df[(df.condition==cond)&(df.modality==mod)]
        subjects=sorted(q.subject.unique())
        vv=[]
        for s in subjects:
            r=q[q.subject==s].sort_values("event_index")
            if len(r)!=24:continue
            arr=np.column_stack([
                (r.H.to_numpy(float)-mu["H"])/sd["H"],
                (r.D.to_numpy(float)-mu["D"])/sd["D"],
                (r["lambda"].to_numpy(float)-mu["lambda"])/sd["lambda"],
            ])
            vv.append(arr)
        return np.mean(np.stack(vv),axis=0),len(vv)

    rows=[];summ=[]
    perms=list(itertools.permutations(range(len(CONDS))))
    for fe in FRONTENDS:
        G=[];E=[]
        for c in CONDS:
            g,n=group_vec(c,"gait");e,n2=group_vec(c,fe)
            G.append(g);E.append(e)
        M=np.array([[cosine(G[i],E[j]) for j in range(5)] for i in range(5)])
        for i,cg in enumerate(CONDS):
            for j,ce in enumerate(CONDS):
                rows.append({"frontend":fe,"gait_condition":cg,"eeg_condition":ce,"cosine":M[i,j]})
        obs=float(np.mean(np.diag(M)))
        vals=[]
        for p in perms:
            vals.append(float(np.mean([M[i,p[i]] for i in range(5)])))
        vals=np.asarray(vals)
        exact_p=float(np.mean(vals>=obs-1e-15))
        diag=np.diag(M);off=M[~np.eye(5,dtype=bool)]
        correct_top1=int(np.sum(np.argmax(M,axis=1)==np.arange(5)))
        # late-abrupt specificity
        i=CONDS.index("late_abrupt")
        la=float(M[i,i]); others=np.delete(M[i],i)
        la_rank=int(1+np.sum(M[i]>la))
        summ.append({
            "frontend":fe,
            "mean_same_condition_cosine":obs,
            "mean_cross_condition_cosine":float(np.mean(off)),
            "difference":float(obs-np.mean(off)),
            "exact_condition_permutation_p":exact_p,
            "correct_condition_top1_count":correct_top1,
            "late_abrupt_same_cosine":la,
            "late_abrupt_other_mean":float(np.mean(others)),
            "late_abrupt_rank_among_5":la_rank,
        })
    # BH across five frontends
    p=np.array([x["exact_condition_permutation_p"] for x in summ],float)
    order=np.argsort(p);m=len(p);qq=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    q=np.empty(m);q[order]=np.minimum(qq,1)
    for r,v in zip(summ,q):r["q_fdr"]=float(v)

    pd.DataFrame(rows).to_csv(OUT/"scaffold_residual_condition_matrix.csv",index=False)
    pd.DataFrame(summ).to_csv(OUT/"scaffold_residual_condition_matching.csv",index=False)
    result={"condition_matching":summ,"matrix_rows":rows}
    (OUT/"scaffold_residual_condition_summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"condition_matching":summ},indent=2))

if __name__=="__main__":
    main()
