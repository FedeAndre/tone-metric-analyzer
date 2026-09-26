#!/usr/bin/env python3
from __future__ import annotations

"""
Prespecified multi-front-end EEG<->gait TMA translation test on ds004475.

The TMA core is frozen. Only the EEG event-definition front end changes:
  1) theta_global 4-8 Hz, all EEG channels
  2) alpha_global 8-13 Hz, all EEG channels
  3) beta_global 13-30 Hz, all EEG channels (replication)
  4) mu_central 8-13 Hz, C3/C4 only
  5) beta_central 13-30 Hz, C3/C4 only

Each front end yields exactly three separated EEG events per LHS-to-LHS stride.
Gait uses RTO, RHS, LTO in the exact same strides.
Both modalities enter the same frozen 8-frame x 3-event x 256-bin TMA core.
"""

import io, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats

from tma_cross_domain_frozen import analyze_frames, FEATURES
from tma_eeg_gait_translation import (
    SUBJECTS, CONDITIONS, PRIMARY_CHANGE, REPO_RAW,
    get_text, parse_events, condition_cycles, fdt_window, gait_frames,
    top3_score_phases, bh_qvalues, rmsdist, pdist_matrix
)

SEED=20260926
RNG=np.random.default_rng(SEED+404)
PERM_B=5000
PHASE_B=300
OUT=Path("research/tma_eeg_gait_multifrontend_results")
OUT.mkdir(parents=True,exist_ok=True)

FRONTENDS={
    "theta_global":{"band":(4.0,8.0),"scope":"global"},
    "alpha_global":{"band":(8.0,13.0),"scope":"global"},
    "beta_global":{"band":(13.0,30.0),"scope":"global"},
    "mu_central":{"band":(8.0,13.0),"scope":"central"},
    "beta_central":{"band":(13.0,30.0),"scope":"central"},
}
LOG=[]
def log(x):
    s=str(x);print(s,flush=True);LOG.append(s)

def eeg_preprocess(raw,ch,fs):
    types=ch["type"].astype(str).str.upper().to_numpy()
    names=ch["name"].astype(str).to_numpy()
    eeg_abs=np.flatnonzero(types=="EEG")
    if len(eeg_abs)<16: raise RuntimeError("too few EEG channels")
    x=np.asarray(raw[:,eeg_abs],dtype=np.float64)
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
    x-=np.median(x,axis=1,keepdims=True)
    # central indices relative to EEG-only matrix
    lower=np.char.lower(names[eeg_abs].astype(str))
    c3=np.flatnonzero(np.char.endswith(lower,"c3"))
    c4=np.flatnonzero(np.char.endswith(lower,"c4"))
    central=np.unique(np.r_[c3,c4])
    if len(central)<2: raise RuntimeError(f"C3/C4 not found: {[names[eeg_abs][i] for i in central]}")
    scores={}
    cache={}
    for f,cfg in FRONTENDS.items():
        band=tuple(cfg["band"])
        if band not in cache:
            sos=signal.butter(4,[band[0],band[1]],btype="bandpass",fs=fs,output="sos")
            y=signal.sosfiltfilt(sos,x,axis=0)
            med=np.median(y,axis=0)
            mad=1.4826*np.median(np.abs(y-med),axis=0)
            finite=np.isfinite(mad)&(mad>np.nanmedian(mad)*1e-4)&(mad>1e-12)
            z=np.zeros_like(y)
            z[:,finite]=(y[:,finite]-med[finite])/mad[finite]
            z=np.clip(z,-8,8)
            env=np.abs(signal.hilbert(z,axis=0))
            cache[band]=(env,finite)
        env,finite=cache[band]
        if cfg["scope"]=="global":
            idx=np.flatnonzero(finite)
            if len(idx)<16:raise RuntimeError(f"{f}: too few usable global channels")
        else:
            idx=np.array([i for i in central if finite[i]],int)
            if len(idx)<2:raise RuntimeError(f"{f}: central channels unusable")
        scores[f]=np.sqrt(np.mean(env[:,idx]*env[:,idx],axis=1))
    return scores

def frames_from_score(score,cycles,raw_t0,fs):
    out=[]
    for c in cycles:
        i0=max(0,int(round((c["a"]-raw_t0)*fs)))
        i1=min(len(score),int(round((c["b"]-raw_t0)*fs)))
        if i1-i0<20:raise RuntimeError("short EEG stride")
        out.append(top3_score_phases(score[i0:i1]))
    return out

def common_space(rawdf,null):
    mu=null[FEATURES].mean();sd=null[FEATURES].std(ddof=1)
    feats=[f for f in FEATURES if np.isfinite(sd[f]) and sd[f]>1e-8]
    z=rawdf.copy()
    for f in feats:z[f]=(z[f]-mu[f])/sd[f]
    return z,mu,sd,feats

def identity_test(zdf,frontend,condition,feats,B=PERM_B):
    g=zdf[(zdf.frontend=="gait")&(zdf.condition==condition)].set_index("subject")
    e=zdf[(zdf.frontend==frontend)&(zdf.condition==condition)].set_index("subject")
    ids=sorted(set(g.index)&set(e.index)); G=g.loc[ids,feats].to_numpy(float);E=e.loc[ids,feats].to_numpy(float)
    D=np.sqrt(np.mean((G[:,None,:]-E[None,:,:])**2,axis=2))
    diag=np.diag(D);off=D[~np.eye(len(ids),dtype=bool)]
    top=float(np.mean(np.argmin(D,axis=1)==np.arange(len(ids))))
    ranks=[int(np.where(np.argsort(D[i])==i)[0][0])+1 for i in range(len(ids))]
    DG=pdist_matrix(G);DE=pdist_matrix(E);iu=np.triu_indices(len(ids),1)
    mantel=float(stats.spearmanr(DG[iu],DE[iu]).statistic)
    rng=np.random.default_rng(SEED+sum(map(ord,frontend+condition)))
    nd=[];nt=[];nm=[]
    for _ in range(B):
        p=rng.permutation(len(ids))
        nd.append(float(np.mean(D[np.arange(len(ids)),p])))
        nt.append(float(np.mean(np.argmin(D[:,p],axis=1)==np.arange(len(ids)))))
        q=DE[p][:,p]
        nm.append(float(stats.spearmanr(DG[iu],q[iu]).statistic))
    return {
        "frontend":frontend,"condition":condition,"n":len(ids),
        "matched_distance":float(np.mean(diag)),"mismatched_distance":float(np.mean(off)),
        "distance_ratio":float(np.mean(diag)/np.mean(off)),
        "distance_p":float((1+np.sum(np.asarray(nd)<=np.mean(diag)))/(B+1)),
        "top1":top,"chance_top1":1/len(ids),
        "top1_p":float((1+np.sum(np.asarray(nt)>=top))/(B+1)),
        "mean_true_rank":float(np.mean(ranks)),
        "mantel_rho":mantel,
        "mantel_p":float((1+np.sum(np.asarray(nm)>=mantel))/(B+1)),
    }

def change_test(zdf,frontend,condition,feats,B=PERM_B):
    def mat(mod,cond):
        return zdf[(zdf.frontend==mod)&(zdf.condition==cond)].set_index("subject")
    gb,gc=mat("gait","baseline"),mat("gait",condition)
    eb,ec=mat(frontend,"baseline"),mat(frontend,condition)
    ids=sorted(set(gb.index)&set(gc.index)&set(eb.index)&set(ec.index))
    G=gc.loc[ids,feats].to_numpy(float)-gb.loc[ids,feats].to_numpy(float)
    E=ec.loc[ids,feats].to_numpy(float)-eb.loc[ids,feats].to_numpy(float)
    D=np.sqrt(np.mean((G[:,None,:]-E[None,:,:])**2,axis=2))
    def cos(a,b):
        den=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/den) if den>0 else np.nan
    C=np.array([[cos(G[i],E[j]) for j in range(len(ids))] for i in range(len(ids))])
    diag=np.diag(D);off=D[~np.eye(len(ids),dtype=bool)]
    cdiag=np.diag(C);coff=C[~np.eye(len(ids),dtype=bool)]
    top=float(np.mean(np.argmin(D,axis=1)==np.arange(len(ids))))
    rng=np.random.default_rng(SEED+999+sum(map(ord,frontend+condition)))
    nd=[];nc=[];nt=[]
    for _ in range(B):
        p=rng.permutation(len(ids))
        nd.append(float(np.mean(D[np.arange(len(ids)),p])))
        nc.append(float(np.nanmean(C[np.arange(len(ids)),p])))
        nt.append(float(np.mean(np.argmin(D[:,p],axis=1)==np.arange(len(ids)))))
    return {
        "frontend":frontend,"condition":condition,"n":len(ids),
        "matched_change_distance":float(np.mean(diag)),
        "mismatched_change_distance":float(np.mean(off)),
        "distance_ratio":float(np.mean(diag)/np.mean(off)),
        "distance_p":float((1+np.sum(np.asarray(nd)<=np.mean(diag)))/(B+1)),
        "matched_change_cosine":float(np.nanmean(cdiag)),
        "mismatched_change_cosine":float(np.nanmean(coff)),
        "cosine_p":float((1+np.sum(np.asarray(nc)>=np.nanmean(cdiag)))/(B+1)),
        "top1":top,"chance_top1":1/len(ids),
        "top1_p":float((1+np.sum(np.asarray(nt)>=top))/(B+1)),
    },ids,G,E

def feature_corrs(zdf,frontend,condition,feats,change=False):
    rows=[]
    g=zdf[zdf.frontend=="gait"];e=zdf[zdf.frontend==frontend]
    for f in feats:
        if change:
            gb=g[g.condition=="baseline"].set_index("subject");gc=g[g.condition==condition].set_index("subject")
            eb=e[e.condition=="baseline"].set_index("subject");ec=e[e.condition==condition].set_index("subject")
            ids=sorted(set(gb.index)&set(gc.index)&set(eb.index)&set(ec.index))
            x=gc.loc[ids,f].to_numpy(float)-gb.loc[ids,f].to_numpy(float)
            y=ec.loc[ids,f].to_numpy(float)-eb.loc[ids,f].to_numpy(float)
        else:
            gg=g[g.condition==condition].set_index("subject");ee=e[e.condition==condition].set_index("subject")
            ids=sorted(set(gg.index)&set(ee.index))
            x=gg.loc[ids,f].to_numpy(float);y=ee.loc[ids,f].to_numpy(float)
        if np.std(x)<1e-12 or np.std(y)<1e-12:
            rho,p=np.nan,np.nan
        else:
            s=stats.spearmanr(x,y);rho,p=float(s.statistic),float(s.pvalue)
        rows.append({"frontend":frontend,"condition":condition,"change":bool(change),"feature":f,"n":len(ids),"rho":rho,"p":p})
    return rows

def phase_control(frame_store,zdf,frontend,mu,sd,feats,B=PHASE_B):
    obs,ids,G,_=change_test(zdf,frontend,PRIMARY_CHANGE,feats,B=1000)
    target=obs["matched_change_cosine"]
    rng=np.random.default_rng(SEED+12345+sum(map(ord,frontend)))
    vals=[]
    for b in range(B):
        cs=[]
        for i,s in enumerate(ids):
            def rot(fr):
                out=[]
                for ph in fr:
                    out.append(np.sort(np.mod(np.asarray(ph,float)+float(rng.random()),1.0)))
                return out
            vb=analyze_frames(rot(frame_store[(s,"baseline",frontend)]))
            vc=analyze_frames(rot(frame_store[(s,PRIMARY_CHANGE,frontend)]))
            de=np.array([(vc[f]-mu[f])/sd[f]-(vb[f]-mu[f])/sd[f] for f in feats],float)
            dg=G[i];den=np.linalg.norm(dg)*np.linalg.norm(de)
            if den>0:cs.append(float(np.dot(dg,de)/den))
        vals.append(float(np.mean(cs)))
    vals=np.asarray(vals)
    return {"frontend":frontend,"condition":PRIMARY_CHANGE,"B":B,"observed_cosine":target,
            "null_mean":float(np.mean(vals)),"null_sd":float(np.std(vals,ddof=1)),
            "p":float((1+np.sum(vals>=target))/(B+1))}

def main():
    t0=time.time()
    null=pd.read_csv("research/tma_cross_domain_frozen_results/universal_null_features.csv")
    rows=[];frame_store={};fail={}
    for si,s in enumerate(SUBJECTS,1):
        try:
            evtxt=get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_events.tsv")
            meta=json.loads(get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_eeg.json"))
            ch=pd.read_csv(io.StringIO(get_text(f"{REPO_RAW}/sub-{s}/eeg/sub-{s}_task-task_channels.tsv")),sep="\t")
            fs=float(meta["SamplingFrequency"]);dur=float(meta["RecordingDuration"]);n_total=len(ch)
            cc=condition_cycles(parse_events(evtxt))
            if any(c not in cc for c in CONDITIONS):raise RuntimeError("missing condition cycles")
            for cond in CONDITIONS:
                cyc=cc[cond];gf=gait_frames(cyc)
                frame_store[(s,cond,"gait")]=gf
                if not any((r.get("subject")==s and r.get("condition")==cond and r.get("frontend")=="gait") for r in rows):
                    rows.append({"subject":s,"condition":cond,"frontend":"gait",**analyze_frames(gf)})
                lo=cyc[0]["a"]-1.5;hi=cyc[-1]["b"]+1.5
                raw,tbase=fdt_window(s,lo,hi,fs,n_total,dur)
                scores=eeg_preprocess(raw,ch,fs)
                for fe,score in scores.items():
                    fr=frames_from_score(score,cyc,tbase,fs)
                    frame_store[(s,cond,fe)]=fr
                    rows.append({"subject":s,"condition":cond,"frontend":fe,**analyze_frames(fr)})
            log(f"{si}/{len(SUBJECTS)} {s}: OK")
        except Exception as e:
            fail[s]=repr(e);log(f"{si}/{len(SUBJECTS)} {s}: FAIL {e!r}")
    rawdf=pd.DataFrame(rows)
    rawdf.to_csv(OUT/"sample_features_raw.csv",index=False)
    expected={"gait",*FRONTENDS.keys()}
    complete=[]
    for s in SUBJECTS:
        q=rawdf[rawdf.subject==s]
        if all(((q.condition==c)&(q.frontend==m)).any() for c in CONDITIONS for m in expected):
            complete.append(s)
    rawdf=rawdf[rawdf.subject.isin(complete)].copy()
    if len(complete)<20:raise RuntimeError(f"too few complete subjects {len(complete)}")
    zdf,mu,sd,feats=common_space(rawdf,null)
    zdf.to_csv(OUT/"sample_signatures_z.csv",index=False)

    identity=[];changes=[];fc=[]
    for fe in FRONTENDS:
        for c in CONDITIONS:
            identity.append(identity_test(zdf,fe,c,feats))
            fc.extend(feature_corrs(zdf,fe,c,feats,change=False))
        for c in CONDITIONS:
            if c=="baseline":continue
            row,*_=change_test(zdf,fe,c,feats);changes.append(row)
            fc.extend(feature_corrs(zdf,fe,c,feats,change=True))

    # Global FDR within each inferential family.
    for key in ("distance_p","top1_p","mantel_p"):
        q=bh_qvalues([r[key] for r in identity])
        for r,qq in zip(identity,q):r[key.replace("_p","_q")]=float(qq) if np.isfinite(qq) else np.nan
    for key in ("distance_p","cosine_p","top1_p"):
        q=bh_qvalues([r[key] for r in changes])
        for r,qq in zip(changes,q):r[key.replace("_p","_q")]=float(qq) if np.isfinite(qq) else np.nan
    # Feature correlations: state and change are separate families, each over all frontends/conditions/features.
    for flag in (False,True):
        ids=[i for i,r in enumerate(fc) if r["change"]==flag]
        q=bh_qvalues([fc[i]["p"] for i in ids])
        for i,qq in zip(ids,q):fc[i]["q_global"]=float(qq) if np.isfinite(qq) else np.nan
    pd.DataFrame(identity).to_csv(OUT/"identity_tests.csv",index=False)
    pd.DataFrame(changes).to_csv(OUT/"change_tests.csv",index=False)
    pd.DataFrame(fc).to_csv(OUT/"feature_correlations.csv",index=False)

    phase=[]
    for fe in FRONTENDS:
        row=phase_control(frame_store,zdf,fe,mu,sd,feats)
        phase.append(row);log(f"phase control {fe}: {row}")
    q=bh_qvalues([r["p"] for r in phase])
    for r,qq in zip(phase,q):r["q_fdr"]=float(qq)
    pd.DataFrame(phase).to_csv(OUT/"phase_rotation_controls.csv",index=False)

    # Direct paired gait effects retained for comparison; EEG effects per frontend.
    eff=[]
    for m in ["gait",*FRONTENDS.keys()]:
        for c in CONDITIONS:
            if c=="baseline":continue
            b=rawdf[(rawdf.frontend==m)&(rawdf.condition=="baseline")].set_index("subject")
            d=rawdf[(rawdf.frontend==m)&(rawdf.condition==c)].set_index("subject")
            ids=sorted(set(b.index)&set(d.index))
            for f in feats:
                x=d.loc[ids,f].to_numpy(float)-b.loc[ids,f].to_numpy(float)
                if np.std(x,ddof=1)<1e-12:p=np.nan;dz=np.nan
                else:
                    p=float(stats.ttest_1samp(x,0).pvalue);dz=float(np.mean(x)/np.std(x,ddof=1))
                eff.append({"frontend":m,"condition":c,"feature":f,"n":len(ids),"mean_change":float(np.mean(x)),"paired_dz":dz,"p":p})
    # FDR per frontend across all condition-feature effects.
    for m in ["gait",*FRONTENDS.keys()]:
        ids=[i for i,r in enumerate(eff) if r["frontend"]==m]
        q=bh_qvalues([eff[i]["p"] for i in ids])
        for i,qq in zip(ids,q):eff[i]["q_fdr"]=float(qq) if np.isfinite(qq) else np.nan
    pd.DataFrame(eff).to_csv(OUT/"condition_effects.csv",index=False)

    summary={
        "dataset":"OpenNeuro ds004475 / NEMAR on004475",
        "seed":SEED,"n_complete_subjects":len(complete),"failed_subjects":fail,
        "frontends":FRONTENDS,
        "frozen_tma_core":{"frames":8,"events_per_frame":3,"grid":256,"arity":2,"features":feats},
        "identity_tests":identity,"change_tests":changes,"phase_rotation_controls":phase,
        "best_state_feature_correlations":sorted([r for r in fc if not r["change"] and np.isfinite(r["p"])],key=lambda r:r["p"])[:20],
        "best_change_feature_correlations":sorted([r for r in fc if r["change"] and np.isfinite(r["p"])],key=lambda r:r["p"])[:20],
        "runtime_s":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
    log(json.dumps(summary,indent=2))
    log(f"runtime_s={time.time()-t0:.1f}")

if __name__=="__main__":
    main()

# workflow trigger
