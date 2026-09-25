from __future__ import annotations
import math
from fractions import Fraction
from collections import defaultdict
import numpy as np, pandas as pd
from scipy import stats
from tone_metric.engine import analyze
from tone_metric.models import Hit, MeasureInfo
from tone_metric.waves import build_wave_profile
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile

def project_offset(offset_s,duration_s,tol_s):
    if offset_s<=0: return Fraction(0),0,0.0
    raw=2.0*float(offset_s)/float(duration_s)
    for depth in range(0,22):
        den=2**depth
        q=Fraction(int(math.floor(raw*den+0.5)),den)
        err=abs(float(q)-raw)*duration_s/2.0
        if err<=tol_s+1e-12:
            return q,depth,1000.0*err
    q=Fraction(raw).limit_denominator(2**21)
    return q,-1,0.0

def analyze_event_frames(frames,tol_s=0.005,label_features=True):
    # frames: [{"duration_s":..., "events":[{"label":..., "offset_s":...},...]}]
    hits=[]; labels=[]; frame_ids=[]; proj=[]; measures=[]
    for i,fr in enumerate(frames):
        dur=float(fr["duration_s"])
        start=Fraction(2*i)
        measures.append(MeasureInfo(index=i,number=str(i+1),start=start,full_duration=Fraction(2),
            actual_duration=Fraction(2),pickup_shift=Fraction(0),numerator=2,denominator=4,
            implicit=False,opening_anacrusis=False))
        for ev in fr["events"]:
            off=float(ev["offset_s"])
            if off<0 or off>=dur: continue
            q,depth,err=project_offset(off,dur,tol_s)
            hits.append(Hit(onset=start+q,duration=Fraction(0),measure_index=i,measure_number=str(i+1),
                offset_in_measure=q,sources=[],canonical_recovered=False))
            labels.append(str(ev.get("label","EVENT"))); frame_ids.append(i)
            proj.append((depth,err,float(q)))
    order=sorted(range(len(hits)),key=lambda j:(hits[j].onset,j))
    hits=[hits[j] for j in order];labels=[labels[j] for j in order];frame_ids=[frame_ids[j] for j in order];proj=[proj[j] for j in order]
    if not hits: raise RuntimeError("no events")
    res=analyze(hits,measures)
    er=[e for seg in res["segments"] for e in seg["events"]]
    if len(er)!=len(hits): raise RuntimeError(f"engine event count mismatch {len(er)} != {len(hits)}")
    evdf=pd.DataFrame({
        "label":labels,"frame_index":frame_ids,
        "H":[int(x["tone_metric_height"]) for x in er],
        "D":[int(x["tone_metric_density"]) for x in er],
        "lambda":[int(x["lowest_tone_metric_level"]) for x in er],
        "levels":[";".join(map(str,x["tone_metric_levels"])) for x in er],
        "projection_depth":[x[0] for x in proj],
        "projection_error_ms":[x[1] for x in proj],
        "projected_phase":[x[2] for x in proj],
    })
    wave=build_wave_profile(res);piv=build_pivot_profile(wave);tree=build_tree_profile(wave);br=tree.get("branches",[])
    f={
        "n_frames":len(frames),"n_events":len(evdf),
        "mean_H":float(evdf.H.mean()),"sd_H":float(evdf.H.std(ddof=1)),
        "mean_D":float(evdf.D.mean()),"sd_D":float(evdf.D.std(ddof=1)),
        "mean_lambda":float(evdf["lambda"].mean()),"sd_lambda":float(evdf["lambda"].std(ddof=1)),
        "multilevel":float((evdf.D>1).mean()),"pivot_rate":float(len(piv)/len(evdf)),
        "compound_pivot_fraction":float(np.mean([p["compound"] for p in piv])) if piv else np.nan,
        "pivot_depth":float(np.mean([p["drop_depth"] for p in piv])) if piv else np.nan,
        "tree_span_events":float(np.mean([int(b["target_node_index"])-int(b["source_node_index"]) for b in br])) if br else np.nan,
        "tree_span_time":float(np.mean([float(Fraction(str(b["distance_quarter"]))) for b in br])) if br else np.nan,
        "tree_root_fraction":float(np.mean([bool(n["root"]) for n in tree.get("nodes",[])])) if tree.get("nodes") else np.nan,
        "projection_depth_mean":float(evdf.projection_depth.replace(-1,np.nan).mean()),
        "projection_error_ms_mean":float(evdf.projection_error_ms.mean()),
        "projection_error_ms_max":float(evdf.projection_error_ms.max()),
    }
    if label_features:
        for lab in sorted(evdf.label.unique()):
            z=evdf[evdf.label==lab]
            safe="".join(c if c.isalnum() else "_" for c in lab)
            f[f"{safe}_mean_H"]=float(z.H.mean());f[f"{safe}_mean_D"]=float(z.D.mean())
            f[f"{safe}_mean_lambda"]=float(z["lambda"].mean());f[f"{safe}_multilevel"]=float((z.D>1).mean())
    return f,evdf,piv,tree

def bh_adjust(p):
    p=np.asarray(p,float);n=len(p);o=np.argsort(p);q=np.empty(n,float);last=1.0
    for rank,idx in reversed(list(enumerate(o,start=1))):
        last=min(last,p[idx]*n/rank);q[idx]=last
    return np.minimum(q,1)

def hedges_g(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);n0=len(a);n1=len(b)
    if n0<2 or n1<2:return np.nan
    sp=((n0-1)*np.var(a,ddof=1)+(n1-1)*np.var(b,ddof=1))/(n0+n1-2)
    if sp<=0:return np.nan
    d=(np.mean(b)-np.mean(a))/math.sqrt(sp);J=1-3/(4*(n0+n1)-9)
    return float(J*d)

def numeric_tma_columns(df,exclude=()):
    ex=set(exclude)|{"n_frames","n_events","projection_depth_mean","projection_error_ms_mean","projection_error_ms_max"}
    return [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().all() and np.nanstd(df[c])>1e-10]

def group_contrast(df,group_col,g0,g1,features):
    rows=[]
    for c in features:
        a=df.loc[df[group_col]==g0,c].dropna().values;b=df.loc[df[group_col]==g1,c].dropna().values
        if len(a)<3 or len(b)<3:continue
        tt=stats.ttest_ind(a,b,equal_var=False)
        rows.append({"contrast":f"{g1}-vs-{g0}","feature":c,"n0":len(a),"n1":len(b),
                     f"{g0}_mean":float(np.mean(a)),f"{g1}_mean":float(np.mean(b)),
                     "diff":float(np.mean(b)-np.mean(a)),"welch_p":float(tt.pvalue),"hedges_g":hedges_g(a,b)})
    out=pd.DataFrame(rows)
    if len(out): out["q_BH"]=bh_adjust(out.welch_p.values);out=out.sort_values("welch_p")
    return out

def paired_contrast(df,subject_col,condition_col,c0,c1,features):
    A=df[df[condition_col]==c0].set_index(subject_col);B=df[df[condition_col]==c1].set_index(subject_col)
    subs=sorted(set(A.index)&set(B.index));rows=[]
    for c in features:
        x=A.loc[subs,c].to_numpy(float);y=B.loc[subs,c].to_numpy(float);d=y-x
        if len(d)<3:continue
        p=float(stats.ttest_rel(y,x).pvalue) if not np.allclose(d,0) else 1.0
        dz=float(np.mean(d)/np.std(d,ddof=1)) if np.std(d,ddof=1)>0 else np.nan
        rows.append({"contrast":f"{c1}-minus-{c0}","feature":c,"n_pairs":len(d),
                     f"{c0}_mean":float(np.mean(x)),f"{c1}_mean":float(np.mean(y)),
                     "mean_difference":float(np.mean(d)),"paired_p":p,"dz":dz})
    out=pd.DataFrame(rows)
    if len(out): out["q_BH"]=bh_adjust(out.paired_p.values);out=out.sort_values("paired_p")
    return out
