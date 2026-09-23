#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import tma_gait_gauge_validation as g
from tone_metric.engine import recursive_integer_structure, _next_seq_anchor
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile

SEED = 20260923
SCRAMBLE_B = 200
OUT = g.OUT


def event_tuples(cycles):
    rows=[]
    for ci,c in enumerate(cycles):
        base=Fraction(2*ci)
        for e in c["events"]:
            q=Fraction(int(e["phase_num"]),int(e["phase_den"]))
            rows.append((base+q,ci,q,e["label"]))
    rows.sort(key=lambda x:(x[0],g.EVENT_TYPES.index(x[3])))
    return rows


def fast_level_sets(cycles):
    rows=event_tuples(cycles)
    n_intervals=2*len(cycles)
    max_pos=n_intervals+1
    top_end=_next_seq_anchor(2,max_pos)
    int_layers={}
    interval_levels={}
    recursive_integer_structure(1,top_end,2,1,int_layers,interval_levels)
    int_layers={p:ls for p,ls in int_layers.items() if p<=max_pos}
    interval_levels={p:l for p,l in interval_levels.items() if p<max_pos}

    unique_times=sorted(set(t for t,_,_,_ in rows))
    levels={t:set() for t in unique_times}
    actual=set(unique_times)

    for pos,ls in int_layers.items():
        t=Fraction(pos-1)
        if t in actual:
            levels[t].update(int(x) for x in ls)

    # Exact specialization of engine._refine_span for binary, non-tuplet gait.
    by_interval={}
    for t in unique_times:
        # An integer boundary can be affected by refinement on both sides; it is
        # included through endpoint updates below, not as an interior event.
        k=int(math.floor(float(t)))
        for i in (k-1,k):
            if 0<=i<n_intervals and Fraction(i)<=t<=Fraction(i+1):
                by_interval.setdefault(i,[]).append(t)

    def refine(a,b,ev,parent):
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
            refine(a,m,left,lvl)
        if any(m<t<b for t in right):
            refine(m,b,right,lvl)

    for i in range(n_intervals):
        a,b=Fraction(i),Fraction(i+1)
        ev=by_interval.get(i,[])
        if any(a<t<b for t in ev):
            refine(a,b,ev,int(interval_levels.get(i+1,1)))

    return rows, levels


def fast_analyze_cycles(cycles):
    rows,levelmap=fast_level_sets(cycles)
    evrows=[]
    for onset,ci,q,label in rows:
        K=sorted(int(x) for x in levelmap.get(onset,set()) if int(x)>0)
        if not K:
            raise RuntimeError(f"unresolved event at {onset}")
        evrows.append({
            "label":label,"cycle_index":ci,"onset_frac":onset,"offset_frac":q,
            "H":max(K),"D":len(K),"lambda":min(K),"levels":K,
        })
    event_df=pd.DataFrame([{
        "label":r["label"],"cycle_index":r["cycle_index"],"H":r["H"],"D":r["D"],
        "lambda":r["lambda"],"levels":";".join(str(x) for x in r["levels"]),
        "onset":str(r["onset_frac"])
    } for r in evrows])

    wave=[]
    seen=set()
    for r in evrows:
        key=(r["cycle_index"],r["offset_frac"])
        if key in seen:
            continue
        seen.add(key)
        wave.append({
            "segment_index":0,
            "measure_index":r["cycle_index"],
            "measure_number":str(r["cycle_index"]+1),
            "onset_quarter":str(r["onset_frac"]),
            "offset_in_measure_quarter":str(r["offset_frac"]),
            "levels":r["levels"],
            "height":r["H"],
            "density":r["D"],
            "lowest_level":r["lambda"],
            "attack":True,
            "parenthetical":False,
        })
    pivots=build_pivot_profile(wave)
    tree=build_tree_profile(wave)
    branches=tree.get("branches",[])

    feat={
        "n_cycles":len(cycles),"n_events":len(event_df),
        "mean_H":float(event_df.H.mean()),
        "mean_D":float(event_df.D.mean()),
        "mean_lambda":float(event_df["lambda"].mean()),
        "multilevel":float((event_df.D>1).mean()),
        "pivot_rate":float(len(pivots)/len(event_df)),
        "compound_pivot_fraction":float(np.mean([p["compound"] for p in pivots])) if pivots else np.nan,
        "pivot_depth":float(np.mean([p["drop_depth"] for p in pivots])) if pivots else np.nan,
        "tree_span_events":float(np.mean([int(b["target_node_index"])-int(b["source_node_index"]) for b in branches])) if branches else np.nan,
        "tree_span_time":float(np.mean([float(Fraction(str(b["distance_quarter"]))) for b in branches])) if branches else np.nan,
        "tree_root_fraction":float(np.mean([bool(n["root"]) for n in tree.get("nodes",[])])) if tree.get("nodes") else np.nan,
    }
    for label in g.EVENT_TYPES:
        z=event_df[event_df.label==label]
        feat[f"{label}_mean_H"]=float(z.H.mean())
        feat[f"{label}_mean_D"]=float(z.D.mean())
        feat[f"{label}_mean_lambda"]=float(z["lambda"].mean())
        feat[f"{label}_multilevel"]=float((z.D>1).mean())
    depths=[e["projection_depth"] for c in cycles for e in c["events"] if e["label"]!="LHS"]
    errors=[e["projection_error_ms"] for c in cycles for e in c["events"] if e["label"]!="LHS"]
    feat["projection_depth_mean"]=float(np.mean(depths))
    feat["projection_error_ms_mean"]=float(np.mean(errors))
    feat["projection_error_ms_max"]=float(np.max(errors))
    return feat,event_df,pivots,tree


def assert_equivalent(cycles,name):
    f0,e0,p0,t0=g.analyze_cycles(cycles)
    f1,e1,p1,t1=fast_analyze_cycles(cycles)
    if e0["levels"].tolist()!=e1["levels"].tolist():
        bad=[i for i,(a,b) in enumerate(zip(e0["levels"],e1["levels"])) if a!=b]
        raise AssertionError(f"{name}: Level mismatch at {bad[:8]}")
    for k in ("mean_H","mean_D","mean_lambda","multilevel","pivot_rate",
              "compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"):
        a,b=f0[k],f1[k]
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a,b,rtol=0,atol=1e-12):
            raise AssertionError(f"{name}: feature {k} mismatch exact={a} fast={b}")
    sig0=[(p["kind"],p["drop_depth"],p["pivot_onset_quarter"]) for p in p0]
    sig1=[(p["kind"],p["drop_depth"],p["pivot_onset_quarter"]) for p in p1]
    if sig0!=sig1:
        raise AssertionError(f"{name}: pivot mismatch")
    b0=[(b["source_node_index"],b["target_node_index"],b["source_level"],b["target_level"]) for b in t0["branches"]]
    b1=[(b["source_node_index"],b["target_node_index"],b["source_level"],b["target_level"]) for b in t1["branches"]]
    if b0!=b1:
        raise AssertionError(f"{name}: tree mismatch")
    return True


def load_one(item):
    record,group=item
    arr=g.download_record(record)
    cycles,fs,integrity=g.extract_cycles(arr,record)
    return {"record":record,"group":group,"cycles":cycles,"fs":fs,"n_rows":len(arr),"integrity":integrity}


def load_subjects():
    subs=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(load_one,item):item for item in g.RECORDS}
        done=0
        for fut in as_completed(futs):
            sub=fut.result()
            subs.append(sub); done+=1
            print(f"[download] {done:02d}/{len(g.RECORDS)} {sub['record']}",flush=True)
    order={r:i for i,(r,_) in enumerate(g.RECORDS)}
    subs.sort(key=lambda s:order[s["record"]])
    return subs


def full_population(subs):
    rows=[]
    for i,s in enumerate(subs,1):
        f,*_=fast_analyze_cycles(s["cycles"])
        rows.append({"record":s["record"],"group":s["group"],**f})
    df=pd.DataFrame(rows).sort_values("record").reset_index(drop=True)
    df.to_csv(OUT/"fast_exact_full_subject_features.csv",index=False)
    return df


def split_half(subs):
    A=[];B=[]
    for s in subs:
        fa,*_=fast_analyze_cycles(s["cycles"][:g.HALF_CYCLES])
        fb,*_=fast_analyze_cycles(s["cycles"][g.HALF_CYCLES:g.N_CYCLES])
        A.append({"record":s["record"],"group":s["group"],**fa})
        B.append({"record":s["record"],"group":s["group"],**fb})
    a=pd.DataFrame(A).sort_values("record").reset_index(drop=True)
    b=pd.DataFrame(B).sort_values("record").reset_index(drop=True)
    a.to_csv(OUT/"fast_exact_split_half_A.csv",index=False)
    b.to_csv(OUT/"fast_exact_split_half_B.csv",index=False)
    reli=[]
    numeric=[c for c in a.columns if c not in {"record","group"} and c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    for c in numeric:
        mask=np.isfinite(a[c].to_numpy(float)) & np.isfinite(b[c].to_numpy(float))
        if mask.sum()<10: continue
        x=a.loc[mask,c].to_numpy(float); y=b.loc[mask,c].to_numpy(float)
        pr,pp=g.corr_safe(x,y,"pearson"); sr,sp=g.corr_safe(x,y,"spearman")
        reli.append({"feature":c,"n":int(mask.sum()),"half_A_mean":float(x.mean()),"half_B_mean":float(y.mean()),
                     "pearson_r":pr,"pearson_p":pp,"spearman_rho":sr,"spearman_p":sp,"icc3_1":g.icc3_1(x,y)})
    rel=pd.DataFrame(reli).sort_values("icc3_1",ascending=False)
    rel.to_csv(OUT/"fast_exact_split_half_reliability.csv",index=False)
    cols=g.select_id_columns(a,b)
    ident,dist=g.identification_metrics(a,b,cols)
    perm=g.permutation_top1_p(dist,np.random.default_rng(SEED+100),B=g.PERM_B)
    return a,b,rel,cols,ident,perm


def scramble(subs,id_cols,real_ident):
    rng=np.random.default_rng(SEED+300)
    out={}
    for mode in ("cycle_order","phase_label"):
        top=[]; within=[]; ratio=[]; ranks=[]
        iccs={k:[] for k in ("mean_D","mean_lambda","tree_span_events","pivot_rate")}
        for rep in range(SCRAMBLE_B):
            A=[];B=[]
            for s in subs:
                ca=s["cycles"][:g.HALF_CYCLES]; cb=s["cycles"][g.HALF_CYCLES:g.N_CYCLES]
                if mode=="cycle_order":
                    ca=g.cycle_order_scramble(ca,rng); cb=g.cycle_order_scramble(cb,rng)
                else:
                    ca=g.phase_label_scramble(ca,rng); cb=g.phase_label_scramble(cb,rng)
                fa,*_=fast_analyze_cycles(ca); fb,*_=fast_analyze_cycles(cb)
                A.append({"record":s["record"],"group":s["group"],**fa})
                B.append({"record":s["record"],"group":s["group"],**fb})
            a=pd.DataFrame(A).sort_values("record").reset_index(drop=True)
            b=pd.DataFrame(B).sort_values("record").reset_index(drop=True)
            cols=[c for c in id_cols if np.isfinite(a[c]).all() and np.isfinite(b[c]).all()
                  and np.nanstd(np.r_[a[c].values,b[c].values])>0]
            im,_=g.identification_metrics(a,b,cols)
            top.append(im["top1_rate"]); within.append(im["within_group_top1_rate"])
            ratio.append(im["distance_ratio_same_over_different"]); ranks.append(im["median_rank"])
            for c in iccs: iccs[c].append(g.icc3_1(a[c],b[c]))
            if (rep+1)%25==0: print(f"[fast scramble:{mode}] {rep+1}/{SCRAMBLE_B}",flush=True)
        arr=np.array(top)
        out[mode]={
            "B":SCRAMBLE_B,
            "top1_mean":float(arr.mean()),"top1_q025":float(np.quantile(arr,.025)),"top1_q975":float(np.quantile(arr,.975)),
            "p_ge_real_top1":float((1+np.sum(arr>=real_ident["top1_rate"]))/(SCRAMBLE_B+1)),
            "within_group_top1_mean":float(np.mean(within)),
            "distance_ratio_mean":float(np.mean(ratio)),
            "median_rank_mean":float(np.mean(ranks)),
            "core_icc_mean":{k:float(np.nanmean(v)) for k,v in iccs.items()},
            "core_icc_q025":{k:float(np.nanquantile(v,.025)) for k,v in iccs.items()},
            "core_icc_q975":{k:float(np.nanquantile(v,.975)) for k,v in iccs.items()},
        }
    return out


def main():
    t=time.time()
    subs=load_subjects()
    # Equivalence gate on real data: full and both halves from three records,
    # spanning control and PD. No statistical result is accepted if this fails.
    tests=[subs[0],subs[len(g.CONTROL_IDS)-1],subs[len(g.CONTROL_IDS)],subs[-1]]
    checks=[]
    for s in tests:
        for tag,cy in (("full",s["cycles"]),("A",s["cycles"][:32]),("B",s["cycles"][32:64])):
            assert_equivalent(cy,f"{s['record']}-{tag}")
            checks.append(f"{s['record']}-{tag}")
            print(f"[equivalence] {checks[-1]} PASS",flush=True)

    full=full_population(subs)
    group=g.group_table(full)
    regression=g.regression_targets(full)
    a,b,rel,cols,ident,perm=split_half(subs)
    scr=scramble(subs,cols,ident)

    summary={
        "seed":SEED,"implementation":"specialized binary evaluator, gated by exact repository-engine equality",
        "equivalence_checks":checks,"equivalence_passed":True,
        "n_subjects":len(subs),"n_controls":len(g.CONTROL_IDS),"n_pd":len(g.PD_IDS),
        "split_half":{"id_feature_columns":cols,"identification":ident,"permutation_top1":perm},
        "scrambling":scr,"runtime_s":time.time()-t,
    }
    (OUT/"fast_exact_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    lines=["# Fast exact TMA gait gauge validation","",
           f"Repository-engine equivalence checks: {len(checks)} / {len(checks)} passed.",
           f"Subjects: {len(subs)} ({len(g.CONTROL_IDS)} controls, {len(g.PD_IDS)} PD).","",
           "## Frozen-population regression"]
    for _,r in regression.iterrows():
        lines.append(f"- {r.feature}: current C={r.current_control:.4f}, frozen C={r.frozen_control:.4f}; current PD={r.current_pd:.4f}, frozen PD={r.frozen_pd:.4f}.")
    lines += ["","## Split-half",
              f"- Top-1 person identification: {ident['top1_rate']:.3f}; chance={1/len(subs):.3f}; permutation p={perm['p_ge_observed']:.5g}.",
              f"- Within-group top-1: {ident['within_group_top1_rate']:.3f}.",
              f"- Median true-match rank: {ident['median_rank']:.2f}.",
              f"- Same/different distance ratio: {ident['distance_ratio_same_over_different']:.3f}.","",
              "## Scrambling"]
    for mode,z in scr.items():
        lines.append(f"- {mode}: top-1 mean={z['top1_mean']:.3f}, 95%={z['top1_q025']:.3f}–{z['top1_q975']:.3f}, p vs real={z['p_ge_real_top1']:.5g}.")
    lines += ["","## Core ICCs"]
    for c in ("mean_D","mean_lambda","tree_span_events","pivot_rate","RHS_mean_D","RHS_mean_lambda","RHS_multilevel"):
        rr=rel[rel.feature==c]
        if len(rr):
            q=rr.iloc[0]; lines.append(f"- {c}: ICC={q.icc3_1:.3f}; Pearson r={q.pearson_r:.3f}.")
    lines += ["","## Strongest current group differences"]
    for _,r in group.head(10).iterrows():
        lines.append(f"- {r.feature}: C={r.control_mean:.4f}, PD={r.pd_mean:.4f}, g={r.hedges_g:+.3f}, p={r.welch_p:.5g}.")
    (OUT/"FAST_EXACT_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
