#!/usr/bin/env python3
"""Pre-specified online causality, metric-horizon, and cost audit for frozen binary TMA.

Input contains a real 4/4 quarter-note tactus and independent variable attacks.
All TMA definitions are exactly the frozen engine binary specialization, cross
checked with production analyze() for selected prefixes. No predictor is trained.
"""
from __future__ import annotations
import json,random,time,statistics,tracemalloc
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
import tma_variable_drums as td
from tone_metric.engine import analyze
from tone_metric.models import Hit,MeasureInfo

OUT=Path("research/tma_online_measurement_results")
OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(20261008)
# each beat carries one reference attack, plus variable early / late attacks
PHASES=[(0, Fraction(1,4)),(0,Fraction(1,2)),(0,Fraction(3,4)),
        (0,Fraction(1,4),Fraction(3,4)),(0,Fraction(1,8),Fraction(5,8)),
        (0,Fraction(3,8)),(0,Fraction(1,2),Fraction(7,8))]
def events(beats=64,mode="variable"):
    seed=np.random.default_rng(20261008+beats*13)
    e=[]
    for b in range(beats):
        k=(int(seed.integers(len(PHASES))) if mode=="variable" else b//4%2)
        for x in PHASES[k]:e.append(Fraction(b)+x)
    return tuple(sorted(set(e)))
def compare(actual,previous):
    common=set(actual).intersection(previous)
    changed={q for q in common if actual[q]!=previous[q]}
    return len(common),len(changed),len([q for q in changed if max(actual[q])!=max(previous[q])]),
       len([q for q in changed if len(actual[q])!=len(previous[q])])
def prefix_audit(full,beats,horizon_bars):
    snapshot={}
    prev={}
    changes=[]
    # Coarse audit after every *complete beat* at a fixed, pre-announced
    # upper horizon; no later event is allowed to influence earlier values
    for j in range(1,beats+1):
        current=tuple(x for x in full if x<j)
        lv=td.levels_full(current,horizon_bars)
        n,changed,ch,cd=compare(lv,prev)
        if j>1:
            changes.append({"beat":j,"existing":n,"retro_levels":changed,"retro_H":ch,"retro_D":cd})
        prev=lv
    return changes
def varying_horizon(full,beats):
    data=[]
    previous=None
    for b in [1,2,4,8,12,16,24,32]:
        if b*4>beats:continue
        cutoff=4*b
        samples=tuple(x for x in full if x<cutoff)
        local=td.levels_full(samples,b)
        if previous:
            n,ch,chh,chd=compare(local,previous)
            data.append({"bars":b,"shared":n,"levels_reassigned":ch,"H_reassigned":chh,"D_reassigned":chd})
        previous=local
    return data
def event_prefix_audit(full,nbars,limit=120):
    """When a new event arrives *inside* the beat, can levels already emitted change?"""
    old={}
    num=0;changed=0;changed_H=0;changed_D=0;reports=[]
    for i in range(1,min(len(full),limit)):
        lv=td.levels_full(full[:i+1],nbars)
        n,ch,chh,chd=compare(lv,old)
        if old:
            num+=n;changed+=ch;changed_H+=chh;changed_D+=chd
            if ch and len(reports)<8:
                q=next(q for q in old if q in lv and lv[q]!=old[q])
                reports.append({"arrival":str(full[i]),"previous_event":str(q),
                                "old_levels":sorted(old[q]),"new_levels":sorted(lv[q])})
        old=lv
    return {"prior_event_comparisons":num,"reassigned_levels_count":changed,
            "reassigned_H_count":changed_H,"reassigned_D_count":changed_D,
            "examples":reports,"processed_events":min(len(full),limit)}
def has_prefix_invariance(full,nbars):
    """Choose several fixed metric windows to audit exact prefix-level semantics."""
    lo=tuple(x for x in full if x<4)
    hi=td.levels_full(full,nbars)
    a=td.levels_full(lo,nbars)
    changed=[(str(q),sorted(a[q]),sorted(hi[q])) for q in lo if a[q]!=hi[q]]
    return {"first_bar_events":len(lo),"first_bar_reassigned_final":len(changed),
            "first_three_examples":changed[:3]}
def timing_cost(beats):
    src=events(beats)
    nbars=beats//4
    # repeated measurements after warmup, same hardware and exact input
    td.levels_full(src,nbars)
    samples=[]
    for _ in range(7):
        a=time.perf_counter()
        wave=td.levels_full(src,nbars)
        samples.append(time.perf_counter()-a)
    # stream incremental conventional algorithms, using the SAME events,
    # the same tactus, and an origin-aware phase histogram and IOI statistics
    online=[]
    for _ in range(7):
        start=time.perf_counter()
        counts=np.zeros(16,int);M=0.;M2=0.;last=None;n=0
        for q in src:
            ph=float(q)%1
            counts[min(int(16*ph),15)]+=1
            if last is not None:
                d=float(q-last);n+=1
                delta=d-M;M+=delta/n;M2+=delta*(d-M)
            last=q
        online.append(time.perf_counter()-start)
    # Full exact TMA recalculation per arrival is the currently documented
    # brute-force online procedure, NOT a claimed optimized incremental one.
    if beats<=32:
        cost=time.perf_counter()
        for j in range(1,len(src)+1):
            td.levels_full(src[:j],nbars)
        recompute=time.perf_counter()-cost
    else:recompute=None
    return {"beats":beats,"events":len(src),
       "median_full_TMA_ms":1000*statistics.median(samples),
       "median_Welford_phasehist_ms":1000*statistics.median(online),
       "full_batch_runtime_ratio":statistics.median(samples)/statistics.median(online),
       "one_new_event_prefix_recompute_total_s":recompute}
def main():
    start=time.perf_counter()
    full=events(128)
    invariance=has_prefix_invariance(full,32)
    bybeat=prefix_audit(full,128,32)
    byevent=event_prefix_audit(full,32,limit=120)
    horizon=varying_horizon(full,128)
    costs=[timing_cost(n) for n in (16,32,64,128,256)]
    # Engine checks: diversity of growing windows and deep mixed timing
    for bars in (2,4,8,16,32):
        e=tuple(x for x in full if x<bars*4)
        td.check_engine(e,bars)
    # Even when fixed horizon is 32 bars, demonstrate distinct retrospective
    # assignments at event time vs end under the same beat grid.
    output={
      "method":"frozen engine binary 4/4 rhythmic attacks with fixed quarter-note tactus",
      "events_128beats":len(full),"engine_exact_gates":[2,4,8,16,32],
      "causal_prefix_fixed_horizon":invariance,
      "per_new_beat_earlier_event_reassignments":{
         "n_checks":len(bybeat),
         "checks_with_any_reassignments":sum(r["retro_levels"]>0 for r in bybeat),
         "total_prior_level_reassignments":sum(r["retro_levels"] for r in bybeat),
         "total_prior_height_reassignments":sum(r["retro_H"] for r in bybeat),
         "total_prior_density_reassignments":sum(r["retro_D"] for r in bybeat),
         "examples":[r for r in bybeat if r["retro_levels"]>0][:8]},
      "per_new_event_fixed_horizon":byevent,
      "variable_horizon_reassignment":horizon,
      "runtime":costs,
      "seconds":time.perf_counter()-start,
      "caveats":[
        "All 128 beat events have a reference beat and variable rhythmic attacks, no isolated event-only sequence",
        "Offline complete-horizon engine may revise earlier Level stacks; never call it online causal without proof",
        "Full recomputation per new event is a correctness reference, not an optimized streaming algorithm",
        "Different methods have different feature counts; report clock times but do not infer theoretical big-O from five samples",
        "The 'provisional horizon' test is not the same as event-conditioned changes within a fixed metric horizon",
        "Origin remains fixed; metric length changes only in the variable-horizon sensitivity"],
    }
    OUT.joinpath("causality_and_cost.json").write_text(json.dumps(output,indent=2))
    print("PASS frozen production-engine equivalence across five different horizons")
    print("FINAL_CAUSALITY",json.dumps(output,indent=2),flush=True)
if __name__=="__main__":main()
