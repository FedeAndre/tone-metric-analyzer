#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import numpy as np
import pandas as pd
import tma_cross_domain_frozen as t
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile

OUT=Path("research/tma_worked_examples")
OUT.mkdir(parents=True,exist_ok=True)
captured={}
orig_mk=t.mkrow

def capture_mk(domain,entity,view,frames,meta=None):
    key=(domain,str(entity),view)
    if domain not in captured and view=="A":
        captured[domain]={
            "entity":str(entity),
            "view":view,
            "frames":[[float(x) for x in np.asarray(fr,float)] for fr in frames],
            "meta":meta or {}
        }
    return orig_mk(domain,entity,view,frames,meta)

t.mkrow=capture_mk

# Capture representative real samples.
for name,fn in [
    ("gait",t.gait_rows),
    ("ultrasound",t.oasbud_rows),
    ("fmri",t.hcp_rows),
]:
    try:
        fn()
    except Exception as e:
        captured.setdefault(name,{"error":repr(e)})

# Add exact deterministic counter example.
captured["binary_counter"]={
    "entity":"0","view":"A",
    "frames":[[0.125,0.25,0.5] for _ in range(8)],
    "meta":{"source":"deterministic nested divider"}
}

def event_detail(frames):
    qframes=[t.quantize_phases(x) for x in frames]
    rows=[]
    for fi,phs in enumerate(qframes):
        for q in phs:
            rows.append((Fraction(fi)+q,fi,q))
    rows.sort(key=lambda z:z[0])
    unique=sorted({x[0] for x in rows})
    actual=set(unique); levels={x:set() for x in unique}
    max_pos=t.F+1
    top_end=t._next_seq_anchor(2,max_pos)
    point_layers={}; interval_levels={}
    t.recursive_integer_structure(1,top_end,2,1,point_layers,interval_levels)
    point_layers={p:ls for p,ls in point_layers.items() if p<=max_pos}
    interval_levels={p:l for p,l in interval_levels.items() if p<max_pos}
    for pos,ls in point_layers.items():
        tt=Fraction(pos-1)
        if tt in actual: levels[tt].update(int(x) for x in ls)
    by_interval=defaultdict(list)
    for tt in unique:
        k=int(math.floor(float(tt)))
        if 0<=k<t.F: by_interval[k].append(tt)
    def refine(a,b,ev,parent,guard=0):
        interior=[tt for tt in ev if a<tt<b]
        if not interior:return
        m=(a+b)/2; lvl=int(parent)+1
        for tt in (a,m,b):
            if tt in actual: levels[tt].add(lvl)
        left=[tt for tt in ev if a<=tt<=m]
        right=[tt for tt in ev if m<=tt<=b]
        if any(a<tt<m for tt in left): refine(a,m,left,lvl,guard+1)
        if any(m<tt<b for tt in right): refine(m,b,right,lvl,guard+1)
    for i in range(t.F):
        a,b=Fraction(i),Fraction(i+1); ev=by_interval.get(i,[])
        if any(a<tt<b for tt in ev):
            refine(a,b,ev,int(interval_levels.get(i+1,1)))
    wave=[]; detail=[]
    for idx,(onset,fi,q) in enumerate(rows):
        ks=sorted(int(x) for x in levels[onset] if int(x)>0)
        wave.append({
            "segment_index":0,"measure_index":fi,"measure_number":str(fi+1),
            "onset_quarter":str(onset),"offset_in_measure_quarter":str(q),
            "levels":ks,"height":max(ks),"density":len(ks),
            "lowest_level":min(ks),"attack":True,"parenthetical":False,
        })
        detail.append({
            "event_index":idx+1,
            "frame":fi+1,
            "raw_phase":float(q),
            "quantized_phase":str(q),
            "absolute_position":str(onset),
            "levels":ks,
            "H":max(ks),
            "D":len(ks),
            "lambda":min(ks)
        })
    piv=build_pivot_profile(wave)
    tree=build_tree_profile(wave)
    return detail,piv,tree,t.analyze_frames(frames)

out={}
for domain,item in captured.items():
    if "frames" not in item:
        out[domain]=item; continue
    det,piv,tree,feat=event_detail(item["frames"])
    out[domain]={**item,"events":det,"pivots":piv,"tree":tree,"features":feat}

(OUT/"worked_examples.json").write_text(json.dumps(out,indent=2,default=str),encoding="utf-8")

# Flat event CSV for publication tables.
rows=[]
for dom,d in out.items():
    for e in d.get("events",[]):
        rows.append({"domain":dom,"entity":d.get("entity"),**e})
pd.DataFrame(rows).to_csv(OUT/"event_level_worked_examples.csv",index=False)

print(json.dumps({k:{"entity":v.get("entity"),"frames":v.get("frames",[None])[:2],"features":v.get("features"),"first_events":v.get("events",[])[:6]} for k,v in out.items()},indent=2,default=str))
