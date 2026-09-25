#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from fractions import Fraction
import pandas as pd
import numpy as np

import tma_gait_gauge_validation as g
import tma_cross_domain_frozen as c

OUT=Path("research/tma_clinical_worked_examples")
OUT.mkdir(parents=True, exist_ok=True)

def native_example(record, group):
    arr=g.download_record(record)
    cycles, fs, integrity=g.extract_cycles(arr, record)
    sample=cycles[:8]
    feat, event_df, pivots, tree=g.analyze_cycles(sample)
    # raw/projection table for the exact same 8 cycles
    raw=[]
    for ci, cyc in enumerate(sample,1):
        for e in cyc["events"]:
            raw.append({
                "record":record,"group":group,"cycle":ci,
                "cycle_start_sample":cyc["start_sample"],
                "cycle_end_sample":cyc["end_sample"],
                "cycle_duration_s":cyc["duration_s"],
                "event":e["label"],"sample":e["sample"],
                "raw_phase":e["raw_phase"],
                "projected_phase":f'{e["phase_num"]}/{e["phase_den"]}',
                "projection_depth":e["projection_depth"],
                "projection_error_ms":e["projection_error_ms"],
            })
    rawdf=pd.DataFrame(raw)
    # merge event-level TMA output in event order
    native=rawdf.copy()
    native["TMA_onset"]=event_df["onset"].to_numpy()
    native["Levels"]=event_df["levels"].to_numpy()
    native["H"]=event_df["H"].to_numpy()
    native["D"]=event_df["D"].to_numpy()
    native["lambda"]=event_df["lambda"].to_numpy()
    native.to_csv(OUT/f"{record}_native_event_trace.csv",index=False)

    pd.DataFrame([{"record":record,"group":group,**feat}]).to_csv(
        OUT/f"{record}_native_features.csv",index=False)

    # Frozen cross-domain core: exclude LHS and divide phase on 2-cycle scale to [0,1)
    frames=[]
    for cyc in sample:
        ph=[float(Fraction(e["phase_num"],e["phase_den"]))/2.0 for e in cyc["events"] if e["label"]!="LHS"]
        frames.append(ph)
    qframes=[c.quantize_phases(x) for x in frames]
    corefeat=c.analyze_frames(frames)
    qrows=[]
    for fi,(orig,qf) in enumerate(zip(frames,qframes),1):
        for j,(o,q) in enumerate(zip(orig,qf),1):
            qrows.append({
                "record":record,"group":group,"frame":fi,"event_index":j,
                "normalized_phase_prequant":o,
                "quantized_bin":int(q* c.GRID),
                "quantized_phase":f"{q.numerator}/{q.denominator}",
            })
    pd.DataFrame(qrows).to_csv(OUT/f"{record}_core_input.csv",index=False)
    pd.DataFrame([{"record":record,"group":group,**corefeat}]).to_csv(
        OUT/f"{record}_core_features.csv",index=False)

    # Serialize pivots/tree compactly
    (OUT/f"{record}_pivots.json").write_text(json.dumps(pivots,indent=2,default=str))
    (OUT/f"{record}_tree.json").write_text(json.dumps(tree,indent=2,default=str))
    return {
        "record":record,"group":group,"fs":fs,"integrity":integrity,
        "native_features":feat,"core_features":corefeat,
        "first_cycle":raw[:4],
        "n_pivots":len(pivots),
        "n_tree_nodes":len(tree.get("nodes",[])),
        "n_tree_branches":len(tree.get("branches",[])),
    }

summ=[]
for rec,grp in [("GaCo01_01","control"),("GaPt03_01","parkinson")]:
    summ.append(native_example(rec,grp))
(OUT/"summary.json").write_text(json.dumps(summ,indent=2,default=str))
print(json.dumps(summ,indent=2,default=str))
