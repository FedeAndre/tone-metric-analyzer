#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd

from tma_gait_gauge_validation import (
    RECORDS, CONTROL_IDS, PD_IDS, OUT,
    download_record, extract_cycles, full_population, group_table,
    regression_targets, split_half
)

def main():
    t0=time.time()
    subjects=[]
    qc=[]
    for i,(record,group) in enumerate(RECORDS,1):
        print(f"[core download] {i:02d}/{len(RECORDS)} {record}", flush=True)
        arr=download_record(record)
        cycles,fs,integrity=extract_cycles(arr,record)
        subjects.append({"record":record,"group":group,"cycles":cycles})
        qc.append({"record":record,"group":group,"n_rows":len(arr),"fs":fs,
                   "n_cycles":len(cycles),"mean_stride_s":sum(c["duration_s"] for c in cycles)/len(cycles),
                   **integrity})
    pd.DataFrame(qc).to_csv(OUT/"core_raw_qc.csv",index=False)
    full,_=full_population(subjects)
    group=group_table(full)
    regression=regression_targets(full)
    a,b,rel,cols,ident,perm=split_half(subjects)
    summary={
        "n_subjects":len(subjects),
        "n_controls":len(CONTROL_IDS),
        "n_pd":len(PD_IDS),
        "id_feature_columns":cols,
        "identification":ident,
        "permutation_top1":perm,
        "runtime_s":time.time()-t0,
    }
    (OUT/"core_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("# CORE RESULT", flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print("# REGRESSION TARGETS",flush=True)
    print(regression.to_string(index=False),flush=True)
    print("# CORE RELIABILITY",flush=True)
    wanted={"mean_D","mean_lambda","tree_span_events","pivot_rate","RHS_mean_D","RHS_mean_lambda","RHS_multilevel"}
    print(rel[rel.feature.isin(wanted)].to_string(index=False),flush=True)
    print("# TOP GROUP DIFFERENCES",flush=True)
    print(group.head(15).to_string(index=False),flush=True)

if __name__=="__main__":
    main()
