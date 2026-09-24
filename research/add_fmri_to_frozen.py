#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import tma_cross_domain_frozen as t

base=Path("research/tma_cross_domain_frozen_results/sample_features_raw.csv")
if not base.exists():
    raise RuntimeError("Existing six-domain frozen raw features are missing")
old=pd.read_csv(base)
old=old[old.domain!="fmri"].copy()
rows=t.fmri_rows()
if len(rows)<10:
    raise RuntimeError(f"fMRI produced only {len(rows)//2} paired entities")
fmri=pd.DataFrame(rows)
df=pd.concat([old,fmri],ignore_index=True,sort=False)
print("domains",sorted(df.domain.unique()),flush=True)
print("fMRI pairs",len(fmri)//2,flush=True)
null_df,mu,sd,keep=t.null_reference()
summary=t.analyze_comparator(df,null_df,mu,sd,keep)
summary["execution"]="existing six-domain frozen raw features + newly recovered raw HCP fMRI, identical frozen comparator"
summary["fmri_added_pairs"]=len(fmri)//2
(t.OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2),flush=True)
