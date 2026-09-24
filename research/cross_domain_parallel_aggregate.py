#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import tma_cross_domain_frozen as t

PART=Path("research/tma_cross_domain_parts")
files=sorted(PART.glob("*.csv"))
if len(files)<4:
    raise RuntimeError(f"Need >=4 domain parts; found {len(files)}: {[p.name for p in files]}")
dfs=[]
for p in files:
    d=pd.read_csv(p)
    dfs.append(d)
    print(f"loaded {p.name}: {len(d)} rows",flush=True)
df=pd.concat(dfs,ignore_index=True)
domains=sorted(df.domain.unique())
print("domains",domains,flush=True)
null_df,mu,sd,keep=t.null_reference()
summary=t.analyze_comparator(df,null_df,mu,sd,keep)
summary["execution"]="parallel raw-domain workers; identical frozen comparator"
summary["part_files"]=[p.name for p in files]
(t.OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2),flush=True)
