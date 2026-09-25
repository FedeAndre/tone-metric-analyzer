#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
P=Path("research/tma_neuroqwerty_exact_results")
df=pd.read_csv(P/"subject_features.csv")
pdq=df[df.group=="pd"].copy()
features=["pivot_rate","tree_root_fraction","mean_lambda","mean_H","multilevel","mean_D"]
controls=[c for c in ["typing_speed_calc","median_ipi","ipi_cv","hold_mean","hold_sd","pause_fraction"] if c in pdq]
def partial_rank(feature):
    cols=["updrs",feature]+controls
    z=pdq[cols].dropna()
    y=stats.rankdata(z.updrs);x=stats.rankdata(z[feature])
    C=np.column_stack([np.ones(len(z))]+[stats.rankdata(z[c]) for c in controls])
    ey=y-C@np.linalg.lstsq(C,y,rcond=None)[0]
    ex=x-C@np.linalg.lstsq(C,x,rcond=None)[0]
    r,p=stats.pearsonr(ey,ex)
    return {"feature":feature,"n":len(z),"partial_rank_r":float(r),"p":float(p),"controls":controls}
out=[partial_rank(f) for f in features]
(P/"updrs_adjusted.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
