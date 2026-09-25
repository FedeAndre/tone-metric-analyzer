#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

BASE=Path("research/tma_gait_ndd_exact_results")
df=pd.read_csv(BASE/"subject_tma_features.csv")
conv=pd.read_csv(BASE/"conventional_features.csv")
OUT=BASE/"covariate_adjusted.json"

def ols_pd(feature):
    z=df[df.group.isin(["pd","control"])][["group","age","gait_speed",feature]].dropna().copy()
    z["pd"]=(z.group=="pd").astype(float)
    y=z[feature].to_numpy(float)
    # standardize outcome so PD coefficient is standardized SD units.
    y=(y-y.mean())/y.std(ddof=1)
    X=np.column_stack([np.ones(len(z)),z.pd,z.age,z.gait_speed])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    resid=y-X@beta; dof=len(y)-X.shape[1]
    s2=np.sum(resid**2)/dof
    cov=s2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov))
    t=beta[1]/se[1]; p=2*stats.t.sf(abs(t),dof)
    return {"feature":feature,"n":len(z),"pd_beta_standardized":float(beta[1]),"se":float(se[1]),"t":float(t),"p":float(p),
            "covariates":["age","gait_speed"]}

def partial_spearman_pd(feature):
    z=df[df.group=="pd"][["severity","age","gait_speed",feature]].dropna()
    # rank-transform y and feature; residualize both on age+speed ranks.
    ry=stats.rankdata(z.severity); rf=stats.rankdata(z[feature])
    C=np.column_stack([np.ones(len(z)),stats.rankdata(z.age),stats.rankdata(z.gait_speed)])
    ey=ry-C@np.linalg.lstsq(C,ry,rcond=None)[0]
    ef=rf-C@np.linalg.lstsq(C,rf,rcond=None)[0]
    r,p=stats.pearsonr(ey,ef)
    return {"feature":feature,"n":len(z),"partial_spearman_like_r":float(r),"p":float(p),
            "controlled":["age","gait_speed"]}

keys=["mean_D","compound_pivot_fraction","LHS_mean_H","LHS_mean_D","tree_span_time","pivot_depth","RHS_multilevel","RHS_mean_D","RHS_mean_lambda"]
out={"pd_vs_control_adjusted":[ols_pd(x) for x in keys],
     "severity_adjusted":[partial_spearman_pd(x) for x in ["tree_span_time","RHS_multilevel","RHS_mean_D","RHS_mean_lambda","tree_span_events","mean_D","multilevel"]]}
OUT.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
