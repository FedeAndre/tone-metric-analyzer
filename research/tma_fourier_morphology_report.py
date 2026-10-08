#!/usr/bin/env python3
"""Report amplitude-spectrum contrasts for TMA H/D, plain onset impulse spectrum,
and singleton-event fractal-template spectra. JSD compares only eight spectral
bands; it is not an independent information-theoretic gain.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
X=pd.read_csv("research/tma_fourier_results/window_features.csv")
OUT=Path("research/tma_fourier_results/spectral_morphology.json")
assert len(X)==679
def bands(k):
    v=X[[f"{k}_band{i}" for i in range(8)]].to_numpy(float)
    row=v.sum(axis=1)
    return np.divide(v,row[:,None],out=np.zeros_like(v),where=row[:,None]>1e-10)
def divergence(a,b):
    A=bands(a);B=bands(b)
    d=[]
    for x,y in zip(A,B):
        if x.sum()>1e-10 and y.sum()>1e-10:d.append(float(jensenshannon(x,y,base=2)))
        else:d.append(np.nan)
    return np.array(d)
def summary(a):return {"n_finite":int(np.isfinite(a).sum()),"mean":float(np.nanmean(a)),
                     "median":float(np.nanmedian(a)),"q25":float(np.nanquantile(a,.25)),
                     "q75":float(np.nanquantile(a,.75)),"q95":float(np.nanquantile(a,.95))}
contrasts={}
for a,b in [("H","onset"),("D","onset"),("H","singleH"),("D","singleD"),
            ("contextH","onset"),("contextD","onset"),("singleH","onset"),("singleD","onset")]:
    c=divergence(a,b)
    contrasts[f"{a}_vs_{b}"]={"all":summary(c),
       "regular":summary(c[(X.selection=="regular").to_numpy()]),
       "variable":summary(c[(X.selection=="variable").to_numpy()])}
slopes={}
for k in ("onset","singleH","singleD","H","D","contextH","contextD","H_ordinal","D_ordinal","ioi_ordinal"):
    slopes[k]=summary(X[f"{k}_logslope"].to_numpy(float))
peaks={}
for k in ("onset","singleH","H","D"):
    peaks[k]={str(f):float(X[f"{k}_peak_{str(f).replace('.','p')}"].mean()) for f in (.25,.5,1,2,4,8,16)}
res={"n_windows":len(X),"n_tracks":X.track.nunique(),
     "comparisons":contrasts,"frequency_slopes_not_fractal_dimensions":slopes,
     "mean_normalized_power_at_dyadic_harmonics":peaks,
     "interpretation_warning":"Nonzero JSD demonstrates differing spectra, not independent information beyond the full event sequence"}
OUT.write_text(json.dumps(res,indent=2))
print("FOURIER_MORPHOLOGY",json.dumps(res,indent=2))
