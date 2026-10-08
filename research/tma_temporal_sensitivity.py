#!/usr/bin/env python3
"""Sensitivity audit: subject-held-out TMA temporal evolution vs conventional
gait and respiratory baselines at 3 regularization strengths and compact TMA.
Uses completed physiology derived features only: no new raw download.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUTPUT=Path('research/tma_temporal_evolution_results/sensitivity.json')
ALPHAS=[3,30,300]
TARGETS={
 'gait':['force_step','force_drift','force_texture'],
 'cardioresp':['wave_step','wave_drift','wave_texture'],
}
CARDIO_URL=('https://raw.githubusercontent.com/FedeAndre/tone-metric-analyzer/'
 'research-tma-cardioresp-evolution-20261008/research/tma_cardioresp_evolution_results/'
 'respiration_evolution_windows.csv')
def load():
    gait=pd.read_csv('research/tma_temporal_evolution_results/evolution_windows.csv')
    r=requests.get(CARDIO_URL,timeout=90);r.raise_for_status()
    from io import StringIO
    cardio=pd.read_csv(StringIO(r.text))
    return gait,cardio
def cols(df,prefix):
    return [c for c in df if c.startswith(prefix+'_')]
def select(df,prefix,base):
    return [f'{prefix}_{s}{z}' for s in base for z in ('','_delta_prev','_delta_origin')]
def choose(df,domain):
    if domain=='gait':
        conventional=cols(df,'timing')
        origin=cols(df,'tma_origin')
        reset=cols(df,'tma_reset')
        core=['Dmean','Hchange','level_turnover','pivot_rate','tree_span']
        os=select(df,'tma_origin',core)
        rs=select(df,'tma_reset',core)
    else:
        conventional=cols(df,'timing')
        origin=cols(df,'origin')
        reset=cols(df,'reset')
        core=['Dmean','Hchange','turnover','pivot_rate','tree_span']
        os=[f'origin_{s}{z}' for s in core for z in ('','_dprev','_dstart')]
        rs=[f'reset_{s}{z}' for s in core for z in ('','_dprev','_dstart')]
    assert all(x in df for x in conventional+origin+reset+os+rs)
    return {'conventional':conventional,
        'origin_TMA':origin,'reset_TMA':reset,
        'conventional_plus_full_origin':conventional+origin,
        'conventional_plus_full_reset':conventional+reset,
        'conventional_plus_core_origin':conventional+os,
        'conventional_plus_core_reset':conventional+rs}
def predict(df,features,target,alpha):
    y=df[target].to_numpy(float)
    output=np.full(y.shape,np.nan)
    for rec in df.record.unique():
        ix=(df.record==rec).to_numpy()
        train=~ix
        x_train=df.loc[train,features].to_numpy(float)
        x_test=df.loc[ix,features].to_numpy(float)
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(x_train,y[train])
        output[ix]=model.predict(x_test)
    assert np.isfinite(output).all()
    return output
def mean_predict(df,target):
    y=df[target].to_numpy(float)
    ret=np.empty_like(y)
    for rec in df.record.unique():
        ix=(df.record==rec).to_numpy()
        ret[ix]=y[~ix].mean()
    return ret
def go():
    out={}
    t=time.monotonic()
    for domain,df in zip(('gait','cardioresp'),load()):
        models=choose(df,domain)
        out[domain]={'subjects':df.record.nunique(),'windows':len(df),'results':{}}
        for target in TARGETS[domain]:
            y=df[target].to_numpy(float)
            denom=np.sum((y-y.mean())**2)
            base=mean_predict(df,target)
            z={'subject_out_mean':{'MAE':float(np.mean(np.abs(y-base))),
                                  'R2':float(1-np.sum((y-base)**2)/denom)}}
            for alpha in ALPHAS:
                z[str(alpha)]={}
                for name,features in models.items():
                    pred=predict(df,features,target,alpha)
                    z[str(alpha)][name]={'MAE':float(np.mean(np.abs(y-pred))),
                                           'R2':float(1-np.sum((y-pred)**2)/denom)}
            out[domain]['results'][target]=z
        print(domain+' complete',flush=True)
    out['seconds']=time.monotonic()-t
    OUTPUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({d:{tar:{k:val for k,val in r.items() if k in ['subject_out_mean','30']}
                   for tar,r in out[d]['results'].items()} for d in TARGETS},indent=2),flush=True)
if __name__=='__main__':go()
