#!/usr/bin/env python3
"""Independently verify three components of TMA Fourier fingerprints against
clinical scores: observed FFT, cycle-order-expected FFT, and order-dependent FFT.
Uses frozen archived clinical metadata and participant features, no new download.
Additional validation: leave-person-out train on walk 01, apply to walk 02.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

OUT=Path('research/tma_individual_fingerprint_results')
df=pd.read_csv(OUT/'subject_spectral_fingerprints.csv')
P=df[(df.group=='pd')&(df.trial==1)].copy()
Q=df[(df.group=='pd')&(df.trial==2)].copy()
bands=['16to32_strides','8to16_strides','4to8_strides','2to4_strides']
keys=[f'{v}_{b}' for v in ('H','D') for b in bands]
trials=sorted(set(P.subject)&set(Q.subject))
assert len(P)==29 and len(Q)==25 and len(trials)==25 and len(df)==86

def col(df,mode):
    if mode=='order_residual':
        return df[[f'tma_{x}' for x in keys]].to_numpy(float)
    if mode=='expected':
        return df[[f'null_{x}' for x in keys]].to_numpy(float)
    if mode=='original':
        return (df[[f'tma_{x}' for x in keys]].to_numpy(float)+
                df[[f'null_{x}' for x in keys]].to_numpy(float))
    if mode=='conventional':
        return df[[f'conv_{v}_{b}' for v in ('stride','RTO','RHS','LTO') for b in bands]].to_numpy(float)
    raise ValueError(mode)

def identity(mode):
    a=P.set_index('subject').loc[trials]
    b=Q.set_index('subject').loc[trials]
    X=col(a,mode);Y=col(b,mode)
    avg=X.mean(axis=0);sd=X.std(axis=0);sd=np.maximum(sd,1e-8)
    X=(X-avg)/sd;Y=(Y-avg)/sd
    dist=np.sqrt(np.mean((Y[:,None,:]-X[None,:,:])**2,axis=2))
    ranks=[1+sum(dist[i,j]<dist[i,i] for j in range(len(trials))) for i in range(len(trials))]
    rng=np.random.default_rng(1042)
    null=np.array([np.mean(dist[np.arange(len(trials)),rng.permutation(len(trials))])
                   for _ in range(9999)])
    return {'n':len(trials),'top1_count':sum(x==1 for x in ranks),
            'top1_rate':float(np.mean(np.array(ranks)==1)),
            'chance':1/len(trials),'mean_within_distance':float(np.mean(np.diag(dist))),
            'mean_other_distance':float(np.mean(dist[~np.eye(len(trials),dtype=bool)])),
            'matching_perm_p':float((1+sum(null<=np.mean(np.diag(dist))))/(1+len(null)))}

def clinical_cols(mode):
    basic=['Age','Gender']
    raw=[f'conv_{x}' for x in ('stride_mean','stride_cv','phase_sd','phase_acf','stance_mean','phase_mean')]
    phase=[f'conv_{x}_{b}' for x in ('stride','RHS') for b in bands]
    full=[f'conv_{x}_{b}' for x in ('stride','RTO','RHS','LTO') for b in bands]
    if mode=='simple':return basic+raw
    if mode=='phase':return basic+raw+phase
    if mode=='full':return basic+raw+full
    if mode.endswith('_expected'):
        return clinical_cols(mode[:-9])+[f'null_{x}' for x in keys]
    if mode=='expected_only':return basic+[f'null_{x}' for x in keys]
    if mode=='order_only':return basic+[f'tma_{x}' for x in keys]
    raise ValueError(mode)

def loo(df,mode,target,alpha,source='first'):
    rows=df[np.isfinite(df[target])]
    cols=clinical_cols(mode)
    pred=[]
    y=[]
    people=[]
    for _,row in rows.iterrows():
        training=P[(P.subject!=row.subject)&np.isfinite(P[target])]
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(training[cols],training[target].to_numpy(float))
        y.append(row[target]);pred.append(float(model.predict(pd.DataFrame([row])[cols])[0]))
        people.append(row.subject)
    y=np.asarray(y,float);pred=np.asarray(pred,float)
    return {'n':len(y),'mae':float(np.mean(abs(y-pred))),
            'r2':float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)),
            'pred':pred,'y':y,'subject':people}

def permutation_rho(x,y,nperm=4999):
    rho=float(stats.spearmanr(x,y).statistic)
    rng=np.random.default_rng(123)
    rankx=stats.rankdata(x);ranky=stats.rankdata(y)
    null=np.array([np.corrcoef(rankx,rng.permutation(ranky))[0,1] for _ in range(nperm)])
    return {'rho':rho,'permutation_p':float((1+sum(abs(null)>=abs(rho)))/(1+nperm))}

def followup_assoc():
    selected=[('HoehnYahr','H_2to4_strides'),('HoehnYahr','H_4to8_strides'),
              ('HoehnYahr','D_4to8_strides'),('UPDRSM','H_2to4_strides'),
              ('UPDRS','H_2to4_strides')]
    a=P.set_index('subject').loc[trials]
    b=Q.set_index('subject').loc[trials]
    result={}
    for target,key in selected:
        aa=a[f'null_{key}'];bb=b[f'null_{key}'];y=a[target]
        result[f'{target}:{key}']={
          'trial1':permutation_rho(aa,y),'trial2':permutation_rho(bb,y),
          'feature_Pearson_repeat_r':float(np.corrcoef(aa,bb)[0,1])}
    return result

def main():
    result={'n_primary_pd':len(P),'n_repeated_pd':len(Q),'n_records':len(df),
            'identification':{m:identity(m) for m in ('order_residual','original','expected','conventional')},
            'crosswalk_correlations':followup_assoc(),
            'models':{},'limitations':['Trial 02 task equivalence is not explicitly certified by dataset metadata',
              'Chronological order-dependent component uses full 64-stride power (phase discarded)',
              'Cycle-order null expected component retains each subject event multiset',
              'First-walk exploratory analyses and multiple alpha settings, no external cohort',
              'Crosswalk validation holds out each patient from training; clinical label is same assessment']}
    modes=['simple','phase','full','simple_expected','phase_expected','full_expected','expected_only','order_only']
    for target in ('UPDRSM','UPDRS','HoehnYahr','TUAG','Speed_01'):
        result['models'][target]={}
        for alpha in (20,80):
            result['models'][target][str(alpha)]={}
            for sample,rows in [('first',P),('second',Q)]:
                result['models'][target][str(alpha)][sample]={
                    m:{k:v for k,v in loo(rows,m,target,alpha,sample).items() if k in ('n','mae','r2')}
                    for m in modes}
    filename=OUT/'ablation_results.json'
    filename.write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    assert result['identification']['order_residual']['top1_count']==1
    assert result['identification']['expected']['top1_count']==6
    assert result['identification']['conventional']['top1_count']==3
    print('PASS identity regression checks and 5 clinical phenotypes')
    print(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)
if __name__=='__main__':main()
