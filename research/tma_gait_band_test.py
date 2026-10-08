#!/usr/bin/env python3
"""Frequency-resolved TMA test on PhysioNet gaitpdb with subject-level nulls.

Frequency units: cycles per stride (64 strides). Main feature is expected-
log-power-corrected height/density spectrum, with 59 whole-cycle permutations
per subject, which preserve all foot-contact phases and their marginal spectrum
but destroy chronological order. A second 4-cycle-block null tests longer order.
Comparison: raw phase timing spectra and conventional gait variables.
"""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
import tma_gait_gauge_validation as g
from tma_gait_fast_exact import fast_level_sets,assert_equivalent

OUT=Path('research/tma_physio_spectral_results')
OUT.mkdir(exist_ok=True,parents=True)
SEED=20261008
B=79
BLOCK_B=39
BANDS=[('64_stride',1,1),('32_stride',2,2),('16to21_stride',3,4),
       ('8to13_stride',5,8),('4to7_stride',9,16),
       ('2to4_stride',17,32)]
K=np.arange(1,33,dtype=float)
F=K/64.
TMA=('H','D')
def logpow(x):
    """Power weighted by actual onset and TMA stack, no periodic resampling."""
    rows,levels=fast_level_sets(x)
    times=np.array([float(t)/2 for t,i,q,l in rows],float)
    height=np.array([max(levels[t]) for t,i,q,l in rows],float)
    density=np.array([len(levels[t]) for t,i,q,l in rows],float)
    # Mean-centering removes the average Level elevation, not event ordering.
    exp=np.exp(-2j*np.pi*np.outer(K/64.,times))
    out={}
    for name,arr in [('H',height),('D',density)]:
        weights=arr-arr.mean()
        power=np.abs(exp@weights)**2
        power=np.maximum(power,1e-10)
        out[name]={label:float(np.log(power[a-1:b].mean()))
           for label,a,b in BANDS}
    return out

def conventional(x):
    ph=np.array([[float(e['raw_phase']) for e in c['events'][1:]]
         for c in x],float)
    stride=np.array([c['duration_s'] for c in x],float)
    spectra=[]
    for a in (ph[:,0],ph[:,1],ph[:,2],stride):
        fft=np.fft.rfft(a-a.mean())
        spectra.append(np.abs(fft)**2)
    s=np.mean(np.stack(spectra,axis=0),axis=0)
    conventional={label:float(np.log(np.maximum(np.mean(s[a:b+1]),1e-12)))
          for label,a,b in BANDS}
    names=['RTO','RHS','LTO']
    standard={}
    for j,name in enumerate(names):
        standard[name+'_mean']=float(ph[:,j].mean())
        standard[name+'_sd']=float(ph[:,j].std(ddof=1))
        standard[name+'_acf1']=float(np.corrcoef(ph[:-1,j],ph[1:,j])[0,1]) if ph[:,j].std()>1e-9 else 0.
    standard['stride_mean']=float(stride.mean())
    standard['stride_cv']=float(stride.std(ddof=1)/stride.mean())
    standard['quant_depth']=float(np.mean([e['projection_depth'] for c in x for e in c['events'][1:]]))
    return conventional,standard

def block_shuffle(c,rng):
    groups=[c[i:i+4] for i in range(0,len(c),4)]
    return [v for j in rng.permutation(len(groups)) for v in groups[int(j)]]

def record(item):
    r,gr,i=item
    a=g.download_record(r)
    c,fs,integ=g.extract_cycles(a,r)
    rng=np.random.default_rng(SEED+7177*i)
    actual=logpow(c)
    plain,cov=conventional(c)
    null=[logpow(g.cycle_order_scramble(c,rng)) for _ in range(B)]
    block=[logpow(block_shuffle(c,rng)) for _ in range(BLOCK_B)]
    result={'record':r,'group':gr,**cov}
    for label,a,b in BANDS:
        result['rawspec_'+label]=plain[label]
        for field in TMA:
            v=actual[field][label]
            result[f'{field}_{label}']=v
            for name,s in [('cycle',null),('block',block)]:
                values=np.array([x[field][label] for x in s])
                m=float(np.mean(values));sd=float(np.std(values,ddof=1))
                result[f'{field}_{label}_{name}_resid']=v-m
                result[f'{field}_{label}_{name}_z']=(v-m)/max(sd,.03)
                result[f'{field}_{label}_{name}_p']=(1+sum(abs(values-m)>=abs(v-m)))/(len(values)+1)
    return result

def permutation_group(y,labels,cov=None,Bp=2999):
    y=np.asarray(y,float);label=np.asarray(labels,int)
    if cov is None:
        stat=float(y[label==1].mean()-y[label==0].mean())
        rng=np.random.default_rng(SEED+91)
        p=(1+sum(abs(y[perm==1].mean()-y[perm==0].mean())>=abs(stat)
                for perm in (rng.permutation(label) for _ in range(Bp))))/(Bp+1)
        return {'diff':stat,'permutation_p':p}
    X0=np.column_stack([np.ones(len(y)),cov])
    X1=np.column_stack([np.ones(len(y)),label,cov])
    def calc(X,y):
        coef,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        resid=y-X@coef
        inv=np.linalg.pinv(X.T@X)
        mse=float(np.sum(resid*resid)/max(len(y)-X.shape[1],1))
        return coef,np.asarray(resid),coef[1]/max(np.sqrt(mse*inv[1,1]),1e-10)
    beta,_,t=calc(X1,y)
    coef0,red,_=calc(np.column_stack([np.ones(len(y)),cov[:,0]]),y) if False else (None,None,None)
    b0=np.linalg.lstsq(X0,y,rcond=None)[0]
    fit=X0@b0
    residual=y-fit
    rng=np.random.default_rng(SEED+311)
    exceed=0
    for i in range(Bp):
        ys=fit+rng.permutation(residual)
        _,_,tp=calc(X1,ys)
        exceed+=abs(tp)>=abs(t)
    return {'adjusted_PD_difference':float(beta[1]),'t':float(t),
       'Freedman_Lane_p':float((exceed+1)/(Bp+1))}

def loocv_auc(df,keys,reg=0.25):
    x=df[keys].to_numpy(float)
    y=(df.group=='pd').astype(int).to_numpy()
    vals=[]
    for i in range(len(y)):
        tr=np.arange(len(y))!=i
        m=make_pipeline(StandardScaler(),LogisticRegression(C=reg,max_iter=500))
        m.fit(x[tr],y[tr])
        vals.append(float(m.predict_proba(x[i:i+1])[:,1][0]))
    pred=np.asarray(vals)
    return {'AUC':float(roc_auc_score(y,pred)),
         'balanced_accuracy':float(balanced_accuracy_score(y,pred>=.5)),
         'n_features':len(keys),'predictions':vals}

def main():
    t=time.monotonic()
    for rec in (g.CONTROL_IDS[0],g.PD_IDS[0]):
        a=g.download_record(rec);c,*_=g.extract_cycles(a,rec)
        for n in (16,32,64):assert_equivalent(c[:n],f'{rec}-{n}')
    print('PASS 6 frozen-engine equivalences',flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        jobs={ex.submit(record,(r,grp,i)):r for i,(r,grp) in enumerate(g.RECORDS)}
        for j,f in enumerate(as_completed(jobs),1):
            results.append(f.result())
            print(f'participant {j}/47 {jobs[f]}',flush=True)
    df=pd.DataFrame(results).sort_values('record').reset_index(drop=True)
    df.to_csv(OUT/'gait_spectral_features.csv',index=False)
    lab=(df.group=='pd').astype(int).to_numpy()
    out={'N':len(df),'group_counts':df.group.value_counts().to_dict(),
        'B_cycle':B,'B_block':BLOCK_B,'bands':BANDS,'results':{},'comparators':{}}
    tests=[]
    for name,a,b in BANDS:
        group=permutation_group(df['rawspec_'+name],lab)
        out['comparators'][name]=group
        for feature in TMA:
            k=f'{feature}_{name}_cycle_resid'
            u=permutation_group(df[k],lab)
            m=df[['RTO_mean','RHS_mean','LTO_mean','RTO_sd','RHS_sd','LTO_sd',
                   'stride_cv','quant_depth','rawspec_'+name]].to_numpy(float)
            # standardize nuisance covariates globally, independent of labels
            m=(m-m.mean(axis=0))/(m.std(axis=0)+1e-8)
            adj=permutation_group(df[k],lab,cov=m)
            out['results'][k]={'control_mean':float(df.loc[df.group=='control',k].mean()),
                 'PD_mean':float(df.loc[df.group=='pd',k].mean()),
                 'group':u,'adjusted_for_conventional':adj}
            tests.append(k)
    p=[out['results'][k]['group']['permutation_p'] for k in tests]
    padj=multipletests(p,method='fdr_bh')[1]
    for key,q in zip(tests,padj):out['results'][key]['q_BH12']=float(q)
    padj=multipletests([out['results'][k]['adjusted_for_conventional']['Freedman_Lane_p'] for k in tests],method='fdr_bh')[1]
    for key,q in zip(tests,padj):out['results'][key]['q_adjusted_BH12']=float(q)
    # the predicted group is not a disease diagnosis tool; all exploratory
    plain=['RTO_mean','RHS_mean','LTO_mean','RTO_sd','RHS_sd','LTO_sd',
           'RTO_acf1','RHS_acf1','LTO_acf1','stride_mean','stride_cv','quant_depth']
    conv=plain+['rawspec_'+a for a,_,_ in BANDS]
    spectral=[f'{v}_{a}_cycle_resid' for a,_,_ in BANDS for v in TMA]
    combos={'phase_and_stride':plain,'conventional_spectral':conv,
           'TMA_corrected_spectral':spectral,'conventional_plus_TMA':conv+spectral}
    cv={name:loocv_auc(df,features) for name,features in combos.items()}
    rng=np.random.default_rng(SEED+222)
    y=lab;p1=np.array(cv['conventional_spectral']['predictions']);p2=np.array(cv['conventional_plus_TMA']['predictions'])
    d=[]
    for i in range(4999):
        ix=rng.choice(len(y),size=len(y),replace=True)
        if len(np.unique(y[ix]))!=2:continue
        d.append(float(roc_auc_score(y[ix],p2[ix])-roc_auc_score(y[ix],p1[ix])))
    out['cv_AUC_gain']={'difference':float(cv['conventional_plus_TMA']['AUC']-cv['conventional_spectral']['AUC']),
         'bootstrap95':np.quantile(d,[.025,.975]).tolist()}
    for v in cv.values():v.pop('predictions')
    out['cv']=cv
    out['runtime_s']=time.monotonic()-t
    out['limits']=['64-cycle gait sample, group comparison observational','Fourier spectrum of attack-weighted H/D not a unique wave interpolation',
      'Cycle-permutation baseline preserves phase multiset and event order within strides',
      'Different frequency windows exploratory, twelve comparisons FDR corrected',
      'Conventional covariates include cadence/phase spread and conventional phase spectra',
      'No longitudinal PD severity testing or causal attribution']
    (OUT/'gait_results.json').write_text(json.dumps(out,indent=2))
    print('RESULT',json.dumps({'results':out['results'],'cv':out['cv'],'cv_AUC_gain':out['cv_AUC_gain']},indent=2),flush=True)
if __name__=='__main__':main()
