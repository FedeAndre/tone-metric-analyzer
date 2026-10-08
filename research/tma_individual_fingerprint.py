#!/usr/bin/env python3
"""Clinical-severity and individual-repeatability validation for TMA Fourier
fingerprints. Real Ga cohort, no PD-vs-HC classification as outcome.

1. Full 64-stride H/D wave Fourier powers at prespecified 4 frequency bands.
2. 39 within-trial cycle-permutation baselines remove fixed fractal geometry.
3. PD-only clinical correlation (motor UPDRS primary; total UPDRS, Hoehn-Yahr,
   timed-up-and-go and speed secondary), FDR / partial controls.
4. Strict leave-one-PD-person-out ridge regression vs timing/PSD model.
5. Independent recorded walks (_01/_02) identification among eligible subjects,
   plus 32+32 cycle repeatability without copying a subject into another's folds.
6. Timing / event-order preservation and exclusion documented.

No claims of a new spectral information source: all deterministic of gait event times.
"""
import io, json, re, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np, pandas as pd, requests
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tma_gait_gauge_validation as g
from tma_gait_fast_exact import fast_level_sets,assert_equivalent

warnings.filterwarnings('ignore',category=RuntimeWarning)
OUT=Path('research/tma_individual_fingerprint_results')
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261008
N=64
NNULL=39
NNULL_REPEAT=19
NNULL_HALF=19
# unit: cycles/stride, uppermost 0.5 cycles/stride
EDGES=(1/32,1/16,1/8,1/4,1/2+1e-8)
BANDS=('16to32_strides','8to16_strides','4to8_strides','2to4_strides')
SIGNALS=('H','D')
SPECTRAL=[f'{v}_{k}' for v in SIGNALS for k in BANDS]
RAWSD=['stride_mean','stride_cv','phase_sd','phase_acf','stance_mean','phase_mean']

def mean(x):return float(np.mean(x))
def clinical_metadata():
    r=requests.get('https://physionet.org/files/gaitpdb/1.0.0/demographics.txt',timeout=50)
    r.raise_for_status()
    tab=pd.read_csv(io.StringIO(r.text),sep='\t',engine='python',on_bad_lines='skip')
    assert len(tab)>=150 and {'ID','UPDRS','UPDRSM','HoehnYahr','TUAG','Age','Gender'}.issubset(tab.columns)
    return tab.set_index('ID')

def available_second(base):
    p=g.CACHE/(base+'_02.txt')
    if p.exists():return True
    url=g.BASE_URLS[0].format(record=base+'_02')
    try:
        r=requests.head(url,timeout=22,allow_redirects=True)
        return r.status_code==200 and int(r.headers.get('content-length',99999))>50000
    except Exception:return False

def timeseries(cycles):
    rows,lev=fast_level_sets(cycles)
    # full TMA wave event means, fixed origin; not a sliding-window restart
    heights=[[] for _ in cycles]
    density=[[] for _ in cycles]
    for t,i,q,label in rows:
        heights[i].append(max(lev[t]))
        density[i].append(len(lev[t]))
    assert all(len(x)==4 for x in heights) and all(len(x)==4 for x in density)
    return np.array([[mean(a) for a in heights],[mean(a) for a in density]],float)

def spectrum(cycles):
    S=timeseries(cycles)
    f=np.fft.rfftfreq(len(cycles),d=1.)
    out={}
    for idx,name in enumerate(SIGNALS):
        sig=S[idx]-np.mean(S[idx])
        sig=sig-np.polyval(np.polyfit(np.arange(len(sig)),sig,1),np.arange(len(sig)))
        p=np.abs(np.fft.rfft(sig))**2
        p[0]=0
        overall=mean(p[f>=EDGES[0]])
        for i,label in enumerate(BANDS):
            mask=(f>=EDGES[i]-1e-9)&(f<EDGES[i+1]-1e-9)
            if i==len(BANDS)-1:mask=(f>=EDGES[i]-1e-9)&(f<=.5+1e-9)
            assert mask.any()
            out[f'{name}_{label}']=float(np.log((np.mean(p[mask])+1e-12)/(overall+1e-12)))
    return out

def conventional(cycles):
    ph=np.array([[float(e['raw_phase'])/2 for e in c['events'][1:]] for c in cycles])
    stride=np.array([c['duration_s'] for c in cycles],float)
    arrays={'stride':stride,'RTO':ph[:,0],'RHS':ph[:,1],'LTO':ph[:,2]}
    X={}
    for name,x in arrays.items():
        x=x-mean(x)
        x=x-np.polyval(np.polyfit(np.arange(len(x)),x,1),np.arange(len(x)))
        f=np.fft.rfftfreq(len(x),1)
        power=np.abs(np.fft.rfft(x))**2;power[0]=0
        total=mean(power[f>=EDGES[0]])
        for i,band in enumerate(BANDS):
            q=(f>=EDGES[i]-1e-9)&(f<EDGES[i+1]-1e-9)
            if i==len(BANDS)-1:q=(f>=EDGES[i]-1e-9)&(f<=.5+1e-9)
            X[f'{name}_{band}']=float(np.log((mean(power[q])+1e-12)/(total+1e-12)))
    x=ph
    X.update({'stride_mean':mean(stride),
      'stride_cv':float(np.std(stride,ddof=1)/mean(stride)),
      'phase_sd':float(np.mean(np.std(x,axis=0,ddof=1))),
      'phase_acf':float(np.mean([np.corrcoef(x[:-1,j],x[1:,j])[0,1] if np.std(x[:,j])>1e-9 else 0 for j in range(3)])),
      'stance_mean':mean(x[:,1]),'phase_mean':mean(x)})
    return X

def correction(cycles,seed,nnull):
    original=spectrum(cycles)
    rng=np.random.default_rng(seed)
    perms=[]
    for k in range(nnull):perms.append(spectrum(g.cycle_order_scramble(cycles,rng)))
    out={};mean_s={}
    for key in SPECTRAL:
        null=np.array([a[key] for a in perms])
        mu=mean(null)
        # log-power relative to same multiset's scrambled expected spectral geometry
        out[key]=float(original[key]-mu)
        mean_s[key]=float(mu)
    return out,mean_s

def feature_record(record,idx,secondary=False):
    force=g.download_record(record)
    cyc,fs,integrity=g.extract_cycles(force,record)
    if idx in (0,20,34) and not secondary:
        for n in (16,32,64):assert_equivalent(cyc[:n],f'{record}-{n}')
        _,full=fast_level_sets(cyc)
        for n in (16,32,48):
            rows,lev=fast_level_sets(cyc[:n])
            assert all(full[t]==lev[t] for t,i,q,l in rows)
    corrected,nullmean=correction(cyc,SEED+idx*3049+(100000 if secondary else 0),
                                NNULL_REPEAT if secondary else NNULL)
    raw=conventional(cyc)
    key=record[:6]
    vals={'record':record,'subject':key,'trial':2 if secondary else 1,
          'group':'pd' if 'Pt' in record else 'control'}
    vals.update({f'tma_{k}':v for k,v in corrected.items()})
    vals.update({f'null_{k}':v for k,v in nullmean.items()})
    vals.update({f'conv_{k}':v for k,v in raw.items()})
    if not secondary:
        # Retain global TMA origin across the entire original recording, then
        # calculate two 32-cycle spectral sections with the exact same physical
        # start (no retroactive Level remapping).
        wave=timeseries(cyc)
        for half in range(2):
            section=cyc[half*32:(half+1)*32]
            # compare 32-cycle restarted absolute metrical coordinates only as
            # matched benchmark; each half's null is internally matched.
            cf,_=correction(section,SEED+idx*3109+half*11+200000,NNULL_HALF)
            cs=conventional(section)
            vals.update({f'half{half+1}_tma_{k}':v for k,v in cf.items()})
            vals.update({f'half{half+1}_conv_{k}':v for k,v in cs.items()})
    return vals

def centered_standard(X):
    x=np.asarray(X,float)
    mu=x.mean(axis=0);sd=x.std(axis=0);sd=np.maximum(sd,1e-8)
    return (x-mu)/sd

def id_reliability(df,kind,subset):
    records=df[df.group.eq(subset)] if subset in ('pd','control') else df
    p=records[records.trial==1].set_index('subject')
    q=records[records.trial==2].set_index('subject')
    ids=sorted(set(p.index)&set(q.index))
    if len(ids)<5:return {'n':len(ids)}
    cols=[f'{kind}_{key}' for key in SPECTRAL] if kind=='tma' else [f'{kind}_{name}_{band}' for name in ('stride','RTO','RHS','LTO') for band in BANDS]
    p=p.loc[ids];q=q.loc[ids]
    train=p[cols].to_numpy(float);test=q[cols].to_numpy(float)
    mu=train.mean(axis=0);sd=np.maximum(train.std(axis=0),1e-8)
    A=(train-mu)/sd;B=(test-mu)/sd
    # within-subject cosine similarity and nearest-neighbour identity test
    # cosine can be unstable for almost-zero corrected spectral profiles;
    # include Euclidean and separate normalization.
    dist=np.sqrt(np.maximum(((B[:,None,:]-A[None,:,:])**2).mean(axis=2),0))
    rank=[int(1+np.sum(dist[i,:]<dist[i,i])) for i in range(len(ids))]
    identity=sum(r==1 for r in rank)/len(ids)
    true=np.array([dist[i,i] for i in range(len(ids))]);other=np.array([dist[i,j] for i in range(len(ids)) for j in range(len(ids)) if i!=j])
    rng=np.random.default_rng(SEED+333)
    perm=[]
    for k in range(9999):
        shuffle=rng.permutation(len(ids))
        perm.append(float(np.mean([dist[i,shuffle[i]] for i in range(len(ids))])))
    obs=mean(true)
    return {'n':len(ids),'identity_top1':float(identity),
      'median_rank':float(np.median(rank)),'chance_top1':1/len(ids),
      'same_subject_mean_distance':obs,'other_subject_mean_distance':mean(other),
      'random_match_distance_p':float((1+np.sum(np.array(perm)<=obs))/(len(perm)+1)),
      'n_correct':sum(r==1 for r in rank)}

def half_reliability(df,kind,subset='pd'):
    rec=df[(df.group==subset)&(df.trial==1)] if subset else df[df.trial==1]
    if len(rec)<5:return {}
    cols1=[f'half1_{kind}_{k}' for k in SPECTRAL] if kind=='tma' else [f'half1_{kind}_{m}_{b}' for m in ('stride','RTO','RHS','LTO') for b in BANDS]
    cols2=[k.replace('half1_','half2_') for k in cols1]
    a=rec[cols1].to_numpy(float);b=rec[cols2].to_numpy(float)
    d=np.maximum(a.std(axis=0),1e-8)
    mu=a.mean(axis=0)
    a=(a-mu)/d;b=(b-mu)/d
    dist=np.sqrt(np.mean((b[:,None,:]-a[None,:,:])**2,axis=2))
    return {'n':len(rec),'top1':float(np.mean(np.argmin(dist,axis=1)==np.arange(len(rec)))),
            'within':float(np.mean(np.diag(dist))),'other':float(np.mean(dist[~np.eye(len(rec),dtype=bool)]))}

def spearman_tests(df,measure,features):
    subset=df.loc[np.isfinite(df[measure])]
    y=subset[measure].to_numpy(float)
    arr=[]
    for feature in features:
        x=subset[feature].to_numpy(float)
        rho=float(stats.spearmanr(x,y).statistic)
        rng=np.random.default_rng(SEED+913)
        perm=np.array([stats.spearmanr(x,rng.permutation(y)).statistic for k in range(4999)])
        pp=float((1+np.sum(np.abs(perm)>=abs(rho)-1e-12))/(len(perm)+1))
        # covariate-adjusted partial rank correlation (clinical marker vs TMA)
        # age, sex, stride CV, phase variability, conventional spectral power.
        band=next((b for b in BANDS if b in feature),BANDS[0])
        xcontrols=['Age','Gender','conv_stride_cv','conv_phase_sd',
                 'conv_stride_'+band,'conv_RHS_'+band]
        control=subset[xcontrols].to_numpy(float)
        rank=lambda v:stats.rankdata(v)
        yrank=rank(y);xrank=rank(x)
        C=np.column_stack([np.ones(len(y)),centered_standard(control)])
        xr=xrank-C@np.linalg.lstsq(C,xrank,rcond=None)[0]
        yr=yrank-C@np.linalg.lstsq(C,yrank,rcond=None)[0]
        partial=float(np.corrcoef(xr,yr)[0,1])
        # label permutation after covariate residualization (Freedman-Lane-like)
        rng=np.random.default_rng(SEED+1911)
        pp_adj=(1+sum(abs(np.corrcoef(xr,rng.permutation(yr))[0,1])>=abs(partial)-1e-12
           for _ in range(4999)))/5000
        arr.append({'feature':feature,'rho':rho,'p':pp,
                    'partial_rank_r':partial,'partial_permutation_p':pp_adj,'n':len(y)})
    q=multipletests([x['p'] for x in arr],method='fdr_bh')[1]
    qa=multipletests([x['partial_permutation_p'] for x in arr],method='fdr_bh')[1]
    for a,b,c in zip(arr,q,qa):a['q8']=float(b);a['adjusted_q8']=float(c)
    return sorted(arr,key=lambda x:x['p'])

def loocv(df,target,cols,alpha=20):
    subset=df[np.isfinite(df[target])].reset_index(drop=True)
    Y=subset[target].to_numpy(float)
    X=subset[cols].to_numpy(float)
    pred=[]
    for i in range(len(Y)):
        tr=np.arange(len(Y))!=i
        xtr=X[tr];xte=X[~tr]
        mdl=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        mdl.fit(xtr,Y[tr])
        pred.append(float(mdl.predict(xte)[0]))
    return subset,Y,np.asarray(pred)

def evaluate(df,outcome,features,alphas=(5,20,80)):
    res={}
    for alpha in alphas:
        res[str(alpha)]={}
        preds={}
        for key,cols in features.items():
            sub,y,p=loocv(df,outcome,cols,alpha)
            pred_name=key
            preds[key]=p
            denom=float(np.sum((y-y.mean())**2))
            res[str(alpha)][key]={'mae':float(np.mean(abs(y-p))),
              'r2':float(1-sum((y-p)**2)/denom),
              'pearson':float(np.corrcoef(y,p)[0,1]),'n':len(y)}
        # mean-of-other-people proper LOOCV null
        y=sub[outcome].to_numpy(float)
        base=np.array([(sum(y)-yi)/(len(y)-1) for yi in y])
        res[str(alpha)]['LOOCV_mean']={'mae':float(np.mean(abs(y-base))),
            'r2':float(1-sum((y-base)**2)/sum((y-y.mean())**2))}
        rng=np.random.default_rng(SEED+171)
        errors=np.abs(y-preds['timing'])-np.abs(y-preds['timing_plus_TMA'])
        gain=float(mean(errors));boot=[]
        for k in range(4999):
            ix=rng.choice(len(y),len(y),replace=True)
            boot.append(float(mean(errors[ix])))
        res[str(alpha)]['incremental_TMA']={'mae_gain_positive_is_better':gain,
             'bootstrap95':[float(v) for v in np.quantile(boot,[.025,.975])],
             'bootstrapP_gain_nonpositive':float(np.mean(np.array(boot)<=0))}
    return res

def main():
    tick=time.monotonic()
    meta=clinical_metadata()
    records=[x for x,_ in g.RECORDS]
    # same-person second walking trial _02, no _10 condition change.
    eligible=[r for r in records if available_second(r[:6])]
    print('Primary recordings',len(records),'second trial accessible',len(eligible),flush=True)
    tasks=[(r,i,False) for i,r in enumerate(records)]+[(r[:6]+'_02',i,True) for i,r in enumerate(records) if r in eligible]
    result=[];failed=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(feature_record,*t):t[0] for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            try:result.append(f.result())
            except Exception as e:
                failed.append({'record':futures[f],'error':str(e)})
            print('processed',i,'/',len(tasks),'record',futures[f], 'success',not f.exception(),flush=True)
    df=pd.DataFrame(result)
    for name in ('UPDRS','UPDRSM','HoehnYahr','TUAG','Age','Gender','Speed_01'):
        df[name]=df.subject.map(meta[name])
    df.sort_values(['subject','trial'],inplace=True)
    df.to_csv(OUT/'subject_spectral_fingerprints.csv',index=False)
    pdonly=df[(df.group=='pd')&(df.trial==1)].copy()
    clinical_counts={name:int(pdonly[name].notna().sum()) for name in ('UPDRS','UPDRSM','HoehnYahr','TUAG','Speed_01')}
    tma=[f'tma_{k}' for k in SPECTRAL]
    conventional_freq=[f'conv_{x}_{b}' for x in ('stride','RTO','RHS','LTO') for b in BANDS]
    basic=['Age','Gender']
    conventional=basic+[f'conv_{x}' for x in RAWSD]+conventional_freq
    modes={'demographic':basic,'timing':conventional,'TMA_only':basic+tma,'timing_plus_TMA':conventional+tma}
    rankcorr={m:spearman_tests(pdonly,m,tma) for m in ('UPDRSM','UPDRS','HoehnYahr','TUAG','Speed_01') if pdonly[m].notna().sum()>=18}
    model={m:evaluate(pdonly,m,modes) for m in ('UPDRSM','UPDRS','HoehnYahr','TUAG','Speed_01') if pdonly[m].notna().sum()>=18}
    fingerprint={}
    for subset in ('pd','control','all'):
        for representation in ('tma','conv'):
            fingerprint[f'{representation}_{subset}']=id_reliability(df,representation,subset)
    halves={representation:half_reliability(df,representation,'pd') for representation in ('tma','conv')}
    outcome={'study':'Ga study, PhysioNet Gait Parkinson Disease gaitpdb/1.0.0',
      'n_pd_baseline':len(pdonly),'n_recordings':len(df),'second_eligible':len(eligible),
      'n_repeated_extracted':len(df[df.trial==2]),
      'failed':failed,'n_cycles':64,'n_cycle_scrambles_primary':NNULL,
      'n_cycle_scrambles_second':NNULL_REPEAT,'n_cycle_scrambles_half':NNULL_HALF,
      'band_definitions_cycles_per_stride':dict(zip(BANDS,[list((EDGES[i],EDGES[i+1])) for i in range(4)])),
      'individual_reliability':fingerprint,'half_reliability':halves,
      'clinical_counts':clinical_counts,'clinical_correlations':rankcorr,
      'LOOCV_clinical_model':model,'seconds':time.monotonic()-tick,
      'limitations':['Only Ga study from three-study PhysioNet database, no Ju/Si raw trials',
       'Motor UPDRS was a cross-sectional disease-severity measure; not longitudinal progression',
       'Repeat trials 01/02 may have recording condition and trial effects',
       'Binary TMA engine from previously regression-tested deterministic implementation',
       'Within trial 32-cycle halves are evaluated after separate finite-interval fractal nulls; no claims of full-origin half causality',
       'Single-label covariate and event thresholds in code, multiple exploratory comparisons',
       'Power spectra discard phase; four prechosen bands only; no discrimination label used as dependent variable']}
    (OUT/'results.json').write_text(json.dumps(outcome,indent=2))
    print('FINAL_RESULT',json.dumps({k:outcome[k] for k in ('n_pd_baseline','n_recordings',
       'n_repeated_extracted','failed','clinical_counts','individual_reliability',
       'half_reliability','clinical_correlations','LOOCV_clinical_model')},indent=2),flush=True)
if __name__=='__main__':main()
