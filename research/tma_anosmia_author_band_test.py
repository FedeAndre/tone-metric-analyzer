#!/usr/bin/env python3
"""Band-specific TMA association with congenital anosmia using public nasal-airflow
recordings of Gorodisky et al., 2024.

Respiratory major inhalation peaks supply tactus; smaller same-polarity nasal
flow peaks supply variable rhythm. No artificial regular sample grid as a beat.
128 consecutive detected breath cycles per individual, frozen binary 1/4 TMA.
Surrogates shuffle complete breaths and preserve all within-breath events.
Assesses ~64,32,16,8,4,2-breath timescales. This is exploratory, subject-level.
"""
import io,json,math,time,csv,collections
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
from statsmodels.stats.multitest import multipletests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from tone_metric.engine import recursive_integer_structure,_next_seq_anchor,analyze
from tone_metric.models import Hit,MeasureInfo

OUT=Path('research/tma_anosmia_author_results')
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261008
N=128
B=39
B_BLOCK=19
BANDS=[('64to128_breath',1,2),('32to42_breath',3,4),
       ('16to25_breath',5,8),('8to14_breath',9,16),
       ('4to7_breath',17,32),('2to4_breath',33,64)]
DATA_ROOT='https://gitlab.com/liorg/anosmics-breathe-differently/-/raw/main/Data/'
META_URL=DATA_ROOT+'Participants.csv'
FIELDS=('H','D')

def mean(a):return float(np.mean(a))
def sd(a):return float(np.std(a,ddof=1)) if len(a)>1 else 0.

def fetch(record):
    r=requests.get(DATA_ROOT+quote(record)+'.mat',timeout=90)
    r.raise_for_status()
    d=loadmat(io.BytesIO(r.content))
    z=np.asarray(d['Data'],float)
    fs=1/float(np.asarray(d['SampleLength']).item())
    awake=[(0,int(np.asarray(d['SleepIndex']).item()),'pre_sleep'),
       (int(np.asarray(d['WakeUpIndex']).item()),len(z),'post_wake')]
    return z,fs,awake,len(r.content)

def get_segment(data,fs,awake):
    """Choose one signal-quality-controlled wake window; selection blind to diagnosis."""
    # Normalize left and right nostrils separately, as in the authors' GetPeaksData.m\n    scale_channels=5./np.maximum(np.ptp(data,axis=0),1e-8)\n    z=(data*scale_channels).sum(axis=1)
    candidates=[]
    wsize=int(fs*1800)
    for start,end,label in awake:
        start=start+int(fs*60)
        end=end-int(fs*60)
        if end-start<int(fs*720):continue
        length=min(wsize,end-start)
        for fraction in (0.,.33,.66,1.):
            left=start+int((end-start-length)*fraction)
            x=z[left:left+length]
            if len(x)<int(fs*720):continue
            try:
                slow=signal.sosfiltfilt(signal.butter(3,[.08,.52],fs=fs,btype='bandpass',output='sos'),x)
                fast=signal.sosfiltfilt(signal.butter(3,[.08,1.85],fs=fs,btype='bandpass',output='sos'),x)
            except ValueError:continue
            scale=1.4826*np.median(np.abs(slow-np.median(slow)))
            if not np.isfinite(scale) or scale<.04:continue
            major,_=signal.find_peaks(slow,distance=max(int(fs*2.),1),prominence=.65*scale)
            if len(major)<N+1:continue
            dura=np.diff(major)/fs
            good=(dura>=1.9)&(dura<=8)
            # Longest uninterrupted stretch of plausibly spaced major peaks
            edges=np.flatnonzero(np.r_[True,~good,True])
            spans=[]
            curr=0
            for i,v in enumerate(good):
                if v:
                    curr+=1
                else:
                    if curr:spans.append((i-curr,i,curr))
                    curr=0
            if curr:spans.append((len(good)-curr,len(good),curr))
            if not spans:continue
            ii,jj,ll=max(spans,key=lambda p:p[2])
            if ll<N:continue
            cycmajor=major[ii:ii+N+1]
            med_d=float(np.median(np.diff(cycmajor)/fs))
            if not (2.15<=med_d<=5.9):continue
            # Prefer strong, uninterrupted respiration and plausible breathing rate.
            quality=math.log(max(scale,1e-8))+2*ll/len(good) + .15*math.log(ll)
            candidates.append((quality,slow,fast,scale,cycmajor,fs,label,med_d,left))
    if not candidates:raise ValueError('No valid 128-breath wake segment after signal QC')
    return max(candidates,key=lambda x:x[0])

def extract(data,fs,awake):
    quality,slow,fast,scale,major,fs,section,period,start=get_segment(data,fs,awake)
    # Published GetPeaksData.m: findpeaks(Data,'MinPeakDistance',0.5*period,\n    # 'MinPeakProminence',0.1), followed by height >0.1 on normalized two-nostril flow.\n    normalized=(data*(5./np.maximum(np.ptp(data,axis=0),1e-8))).sum(axis=1)\n    segment=normalized[start:start+len(fast)]\n    extras,_=signal.find_peaks(segment,prominence=.1,\n        distance=max(1,int(.5*period*fs)),height=.1)
    events=[]
    rows=[]
    for i,(a,b) in enumerate(zip(major[:-1],major[1:])):
        minor=extras[(extras>a+max(int(.40*fs),1))&(extras<b-max(int(.35*fs),1))]
        minor=minor[segment[minor]>.1]
        phases=[]
        for p in minor:
            q=Fraction(int(round(64*(p-a)/(b-a))),64)
            if 0<q<1:phases.append(q)
        phases=sorted(set(phases))
        ev=[Fraction(0)]+phases[:5]
        events.append({'q':ev,
          'duration':float((b-a)/fs),
          'amp':float(np.max(fast[a:b])-np.min(fast[a:b])),
          'minor':len(ev)-1,
          'mean_minor_phase':float(np.mean([float(x) for x in phases])) if phases else 0.0})
    return events,{'selected_wake_section':section,'median_cycle_seconds':period,
        'amplitude_scale':float(scale),'mean_minor_peaks':mean([c['minor'] for c in events]),
        'fraction_multi':mean([c['minor']>0 for c in events]),
        'quality_score':quality,'n_cycles':len(events)}

def level_map(cycles):
    n=len(cycles)
    rows=sorted([(Fraction(i)+q,i,q) for i,c in enumerate(cycles) for q in c['q']])
    actual={t for t,i,q in rows}
    levels={t:set() for t in actual}
    grid={};interval={}
    recursive_integer_structure(1,_next_seq_anchor(2,n+1),2,1,grid,interval)
    for pos,lev in grid.items():
        if pos<=n+1:
            t=Fraction(pos-1)
            if t in actual:levels[t].update(lev)
    bybeat=collections.defaultdict(list)
    for t in actual:
        b=int(t)
        if b<n:bybeat[b].append(t)
    def recurse(a,b,es,parent):
        inner=[t for t in es if a<t<b]
        if not inner:return
        mid=(a+b)/2
        lvl=int(parent)+1
        for t in (a,mid,b):
            if t in actual:levels[t].add(lvl)
        L=[t for t in es if a<=t<=mid]
        R=[t for t in es if mid<=t<=b]
        if any(a<t<mid for t in L):recurse(a,mid,L,lvl)
        if any(mid<t<b for t in R):recurse(mid,b,R,lvl)
    for i,ev in bybeat.items():
        if any(Fraction(i)<t<Fraction(i+1) for t in ev):
            recurse(Fraction(i),Fraction(i+1),ev,int(interval.get(i+1,1)))
    if any(not v for v in levels.values()):raise RuntimeError('Unresolved event')
    return rows,levels

def validate(cycles):
    for n in (8,32,128):
        sub=cycles[:n]
        rows,ours=level_map(sub)
        hits=[];bars=[]
        for i,c in enumerate(sub):
            bars.append(MeasureInfo(index=i,number=str(i+1),start=Fraction(i),
                 full_duration=Fraction(1),actual_duration=Fraction(1),pickup_shift=Fraction(0),
                 numerator=1,denominator=4,implicit=False,opening_anacrusis=False))
            for q in c['q']:
                hits.append(Hit(onset=Fraction(i)+q,duration=Fraction(0),
                   measure_index=i,measure_number=str(i+1),offset_in_measure=q,
                   sources=[],canonical_recovered=False))
        hits.sort(key=lambda h:h.onset)
        ref=analyze(hits,bars)
        actual={Fraction(x['onset_quarter']):set(x['tone_metric_levels'])
                for s in ref['segments'] for x in s['events']}
        assert actual==ours,f'frozen engine fails at {n}, {len(actual)} vs {len(ours)}'
    fullrows,full=level_map(cycles)
    for n in (8,32,64):
        rows,lev=level_map(cycles[:n])
        assert all(full[t]==lev[t] for t,i,q in rows),'prefix instability'

def powers(cycles):
    rows,lev=level_map(cycles)
    H=[[] for _ in range(len(cycles))]
    D=[[] for _ in range(len(cycles))]
    for t,i,q in rows:
        H[i].append(max(lev[t]))
        D[i].append(len(lev[t]))
    out={}
    for key,seq in [('H',np.array([mean(v) for v in H])),
                    ('D',np.array([mean(v) for v in D]))]:
        s=np.abs(np.fft.rfft(seq-seq.mean()))**2
        out[key]={label:float(np.log(max(float(s[a:b+1].mean()),1e-10)))
             for label,a,b in BANDS}
    return out

def raw_spectra(c):
    count=np.array([e['minor'] for e in c],float)
    duration=np.array([e['duration'] for e in c],float)
    amps=np.array([e['amp'] for e in c],float)
    meanph=np.array([e['mean_minor_phase'] for e in c],float)
    X=[count,duration,amps,meanph]
    power=[]
    for x in X:
        x=(x-x.mean())/(x.std()+1e-10)
        power.append(np.abs(np.fft.rfft(x))**2)
    mix=np.mean(power,axis=0)
    return {label:float(np.log(max(float(mix[a:b+1].mean()),1e-10)))
         for label,a,b in BANDS}

def shuffles(c,rng,mode):
    if mode=='cycle':return [c[int(i)] for i in rng.permutation(N)]
    groups=[c[i:i+4] for i in range(0,N,4)]
    return [v for i in rng.permutation(len(groups)) for v in groups[int(i)]]

def record(item):
    row,i=item
    code=row['Code']
    x,fs,awake,nbytes=fetch(code)
    events,qc=extract(x,fs,awake)
    if i in (0,22):validate(events)
    orig=powers(events)
    conv=raw_spectra(events)
    rng=np.random.default_rng(SEED+i*1117)
    null=[powers(shuffles(events,rng,'cycle')) for _ in range(B)]
    block=[powers(shuffles(events,rng,'block')) for _ in range(B_BLOCK)]
    count=np.array([e['minor'] for e in events],float)
    times=np.array([e['duration'] for e in events],float)
    amps=np.array([e['amp'] for e in events],float)
    result={'record':code,'group':'anosmic' if row['Group'].lower()=='anosmic' else 'normosmic',
      'age':float(row['Age']),'is_male':float(row['Gender'].upper()=='M'),
      'fs':fs,'record_MB':nbytes/1e6,**qc,
      'breaths_per_minute':60/mean(times),'breath_CV':sd(times)/mean(times),
      'minor_peak_mean':mean(count),'minor_peak_sd':sd(count),
      'minor_peak_autocorr':float(np.corrcoef(count[:-1],count[1:])[0,1]) if np.std(count)>1e-9 else 0.,
      'amp_CV':sd(amps)/mean(amps)}
    for label,a,b in BANDS:
        result['rawspec_'+label]=conv[label]
        for field in FIELDS:
            observed=orig[field][label]
            result[f'{field}_{label}']=observed
            for name,ss in [('cycle',null),('block',block)]:
                vec=np.array([x[field][label] for x in ss],float)
                m=mean(vec);s=sd(vec)
                result[f'{field}_{label}_{name}_resid']=observed-m
                result[f'{field}_{label}_{name}_z']=(observed-m)/max(s,.03)
                result[f'{field}_{label}_{name}_p']=float((1+sum(np.abs(vec-m)>=abs(observed-m)))/(len(vec)+1))
    return result

def gp(y,g,cov=None,Bp=4999,seed=SEED):
    rng=np.random.default_rng(seed)
    y=np.asarray(y,float);grp=np.asarray(g,int)
    diff=mean(y[grp==1])-mean(y[grp==0])
    if cov is None:
        hit=0
        for i in range(Bp):
            p=rng.permutation(grp)
            hit+=abs(mean(y[p==1])-mean(y[p==0]))>=abs(diff)
        return {'difference':diff,'permutation_p':(1+hit)/(Bp+1)}
    C=np.asarray(cov,float)
    C=(C-C.mean(axis=0))/(C.std(axis=0)+1e-8)
    X0=np.column_stack([np.ones(len(y)),C])
    X1=np.column_stack([np.ones(len(y)),grp,C])
    def tstat(ys):
        b=np.linalg.lstsq(X1,ys,rcond=None)[0]
        eps=ys-X1@b
        V=np.linalg.pinv(X1.T@X1)
        mse=float(np.sum(eps*eps)/(len(ys)-X1.shape[1]))
        t=b[1]/max(np.sqrt(mse*V[1,1]),1e-10)
        return float(b[1]),float(t)
    coef,t=tstat(y)
    b0=np.linalg.lstsq(X0,y,rcond=None)[0]
    f=X0@b0
    e=y-f
    exceed=0
    for i in range(Bp):
        _,tp=tstat(f+rng.permutation(e))
        exceed+=abs(tp)>=abs(t)
    return {'adjusted_difference':coef,'t':t,'Freedman_Lane_p':(1+exceed)/(Bp+1)}

def cv_auc(df,features):
    x=df[features].to_numpy(float)
    y=(df.group=='anosmic').astype(int).to_numpy()
    scores=[]
    for i in range(len(y)):
        tr=np.arange(len(y))!=i
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1500))
        clf.fit(x[tr],y[tr])
        scores.append(float(clf.predict_proba(x[i:i+1])[:,1][0]))
    return {'AUC':float(roc_auc_score(y,scores)),
      'balanced_accuracy':float(balanced_accuracy_score(y,np.array(scores)>=.5)),
      'n_features':len(features),'scores':scores}

def main():
    clock=time.monotonic()
    r=requests.get(META_URL,timeout=60);r.raise_for_status()
    people=list(csv.DictReader(io.StringIO(r.text)))
    assert len(people)==52
    records=[];excluded=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        jobs={ex.submit(record,(row,i)):row['Code'] for i,row in enumerate(people)}
        for j,fut in enumerate(as_completed(jobs),1):
            try:records.append(fut.result())
            except Exception as e:\n                print('RECORD_FAILURE',jobs[fut],repr(e),flush=True)\n                excluded.append({'record':jobs[fut],'error':str(e)})
            print(f'participant {j}/52 {jobs[fut]} success {not fut.exception()}',flush=True)
    if not records:raise RuntimeError('No usable data, first errors: '+repr(excluded[:5]))\n    df=pd.DataFrame(records).sort_values('record').reset_index(drop=True)
    df.to_csv(OUT/'anosmia_spectral_features.csv',index=False)
    labels=(df.group=='anosmic').astype(int).to_numpy()
    out={'n':len(df),'groups':df.group.value_counts().to_dict(),'excluded':excluded,
         'n_cycles':N,'B_cycle':B,'B_block':B_BLOCK,'bands':BANDS,
         'QC':df[['breaths_per_minute','minor_peak_mean','fraction_multi','breath_CV','selected_wake_section','amplitude_scale']].describe(include='all').fillna(0).to_dict(),
         'raw_feature_comparisons':{},'TMA':{}}
    for cov in ('breaths_per_minute','minor_peak_mean','fraction_multi','breath_CV','minor_peak_sd','amp_CV'):
        out['raw_feature_comparisons'][cov]=gp(df[cov],labels)
    tested=[]
    for label,a,b in BANDS:
        out['raw_feature_comparisons']['PSD_'+label]=gp(df['rawspec_'+label],labels)
        for t in FIELDS:
            col=f'{t}_{label}_cycle_resid'
            adjcov=['age','is_male','fs','breaths_per_minute','breath_CV','minor_peak_mean',
                    'minor_peak_sd','amp_CV','rawspec_'+label]
            out['TMA'][col]={'simple':gp(df[col],labels),
                             'adjusted':gp(df[col],labels,df[adjcov])}
            tested.append(col)
    for mode in ['simple','adjusted']:
        p=[out['TMA'][k][mode]['permutation_p' if mode=='simple' else 'Freedman_Lane_p'] for k in tested]
        q=multipletests(p,method='fdr_bh')[1]
        for k,qq in zip(tested,q):out['TMA'][k][mode]['q_BH12']=float(qq)
    baseline=['age','is_male','fs','breaths_per_minute','breath_CV','minor_peak_mean',
              'minor_peak_sd','minor_peak_autocorr','amp_CV']
    conv=baseline+['rawspec_'+l for l,a,b in BANDS]
    tma=[f'{x}_{l}_cycle_resid' for l,a,b in BANDS for x in FIELDS]
    cv={n:cv_auc(df,f) for n,f in {
        'traditional_timing':baseline,'traditional_spectral':conv,
        'TMA_only':tma,'traditional_plus_TMA':conv+tma}.items()}
    rng=np.random.default_rng(SEED+2043)
    y=labels
    a=np.array(cv['traditional_spectral']['scores']);b=np.array(cv['traditional_plus_TMA']['scores'])
    delta=[]
    for i in range(4999):
        ix=rng.choice(len(y),len(y),replace=True)
        if len(np.unique(y[ix]))==2:delta.append(float(roc_auc_score(y[ix],b[ix])-roc_auc_score(y[ix],a[ix])))
    out['incremental_auc']={'delta':cv['traditional_plus_TMA']['AUC']-cv['traditional_spectral']['AUC'],
          'boot95':np.quantile(delta,[.025,.975]).tolist()}
    for v in cv.values():v.pop('scores')
    out['cv']=cv
    out['limits']=['Wake subsections chosen by signal QC without diagnosis labels',
      'Breath boundary detection is independently implemented; minor inhalation peaks approximate published GetPeaksData MATLAB normalization and findpeaks settings',
      'Temporal frequency in cycles/breath, with dyadic onset projection to 1/64 beat',
      'FFT is of per-breath aggregate full TMA event H and D, not direct event-time impulse wave',
      'Only 128 breaths per person, so long-scale resolution limited',
      'Surrogates preserve each breath-internal peak pattern while scrambling longer order',
      'No causal or clinical validation; 12 nominal spectral disease comparisons FDR corrected']
    out['runtime_seconds']=time.monotonic()-clock
    (OUT/'anosmia_results.json').write_text(json.dumps(out,indent=2))
    print('RESULT',json.dumps({'n':out['n'],'groups':out['groups'],'excluded':out['excluded'],'raw':out['raw_feature_comparisons'],
                              'TMA':out['TMA'],'cv':out['cv'],'incremental':out['incremental_auc']},indent=2),flush=True)
if __name__=='__main__':main()
