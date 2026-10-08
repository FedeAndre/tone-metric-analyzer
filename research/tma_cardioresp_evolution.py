#!/usr/bin/env python3
"""Origin-dependent TMA evolution in respiration-anchored ECG timing.
Heart beats are variable rhythm; successive detected breaths define tactus.
Independent respiration waveform shape is the evaluation target.
"""
import json
import math
import time
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

from tone_metric.engine import recursive_integer_structure,_next_seq_anchor,analyze
from tone_metric.models import Hit,MeasureInfo
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tma_physio_fantasia import bandpass_resp,respiratory_boundaries

OUT=Path('research/tma_cardioresp_evolution_results')
OUT.mkdir(parents=True,exist_ok=True)
BLOCK=8
POINTS=64
LIMIT_S=1200
RECORDS=[f'f1y{i:02d}' for i in range(1,7)]+[f'f1o{i:02d}' for i in range(1,7)]
SEED=20261008

TMA_KEYS=['Dmean','Dstd','Hmean','Hstd','multilevel','Hchange',
          'turnover','pivot_rate','pivot_depth','tree_span']
TEMP_KEYS=['duration_mean','duration_std','duration_acf',
           'n_beats_mean','n_beats_std','n_beats_acf',
           'rr_mean','rr_std','phase_sin','phase_cos','phase_R',
           'phase_entropy8','phase_entropy16','phase_lag1']

def phase_projection(sample,a,b,fs):
    raw=Fraction(sample-a,b-a)
    for depth in range(17):
        d=2**depth;q=Fraction(int(round(float(raw)*d)),d)
        if abs(float(q-raw)*(b-a)/fs)<=0.005000001:return q
    raise RuntimeError('Dyadic projection failed')

def read_data(rec):
    kw={'pn_dir':'fantasia'}
    head=wfdb.rdheader(rec,**kw)
    fs=float(head.fs)
    resp_index=[j for j,nm in enumerate(head.sig_name) if nm.upper()=='RESP'][0]
    n=min(int(LIMIT_S*fs),int(head.sig_len))
    record=wfdb.rdrecord(rec,channels=[resp_index],sampfrom=0,sampto=n,**kw)
    resp=np.asarray(record.p_signal[:,0],dtype=float)
    y=bandpass_resp(resp,fs)
    ann=wfdb.rdann(rec,'ecg',sampfrom=0,sampto=n,**kw)
    beats=np.asarray(ann.sample,dtype=int)
    bounds,_=respiratory_boundaries(y,fs,polarity=1)
    cycles=[]
    wave=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        dur=(b-a)/fs
        if not (1.5<=dur<=12):continue
        sel=beats[(beats>=a)&(beats<b)]
        if not (2<=len(sel)<=12):continue
        if np.any((np.diff(sel)/fs)<0.3) or np.any((np.diff(sel)/fs)>2):continue
        phases=[phase_projection(int(e),int(a),int(b),fs) for e in sel]
        if len(set(phases))!=len(phases):continue
        events=[{'q':q,'sample':int(s)} for q,s in zip(phases,sel)]
        idx=np.linspace(a,b-1,POINTS)
        waveform=np.interp(idx,np.arange(a,b),y[a:b])
        cycles.append({'start':int(a),'end':int(b),'duration':float(dur),
                       'events':events})
        wave.append(waveform)
        if len(cycles)>=64:break
    if len(cycles)<64:raise RuntimeError(f'{rec}: only {len(cycles)} respiration cycles')
    waves=np.array(wave[:64])
    scale=np.median(np.max(waves[:8],axis=1)-np.min(waves[:8],axis=1))
    waves/=max(scale,1e-9)
    return cycles[:64],waves

def level_map(cycles):
    """Exact binary, non-tuplet repository-engine specialization for 1/4 meter."""
    n=len(cycles)
    rows=sorted([(Fraction(i)+e['q'],i,e['q']) for i,c in enumerate(cycles) for e in c['events']])
    unique=sorted(set(t for t,i,q in rows))
    actual=set(unique)
    levels={t:set() for t in unique}
    grid={}
    parent={}
    recursive_integer_structure(1,_next_seq_anchor(2,n+1),2,1,grid,parent)
    for pos,ll in grid.items():
        if pos<=n+1:
            t=Fraction(pos-1)
            if t in actual:levels[t].update(int(x) for x in ll)
    byint={}
    for t in unique:
        f=int(math.floor(float(t)))
        for k in (f-1,f):
            if 0<=k<n and Fraction(k)<=t<=Fraction(k+1):
                byint.setdefault(k,[]).append(t)
    def refine(a,b,ev,p):
        interior=[t for t in ev if a<t<b]
        if not interior:return
        m=(a+b)/2
        lev=int(p)+1
        for t in (a,m,b):
            if t in actual: levels[t].add(lev)
        left=[t for t in ev if a<=t<=m]
        right=[t for t in ev if m<=t<=b]
        if any(a<t<m for t in left):refine(a,m,left,lev)
        if any(m<t<b for t in right):refine(m,b,right,lev)
    for i in range(n):
        ev=byint.get(i,[])
        if any(Fraction(i)<t<Fraction(i+1) for t in ev):
            refine(Fraction(i),Fraction(i+1),ev,int(parent.get(i+1,1)))
    if any(not levels[t] for t in unique):raise RuntimeError('Unresolved beat event')
    return rows,levels

def validate_engine(cycles):
    for n in (8,16,32,64):
        subset=cycles[:n]
        rows,levels=level_map(subset)
        hits=[];measures=[]
        for i,c in enumerate(subset):
            measures.append(MeasureInfo(index=i,number=str(i+1),start=Fraction(i),full_duration=Fraction(1),
                          actual_duration=Fraction(1),pickup_shift=Fraction(0),numerator=1,denominator=4,
                          implicit=False,opening_anacrusis=False))
            for e in c['events']:
                hits.append(Hit(onset=Fraction(i)+e['q'],duration=Fraction(0),measure_index=i,
                   measure_number=str(i+1),offset_in_measure=e['q'],sources=[],canonical_recovered=False))
        hits.sort(key=lambda e:e.onset)
        computed=[r for seg in analyze(hits,measures)['segments'] for r in seg['events']]
        assert len(computed)==len(rows)
        for (t,i,q),r in zip(rows,computed):
            assert levels[t]==set(r['tone_metric_levels']),(n,t,levels[t],r['tone_metric_levels'])
    _,whole=level_map(cycles)
    for n in (8,16,32,48):
        rows,partial=level_map(cycles[:n])
        assert all(whole[t]==partial[t] for t,i,q in rows)
    return True

def tma_features(cycles):
    rows,levelmap=level_map(cycles)
    wave=[]
    for t,i,q in rows:
        lev=sorted(levelmap[t])
        wave.append({'segment_index':0,'measure_index':i,'measure_number':str(i+1),
                     'onset_quarter':str(t),'offset_in_measure_quarter':str(q),
                     'levels':lev,'height':max(lev),'density':len(lev),
                     'lowest_level':min(lev),'attack':True,'parenthetical':False})
    piv=build_pivot_profile(wave)
    branches=build_tree_profile(wave)['branches']
    start=len(cycles)-BLOCK
    w=[p for p in wave if p['measure_index']>=start]
    H=np.array([p['height'] for p in w])
    D=np.array([p['density'] for p in w])
    p=[q for q in piv if q['pivot_measure_index']>=start]
    b=[q for q in branches if q['target_measure_index']>=start]
    tv=[]
    for x,y in zip(w[:-1],w[1:]):
        a,c=set(x['levels']),set(y['levels'])
        tv.append(1-len(a&c)/len(a|c))
    return {'Dmean':float(np.mean(D)),'Dstd':float(np.std(D,ddof=1)),
            'Hmean':float(np.mean(H)),'Hstd':float(np.std(H,ddof=1)),
            'multilevel':float(np.mean(D>1)),
            'Hchange':float(np.mean(np.abs(np.diff(H)))),
            'turnover':float(np.mean(tv)),'pivot_rate':len(p)/len(w),
            'pivot_depth':float(np.mean([q['drop_depth'] for q in p])) if p else 0.,
            'tree_span':float(np.mean([float(Fraction(q['distance_quarter'])) for q in b])) if b else 0.}

def temporal(cycles):
    v={}
    for name,x in [('duration',np.array([c['duration'] for c in cycles])),
                   ('n_beats',np.array([len(c['events']) for c in cycles]))]:
        v[name+'_mean']=float(x.mean());v[name+'_std']=float(x.std(ddof=1))
        v[name+'_acf']=float(np.corrcoef(x[:-1],x[1:])[0,1]) if np.std(x[:-1])>1e-8 and np.std(x[1:])>1e-8 else 0.
    rr=[]
    for c in cycles:rr+=list(np.diff([e['sample'] for e in c['events']])/250)
    v['rr_mean']=float(np.mean(rr));v['rr_std']=float(np.std(rr,ddof=1))
    ph=np.array([float(e['q']) for c in cycles for e in c['events']])
    angles=ph*2*np.pi
    v['phase_sin']=float(np.mean(np.sin(angles)))
    v['phase_cos']=float(np.mean(np.cos(angles)))
    v['phase_R']=float(np.hypot(v['phase_sin'],v['phase_cos']))
    for d in (8,16):
        bins=np.bincount(np.minimum(d-1,(ph*d).astype(int)),minlength=d)
        ps=bins[bins>0]/len(ph)
        v[f'phase_entropy{d}']=float(-np.sum(ps*np.log(ps))/np.log(d))
    v['phase_lag1']=float(np.corrcoef(ph[:-1],ph[1:])[0,1]) if np.std(ph[:-1])>1e-8 and np.std(ph[1:])>1e-8 else 0.
    return v

def one(rec):
    cycles,wave=read_data(rec)
    if rec in ('f1y01','f1o01'):validate_engine(cycles)
    orig=[];reset=[];conventional=[]
    for i in range(8):
        original=cycles[:(i+1)*BLOCK]
        window=cycles[i*BLOCK:(i+1)*BLOCK]
        orig.append(tma_features(original))
        reset.append(tma_features(window))
        conventional.append(temporal(window))
    block_means=[wave[i*BLOCK:(i+1)*BLOCK].mean(axis=0) for i in range(8)]
    ret=[]
    for i in range(1,8):
        x={'record':rec,'block':i,
           'wave_drift':float(np.sqrt(np.mean((block_means[i]-block_means[0])**2))),
           'wave_step':float(np.sqrt(np.mean((block_means[i]-block_means[i-1])**2))),
           'wave_texture':float(np.mean(np.sqrt(np.mean((wave[i*BLOCK:(i+1)*BLOCK]-block_means[i])**2,axis=1))))}
        for nam,data,keys in [('timing',conventional,TEMP_KEYS),('origin',orig,TMA_KEYS),('reset',reset,TMA_KEYS)]:
            for k in keys:
                x[f'{nam}_{k}']=data[i][k]
                x[f'{nam}_{k}_dprev']=data[i][k]-data[i-1][k]
                x[f'{nam}_{k}_dstart']=data[i][k]-data[0][k]
        ret.append(x)
    return ret

def cv(df,features,target):
    y=df[target].to_numpy(float);pred=np.full(len(y),np.nan)
    for name in df.record.unique():
        test=(df.record==name).to_numpy();train=~test
        m=make_pipeline(StandardScaler(),Ridge(alpha=30))
        m.fit(df.loc[train,features].to_numpy(float),y[train])
        pred[test]=m.predict(df.loc[test,features].to_numpy(float))
    return pred

def main():
    tic=time.monotonic()
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures={ex.submit(one,name):name for name in RECORDS}
        for i,f in enumerate(as_completed(futures),1):
            rows+=f.result()
            print(f'{i}/12 {futures[f]} finished',flush=True)
    df=pd.DataFrame(rows).sort_values(['record','block']).reset_index(drop=True)
    df.to_csv(OUT/'respiration_evolution_windows.csv',index=False)
    make=lambda nm,keys:[f'{nm}_{k}{s}' for k in keys for s in ('','_dprev','_dstart')]
    timing=make('timing',TEMP_KEYS)
    configs={'conventional':timing,'origin_TMA_only':make('origin',TMA_KEYS),
             'reset_TMA_only':make('reset',TMA_KEYS),
             'conventional_plus_origin':timing+make('origin',TMA_KEYS),
             'conventional_plus_reset':timing+make('reset',TMA_KEYS)}
    results={}
    predfile=df[['record','block']].copy()
    rng=np.random.default_rng(SEED)
    for tar in ('wave_drift','wave_step','wave_texture'):
        y=df[tar].to_numpy(float);predfile[tar]=y
        results[tar]={}
        for name,f in configs.items():
            pred=cv(df,f,tar)
            predfile[f'{tar}_{name}']=pred
            results[tar][name]={'MAE':float(mean_absolute_error(y,pred)),
                'R2':float(1-sum((y-pred)**2)/sum((y-y.mean())**2))}
        ids=df.record.unique()
        maps={x:np.flatnonzero(df.record.to_numpy()==x) for x in ids}
        for tag,m1,m2 in [('incremental_origin','conventional','conventional_plus_origin'),
                          ('incremental_reset','conventional','conventional_plus_reset')]:
            e1=np.abs(y-predfile[f'{tar}_{m1}'].to_numpy(float))
            e2=np.abs(y-predfile[f'{tar}_{m2}'].to_numpy(float))
            boot=[]
            for k in range(2999):
                ix=np.concatenate([maps[v] for v in rng.choice(ids,size=len(ids),replace=True)])
                boot.append(float(np.mean(e1[ix]-e2[ix])))
            results[tar][tag]={'MAE_improvement':float(e1.mean()-e2.mean()),
                                'bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist()}
    predfile.to_csv(OUT/'heldout_predictions.csv',index=False)
    result={'dataset':'PhysioNet Fantasia, peak-to-peak respiration as tactus and ECG R events as rhythm',
            'n_subjects':len(df.record.unique()),'n_windows':len(df),
            'engine_equivalence':'8 exact comparisons: 2 subjects x 4 prefixes; plus prefix invariance',
            'results':results,'seconds':time.monotonic()-tic,
            'limitations':['12 prespecified recordings; first 64 valid breaths; first 1200 s downloaded',
                           'Heartbeats constitute rhythm within respiratory reference; does not measure breathing-alone rhythm',
                           'Respiration waveforms define beat anchors and independent morphology criterion',
                           'Small sample, exploratory; not full EEG recordings']}
    (OUT/'cardioresp_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT '+json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
