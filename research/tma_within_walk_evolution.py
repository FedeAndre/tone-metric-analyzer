#!/usr/bin/env python3
"""Test whether origin-anchored full TMA morphology describes *within-walk*
evolution of independent raw bilateral force-shape measures better than matched
timing and phase-history features. No PD/HC labels used in modelling.

All features use the prefix observed up to the end of each eight-cycle block.
Subjects, never cycles, are held out. Frozen TMA computation is imported, not modified.
"""
import json
import time
import math
from fractions import Fraction
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tma_gait_gauge_validation as g
from tma_gait_fast_exact import fast_level_sets, assert_equivalent
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile

OUT=Path('research/tma_temporal_evolution_results')
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261008
BLOCK=8
POINTS=64
N_CYCLES=64

TMA_KEYS=['Dmean','Dstd','Hmean','Hstd','lambda_mean','multilevel',
          'Hchange','level_turnover','pivot_rate','pivot_depth',
          'pivot_compound','tree_span','tree_cross_block']
TIMING_KEYS=[f'{item}_{suffix}' for item in ['stride','RTO','RHS','LTO']
              for suffix in ['mean','std','madiff','acf1','acf2']]+[
    'phase_hist_8_entropy','phase_hist_16_entropy','phase_bigrams'
]

def finite(x,default=0.):
    x=float(x)
    return x if np.isfinite(x) else default

def summarize8(cycles):
    a=np.array([[c['duration_s']]+[float(c['events'][j]['raw_phase']) for j in range(1,4)]
                for c in cycles],float)
    vals={}
    for j,name in enumerate(('stride','RTO','RHS','LTO')):
        x=a[:,j]
        vals[name+'_mean']=float(np.mean(x))
        vals[name+'_std']=float(np.std(x,ddof=1))
        vals[name+'_madiff']=float(np.mean(np.abs(np.diff(x))))
        for lag in (1,2):
            vals[f'{name}_acf{lag}']=finite(np.corrcoef(x[:-lag],x[lag:])[0,1]) if np.std(x)>1e-8 and np.std(x[:-lag])>1e-8 and np.std(x[lag:])>1e-8 else 0.
    phases=np.ravel(a[:,1:]/2.)
    for nb in (8,16):
        h=np.bincount(np.minimum(nb-1,np.floor(phases*nb).astype(int)),minlength=nb)
        p=h[h>0]/h.sum()
        vals[f'phase_hist_{nb}_entropy']=float(-sum(p*np.log(p))/np.log(nb))
    # one conventional serial-order descriptor: frequency of positive/negative
    # changes in right foot contact phase.
    dx=np.diff(a[:,2]);vals['phase_bigrams']=float(np.mean(np.sign(dx[1:])==np.sign(dx[:-1]))) if len(dx)>1 else 0.
    return vals

def extract_force_profiles(arr,cy):
    left=np.maximum(arr[:,17],0)
    right=np.maximum(arr[:,18],0)
    first=cy[:BLOCK]
    reference=np.concatenate([left[c['start_sample']:c['end_sample']]+right[c['start_sample']:c['end_sample']] for c in first])
    scale=float(np.median(reference[reference>20])) if (reference>20).any() else float(np.mean(reference))
    scale=max(scale,1)
    profiles=[]
    for c in cy:
        a=c['start_sample'];b=c['end_sample']
        idx=np.linspace(a,b-1,POINTS)
        profiles.append(np.stack([np.interp(idx,np.arange(a,b),left[a:b]),
                                  np.interp(idx,np.arange(a,b),right[a:b])],axis=0)/scale)
    return np.asarray(profiles)

def feature_tma(cycles,window=BLOCK):
    rows,levels=fast_level_sets(cycles)
    wave=[]
    seen=set()
    for t,ci,phase,label in rows:
        if t in seen:continue
        seen.add(t)
        ll=sorted(int(v) for v in levels[t])
        assert ll
        wave.append({
            'segment_index':0,'measure_index':ci,'measure_number':str(ci+1),
            'onset_quarter':str(t),'offset_in_measure_quarter':str(phase),
            'levels':ll,'height':max(ll),'density':len(ll),
            'lowest_level':min(ll),'attack':True,'parenthetical':False})
    pivots=build_pivot_profile(wave)
    trees=build_tree_profile(wave)
    start=len(cycles)-window
    w=[p for p in wave if p['measure_index']>=start]
    H=np.array([p['height'] for p in w],float)
    D=np.array([p['density'] for p in w],float)
    L=np.array([p['lowest_level'] for p in w],float)
    tp=[p for p in pivots if p['pivot_measure_index']>=start]
    br=[b for b in trees['branches'] if b['target_measure_index']>=start]
    turnovers=[]
    for x,y in zip(w[:-1],w[1:]):
        a,b=set(x['levels']),set(y['levels'])
        turnovers.append(1-len(a&b)/len(a|b))
    return {
        'Dmean':float(D.mean()),'Dstd':float(D.std(ddof=1)),
        'Hmean':float(H.mean()),'Hstd':float(H.std(ddof=1)),
        'lambda_mean':float(L.mean()),'multilevel':float(np.mean(D>1)),
        'Hchange':float(np.mean(np.abs(np.diff(H)))),
        'level_turnover':float(np.mean(turnovers)),'pivot_rate':len(tp)/len(w),
        'pivot_depth':float(np.mean([p['drop_depth'] for p in tp])) if tp else 0.,
        'pivot_compound':float(np.mean([p['compound'] for p in tp])) if tp else 0.,
        'tree_span':float(np.mean([float(Fraction(b['distance_quarter'])) for b in br])) if br else 0.,
        'tree_cross_block':float(np.mean([b['source_measure_index']<start for b in br])) if br else 0.,
    }

def analyze_record(record,group,i):
    arr=g.download_record(record)
    cycles,fs,_=g.extract_cycles(arr,record)
    raw=extract_force_profiles(arr,cycles)
    windows=[raw[j*BLOCK:(j+1)*BLOCK] for j in range(8)]
    avgs=[w.mean(axis=0) for w in windows]
    reference=avgs[0]
    # waveform outcomes measure changes in both feet's FORCE profiles
    outcomes=[]
    for j in range(8):
        step=np.sqrt(np.mean((avgs[j]-avgs[j-1])**2)) if j>0 else 0.
        drift=np.sqrt(np.mean((avgs[j]-reference)**2))
        texture=float(np.mean(np.sqrt(np.mean((windows[j]-avgs[j])**2,axis=(1,2)))))
        outcomes.append({'step':float(step),'drift':float(drift),'texture':texture})
    orig=[]
    reset=[]
    conventional=[]
    # Genuine online prefixes; later gait never informs prefix feature calculation.
    for j in range(8):
        sub=cycles[j*BLOCK:(j+1)*BLOCK]
        orig.append(feature_tma(cycles[:(j+1)*BLOCK],BLOCK))
        reset.append(feature_tma(sub,BLOCK))
        conventional.append(summarize8(sub))
    ret=[]
    for j in range(1,8):
        row={'record':record,'group':group,'block':j,
             **{f'force_{n}':v for n,v in outcomes[j].items()}}
        for name,results,keys in [('timing',conventional,TIMING_KEYS),('tma_origin',orig,TMA_KEYS),('tma_reset',reset,TMA_KEYS)]:
            for f in keys:
                row[f'{name}_{f}']=results[j][f]
                row[f'{name}_{f}_delta_prev']=results[j][f]-results[j-1][f]
                row[f'{name}_{f}_delta_origin']=results[j][f]-results[0][f]
        ret.append(row)
    return ret

def lopo_predictions(df,features,outcome,alpha=30):
    y=df[outcome].to_numpy(float)
    pred=np.full(len(df),np.nan)
    for rid in df.record.unique():
        test=(df.record==rid).to_numpy()
        train=~test
        Xtr=df.loc[train,features].to_numpy(float)
        Xte=df.loc[test,features].to_numpy(float)
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(Xtr,y[train]);pred[test]=model.predict(Xte)
    return pred

def report_predictions(df,outcomes,config):
    summaries={}
    saved=pd.DataFrame({'record':df.record,'block':df.block})
    for target in outcomes:
        y=df[target].to_numpy(float)
        summaries[target]={}
        for name,features in config.items():
            pred=lopo_predictions(df,features,target)
            summaries[target][name]={
                'MAE':float(mean_absolute_error(y,pred)),
                'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
                'R2':float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)),
                'Spearman_r':float(stats.spearmanr(y,pred).statistic)}
            saved[f'{target}_{name}']=pred
        saved[target]=y
    saved.to_csv(OUT/'heldout_subject_predictions.csv',index=False)
    return summaries,saved

def bootstrap_gain(df,pred,target,name1,name2,B=4999):
    y=pred[target].to_numpy(float)
    a=np.abs(y-pred[f'{target}_{name1}'].to_numpy(float))
    b=np.abs(y-pred[f'{target}_{name2}'].to_numpy(float))
    ids=df.record.unique()
    blocks={name:np.flatnonzero(df.record.to_numpy()==name) for name in ids}
    rng=np.random.default_rng(SEED+10)
    differences=[]
    for _ in range(B):
        choice=rng.choice(ids,size=len(ids),replace=True)
        ix=np.concatenate([blocks[x] for x in choice])
        differences.append(float(np.mean(a[ix])-np.mean(b[ix])))
    return {'MAE_gain_positive_if_second_better':float(a.mean()-b.mean()),
            'subject_bootstrap_95CI':np.quantile(differences,[.025,.975]).tolist(),
            'P_bootstrap_nonpositive':float(np.mean(np.array(differences)<=0))}

def main():
    tic=time.monotonic()
    # Independent full-engine versus specialized implementation at multiple origins.
    for rec in (g.CONTROL_IDS[0],g.PD_IDS[0]):
        arr=g.download_record(rec)
        cy,*_=g.extract_cycles(arr,rec)
        for k in (8,16,32,64):
            assert_equivalent(cy[:k],f'{rec} first{k}')
        # Prove true prefix invariance before using a completed sequence.
        _,full=fast_level_sets(cy)
        for k in (8,16,32,48):
            oldrows,old=fast_level_sets(cy[:k])
            assert all(old[t]==full[t] for t,_,_,_ in oldrows),f'Noncausal prefix: {rec}-{k}'
    print('PASS: eight engine-equivalence plus eight prefix-causality gates',flush=True)
    records=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(analyze_record,r,group,i):r for i,(r,group) in enumerate(g.RECORDS)}
        for j,f in enumerate(as_completed(fut),1):
            records+=f.result()
            print(f'{j}/47 {fut[f]}',flush=True)
    df=pd.DataFrame(records).sort_values(['record','block']).reset_index(drop=True)
    assert len(df)==47*7
    df.to_csv(OUT/'evolution_windows.csv',index=False)
    pick=lambda prefix,changes:[c for c in df.columns if c.startswith(prefix) and c.endswith(changes)]
    def fset(prefix,types):
        return [f'{prefix}_{k}{suffix}' for k in (TIMING_KEYS if prefix=='timing' else TMA_KEYS) for suffix in types]
    config={
        'intercept':[],
        'conventional_timing':fset('timing',['','_delta_prev','_delta_origin']),
        'origin_TMA_only':fset('tma_origin',['','_delta_prev','_delta_origin']),
        'reset_TMA_only':fset('tma_reset',['','_delta_prev','_delta_origin']),
        'timing_plus_origin_TMA':fset('timing',['','_delta_prev','_delta_origin'])+fset('tma_origin',['','_delta_prev','_delta_origin']),
        'timing_plus_reset_TMA':fset('timing',['','_delta_prev','_delta_origin'])+fset('tma_reset',['','_delta_prev','_delta_origin']),
    }
    # Intercept-only is training-subject mean; no subject/test observations used.
    config.pop('intercept')
    summaries,pred=report_predictions(df,['force_step','force_drift','force_texture'],config)
    gains={}
    for target in ['force_step','force_drift','force_texture']:
        gains[target]={
            'origin_incremental':bootstrap_gain(df,pred,target,'conventional_timing','timing_plus_origin_TMA'),
            'reset_incremental':bootstrap_gain(df,pred,target,'conventional_timing','timing_plus_reset_TMA'),
            'origin_minus_reset':bootstrap_gain(df,pred,target,'timing_plus_reset_TMA','timing_plus_origin_TMA'),
        }
    result={
        'dataset':'PhysioNet gaitpdb 47 recordings, 64 cycles each','subject_count':47,'windows':len(df),
        'method':'8-cycle online prefixes; full-registered TMA wave/pivots/tree; true subject-held-out Ridge(alpha=30)',
        'targets':'Independent force waveform evolution (mean bilateral normalized force shapes)',
        'models':summaries,'paired_subject_bootstrap':gains,
        'time_seconds':time.monotonic()-tic,
        'limits':['Retrospective sample; outcomes are force signals from which step events are thresholded',
                  'Limited 64 cycles and 8-cycle windows; overlapping hypotheses exploratory',
                  'Group labels unused','Baseline includes phase autocorrelations and dyadic phase entropies',
                  'Bilateral force profile is physiological proxy, not clinical ground truth']}
    (OUT/'gait_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT '+json.dumps({'models':summaries,'paired_bootstrap':gains},indent=2),flush=True)

if __name__=='__main__':main()
