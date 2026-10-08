#!/usr/bin/env python3
"""TMA narrative test in REAL continuous gait, no clinical labels.
Controls exact cycle multiset by cycle shuffle and 4-cycle-block shuffle.
Same frozen algorithm and read-only pivot/tree definitions as the analyzer.
All TMA prefix features are computed from a fixed origin and confirmed causally.
"""
import json,time,math
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy import stats
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tma_gait_fast_exact import fast_level_sets,assert_equivalent
import tma_gait_gauge_validation as g

OUT=Path('research/tma_narrative_results')
OUT.mkdir(parents=True,exist_ok=True)
B=59
SEED=20261008
WINDOW=8
CHECKPOINTS=list(range(8,65,8))
TMACOLS=['H_w','D_w','lambda_w','H_sd','D_sd','H_jump','level_turn','multilevel',
         'pivot_confirmed','pivot_depth','tree_span','tree_long','tree_cross','meanD_history']
TIMCOLS=['stride_m','stride_sd','stride_jump','stride_acf1',
         'RTO_m','RTO_sd','RTO_jump','RTO_acf1',
         'RHS_m','RHS_sd','RHS_jump','RHS_acf1',
         'LTO_m','LTO_sd','LTO_jump','LTO_acf1',
         'phase_m_history','phase_sd_history','stride_m_history','stride_sd_history']

def block_shuffle(cy,rng,bs=4):
    block=[cy[i:i+bs] for i in range(0,len(cy),bs)]
    return [v for j in rng.permutation(len(block)) for v in block[int(j)]]

def tma_signature(cycles):
    rows,levmap=fast_level_sets(cycles)
    wave=[]
    for onset,ci,phase,label in rows:
        ll=sorted(levmap[onset])
        wave.append({'segment_index':0,'measure_index':ci,'measure_number':str(ci+1),
         'onset_quarter':str(onset),'offset_in_measure_quarter':str(phase),
         'levels':ll,'height':max(ll),'density':len(ll),'lowest_level':min(ll),
         'attack':True,'parenthetical':False})
    piv=build_pivot_profile(wave)
    branches=build_tree_profile(wave)['branches']
    heights=np.array([a['height'] for a in wave],float)
    density=np.array([a['density'] for a in wave],float)
    lowest=np.array([a['lowest_level'] for a in wave],float)
    turnover=np.array([0.]+[1-len(set(x['levels'])&set(y['levels']))/len(set(x['levels'])|set(y['levels'])) for x,y in zip(wave[:-1],wave[1:])])
    profile=[]
    for end in CHECKPOINTS:
        k=4*end;start=end-WINDOW;lo=4*start
        hm=heights[lo:k];dm=density[lo:k];lm=lowest[lo:k]
        # Only a pivot with a recovery before this window's end is confirmed.
        current=[p for p in piv if
            start<=int(math.floor(float(Fraction(str(p['recovery_onset_quarter']))/2)))<end]
        br=[b for b in branches if start<=int(b['target_measure_index'])<end]
        dist=np.array([int(b['target_node_index'])-int(b['source_node_index']) for b in br],float)
        feats=[
          float(hm.mean()),float(dm.mean()),float(lm.mean()),
          float(np.std(hm,ddof=1)),float(np.std(dm,ddof=1)),
          float(np.mean(abs(np.diff(hm)))) if len(hm)>1 else 0.,
          float(np.mean(turnover[lo:k])),
          float(np.mean(dm>1)),
          float(len(current)/len(hm)),
          float(np.mean([p['drop_depth'] for p in current])) if current else 0.,
          float(np.mean(dist)) if len(dist) else 0.,
          float(np.mean(dist>4)) if len(dist) else 0.,
          float(np.mean([b['source_measure_index']<start for b in br])) if br else 0.,
          float(np.mean(density[:k]))
        ]
        profile.append(feats)
    return np.array(profile)

def timing_signature(cycles):
    X=np.array([[c['duration_s']]+[float(e['raw_phase']) for e in c['events'][1:]]
                for c in cycles],float)
    rows=[]
    for end in CHECKPOINTS:
        sub=X[end-WINDOW:end];prefix=X[:end]
        f=[]
        for j in range(4):
            z=sub[:,j]
            f.extend([float(np.mean(z)),float(np.std(z,ddof=1)),
             float(np.mean(np.abs(np.diff(z)))),
             float(np.corrcoef(z[:-1],z[1:])[0,1]) if
               np.std(z[:-1])>1e-9 and np.std(z[1:])>1e-9 else 0.])
        f.extend([float(np.mean(prefix[:,1:])),float(np.std(prefix[:,1:])),
                  float(np.mean(prefix[:,0])),float(np.std(prefix[:,0]))])
        rows.append(f)
    return np.array(rows)

def force_signature(data,cycles):
    v=np.maximum(data[:,17:19],0)
    refs=[]
    for c in cycles:
        a,b=c['start_sample'],c['end_sample']
        xi=np.linspace(a,b-1,64)
        refs.append(np.stack([np.interp(xi,np.arange(a,b),v[a:b,0]),
                              np.interp(xi,np.arange(a,b),v[a:b,1])]))
    refs=np.array(refs)
    scale=np.median(refs[:8][refs[:8]>20])
    refs/=max(float(scale),1.)
    avgs=np.array([refs[i:i+8].mean(axis=0) for i in range(0,64,8)])
    return np.array([0.]+[float(np.sqrt(np.mean((avgs[j]-avgs[j-1])**2))) for j in range(1,8)])

def score(actual,null):
    mean=null.mean(axis=0)
    sd=null.std(axis=0,ddof=1)
    valid=np.mean(sd,axis=0)>1e-5
    # Per-time null normalization; completely deterministic grid features cannot
    # contribute evidence of event-order differences.
    sd=np.maximum(sd,1e-5)
    z=np.clip((actual[:,valid]-mean[:,valid])/sd[:,valid],-20,20)
    zs=np.clip((null[:,:,valid]-mean[:,valid])/sd[:,valid],-20,20)
    # Global temporal-narrative deviation, with every time and selected
    # morphological component retaining exactly the same origin.
    observed=float(np.mean(z**2))
    nullstat=np.mean(zs**2,axis=(1,2))
    p=float((1+np.sum(nullstat>=observed))/(len(nullstat)+1))
    # How much the narrative *changes*, after taking out the expected
    # deterministic unfolding at each time coordinate.
    diffs=np.diff(z,axis=0)
    ndiffs=np.diff(zs,axis=1)
    observed_turn=float(np.mean(diffs**2))
    nulturn=np.mean(ndiffs**2,axis=(1,2))
    pt=float((1+np.sum(nulturn>=observed_turn))/(len(nulturn)+1))
    # Window-specific anomaly to compare with independently measured force shape.
    anomaly=np.sqrt(np.mean(z**2,axis=1))
    return {'observed':observed,'null_mean':float(nullstat.mean()),'p':p,
        'turn_observed':observed_turn,'turn_null_mean':float(nulturn.mean()),
        'p_turn':pt,'anomaly':anomaly.tolist(),'rank':float(np.mean(nullstat<=observed))}

def one(item):
    rec,group,index=item
    data=g.download_record(rec)
    cycles,fs,check=g.extract_cycles(data,rec)
    rng=np.random.default_rng(SEED+index*99173)
    obs1=tma_signature(cycles)
    obs2=timing_signature(cycles)
    n1=[];n2=[];n3=[];n4=[]
    for i in range(B):
        a=g.cycle_order_scramble(cycles,rng)
        b=block_shuffle(cycles,rng,4)
        n1.append(tma_signature(a));n2.append(timing_signature(a))
        n3.append(tma_signature(b));n4.append(timing_signature(b))
    ret={'record':rec,
         'force_step':force_signature(data,cycles).tolist(),
         'TMA_order':score(obs1,np.array(n1)),
         'timing_order':score(obs2,np.array(n2)),
         'TMA_block':score(obs1,np.array(n3)),
         'timing_block':score(obs2,np.array(n4))}
    return ret

def across_subject(records,k,metric,Bglobal=9999):
    obs=np.mean([o[k][metric] for o in records])
    chance=float(np.mean([o[k]['null_mean'] if metric=='observed' else o[k]['turn_null_mean'] for o in records]))
    # Outcome is independent subject-specific p-value from a full reassignment.
    # Under null, percentile rank should be uniform.
    ps=np.array([o[k]['p'] if metric=='observed' else o[k]['p_turn'] for o in records])
    rng=np.random.default_rng(SEED+999)
    Bn=99999
    null=rng.random((Bn,len(ps))).mean(axis=1)
    actual=float(np.mean(1-ps))
    p=float((1+np.sum(null>=actual))/(Bn+1))
    return {'observed_mean':float(obs),'expected_mean':chance,
            'n_p_lt_0p05':int(sum(ps<=.05)),'n_p_lt_0p10':int(sum(ps<=.10)),
            'mean_one_minus_p':actual,'global_rank_p':p}

def main():
    start=time.monotonic()
    for rec in (g.CONTROL_IDS[0],g.PD_IDS[0]):
        arr=g.download_record(rec)
        cy,*_=g.extract_cycles(arr,rec)
        for n in (8,16,32,64):
            assert_equivalent(cy[:n],f'{rec}-{n}')
        full,gl=fast_level_sets(cy)
        for n in (8,16,32,48):
            sub,sl=fast_level_sets(cy[:n])
            assert all(sl[t]==gl[t] for t,i,q,label in sub)
    print('PASS engine and origin-causality checks',flush=True)
    dataset=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(one,(rec,group,i)):rec for i,(rec,group) in enumerate(g.RECORDS)}
        for num,f in enumerate(as_completed(futures),1):
            item=f.result();dataset.append(item)
            print(f'Analyzed {num}/47 {item["record"]}',flush=True)
    dataset.sort(key=lambda x:x['record'])
    results={'n_subjects':len(dataset),'n_cycles':64,'B_per_mode_per_subject':B,
        'methods':{'TMA':TMACOLS,'timing':TIMCOLS},
        'definition':'Origin anchored chronological wave, Levels, confirmed pivots, closest-predecessor trees at eight successive checkpoints, compared with within-subject timing-multiset-preserving surrogates',
        'per_record':dataset,
        'global':{k:{x:across_subject(dataset,k,x) for x in ('observed','turn_observed')}
                  for k in ('TMA_order','timing_order','TMA_block','timing_block')},
        'runtime_s':time.monotonic()-start}
    (OUT/'real_gait_narrative_results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print('GLOBAL '+json.dumps(results['global'],indent=2),flush=True)
if __name__=='__main__':main()
