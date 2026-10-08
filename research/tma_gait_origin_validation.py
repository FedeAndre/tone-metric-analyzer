#!/usr/bin/env python3
"""Fixed-origin TMA gait test against exact cycle-order and block-order nulls.
Uses the repository's validated specialist binary gait evaluator. Does not alter TMA.
"""
import json, math, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tma_gait_gauge_validation as g
from tma_gait_fast_exact import fast_level_sets, fast_analyze_cycles, assert_equivalent

SEED=20261008
B=119
CHECKPOINTS=(16,32,48,64)
WINDOW=16
OUT=Path("research/tma_gait_origin_results")
OUT.mkdir(parents=True, exist_ok=True)

def score_sequence(cycles):
    """Fixed-origin 16-cycle contributions. Prefix stability is independently gated."""
    rows, levelmap=fast_level_sets(cycles)
    traj=[]
    for end in CHECKPOINTS:
        tail=[levelmap[t] for t,ci,q,label in rows if end-WINDOW<=ci<end]
        assert len(tail)==WINDOW*4
        n=len(tail)
        traj.append({
            "end":end,
            "D":sum(len(lev) for lev in tail)/n,
            "H":sum(max(lev) for lev in tail)/n,
            "lambda":sum(min(lev) for lev in tail)/n,
            "multilevel":sum(len(lev)>1 for lev in tail)/n,
        })
    return traj

def raw_metrics(cycles):
    """Physically timed stride intervals plus conventional phase variability / serial dependence."""
    f=np.array([[c['duration_s']]+[float(e['raw_phase']) for e in c['events'][1:]] for c in cycles],dtype=float)
    means=f.mean(axis=0); sds=f.std(axis=0,ddof=1)
    dv=np.diff(f,axis=0)
    lagcorr=[]
    for j in range(4):
        if np.std(f[:,j])<1e-10: lagcorr.append(0.)
        else:lagcorr.append(float(np.corrcoef(f[:-1,j],f[1:,j])[0,1]))
    names=['duration','RTO','RHS','LTO']
    ret={}
    for j,n in enumerate(names):
        ret[n+'_mean']=float(means[j])
        ret[n+'_sd']=float(sds[j])
        ret[n+'_diff1']=float(np.mean(np.abs(dv[:,j])))
        ret[n+'_acf1']=float(lagcorr[j])
    return ret

def block_shuffle(cycles,rng,block=4):
    groups=[cycles[i:i+block] for i in range(0,len(cycles),block)]
    return [c for i in rng.permutation(len(groups)) for c in groups[int(i)]]

def perm_score(obs, vals):
    mu=float(np.mean(vals)); s=float(np.std(vals,ddof=1))
    z=(obs-mu)/s if s>1e-10 else 0.
    p=float((1+sum(abs(v-mu)>=abs(obs-mu)-1e-12 for v in vals))/(len(vals)+1))
    return mu,s,z,p

def one(record,group,ix):
    arr=g.download_record(record)
    cycles,fs,integ=g.extract_cycles(arr,record)
    rng=np.random.default_rng(SEED+ix*10091)
    orig=score_sequence(cycles)
    class_orig=raw_metrics(cycles)
    order=[]
    block=[]
    for mode in ('order','block'):
        bag=[]
        for rep in range(B):
            cy=g.cycle_order_scramble(cycles,rng) if mode=='order' else block_shuffle(cycles,rng,4)
            bag.append(score_sequence(cy))
        if mode=='order':order=bag
        else:block=bag
    result={'record':record,'group':group,**class_orig}
    for ki,end in enumerate(CHECKPOINTS):
        for feat in ('D','H','lambda','multilevel'):
            observed=orig[ki][feat]
            result[f'{feat}_{end}']=observed
            for mode,bag in [('order',order),('block',block)]:
                m,s,z,p=perm_score(observed,[x[ki][feat] for x in bag])
                for name,val in [('null_mean',m),('null_sd',s),('z',z),('perm_p',p)]:
                    result[f'{feat}_{end}_{mode}_{name}']=val
    # Origin-dependent per-record changes in 16-cycle-window morphology, null standardized
    for mode in ('order','block'):
        result[f'D_zchange_{mode}']=result[f'D_64_{mode}_z']-result[f'D_16_{mode}_z']
        result[f'D_zlinear_{mode}']=float(np.polyfit(CHECKPOINTS,[result[f'D_{t}_{mode}_z'] for t in CHECKPOINTS],1)[0])
    return result

def permutation_group_p(values,labels,B=9999):
    rng=np.random.default_rng(SEED+887)
    x=np.asarray(values,float); y=np.asarray(labels,int)
    a=x[y==0];b=x[y==1]
    diff=float(b.mean()-a.mean())
    hit=0
    for _ in range(B):
        p=rng.permutation(y)
        d=float(x[p==1].mean()-x[p==0].mean())
        hit+=abs(d)>=abs(diff)-1e-12
    return diff, (1+hit)/(B+1)

def loocv_auc(df, cols):
    X=df[cols].to_numpy(float)
    y=(df.group=='pd').astype(int).to_numpy()
    preds=[]
    for i in range(len(y)):
        train=np.arange(len(y))!=i
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
        clf.fit(X[train],y[train])
        preds.append(float(clf.predict_proba(X[i:i+1])[:,1][0]))
    preds=np.array(preds)
    return {'AUC':float(roc_auc_score(y,preds)),
            'balanced_accuracy_0p5':float(balanced_accuracy_score(y,preds>=0.5)),
            'n':len(y), 'predictions':preds.tolist()}

def main():
    clock=time.monotonic()
    jobs=[(r,gr,i) for i,(r,gr) in enumerate(g.RECORDS)]
    # Existing frozen engine is an independent reference. Confirm no silent override.
    for name in [g.CONTROL_IDS[0],g.PD_IDS[0]]:
        a=g.download_record(name)
        cy,*_=g.extract_cycles(a,name)
        for n in (16,32,64): assert_equivalent(cy[:n],f'{name}-n{n}')
        fullrows,fullmap=fast_level_sets(cy)
        for n in (16,32,48):
            partial,partialmap=fast_level_sets(cy[:n])
            assert all(tuple(sorted(fullmap[t]))==tuple(sorted(partialmap[t]))
                       for t,_,_,_ in partial),f'prefix stability failed: {name} n={n}'
    print("Frozen engine equivalence: six complete-prefix checks passed",flush=True)
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(one,*job):job for job in jobs}
        for k,fut in enumerate(as_completed(futures),1):
            rec=fut.result()
            records.append(rec)
            print(f'Analyzed {k}/47 {rec["record"]}',flush=True)
    df=pd.DataFrame(records).sort_values('record').reset_index(drop=True)
    df.to_csv(OUT/'participant_origin_features.csv',index=False)
    y=(df.group=='pd').astype(int).to_numpy()
    fixed_tests={}
    selected=['D_64','D_64_order_z','D_64_block_z','D_zchange_order','D_zchange_block',
              'D_16_order_z','D_32_order_z','D_48_order_z','H_64_order_z',
              'multilevel_64_order_z']
    for col in selected:
        diff,p=permutation_group_p(df[col].to_numpy(float),y)
        c=float(df.loc[df.group=='control',col].mean())
        d=float(df.loc[df.group=='pd',col].mean())
        fixed_tests[col]={'control_mean':c,'pd_mean':d,'PD_minus_control':diff,'group_permutation_p':p}
    # FDR within 10 planned tests and separately within central three tests
    from statsmodels.stats.multitest import multipletests
    pvals=[fixed_tests[c]['group_permutation_p'] for c in selected]
    adj=multipletests(pvals,method='fdr_bh')[1]
    for col,q in zip(selected,adj):fixed_tests[col]['q_BH_10']=float(q)
    # Count subjects whose order-aware Level density changes are unusual
    null_tests={}
    for end in CHECKPOINTS:
        for mode in ('order','block'):
            pvals=df[f'D_{end}_{mode}_perm_p'].to_numpy(float)
            null_tests[f'D_{end}_{mode}']={
                'n_p_le_0p05':int(np.sum(pvals<=.05)),
                'n_p_le_0p10':int(np.sum(pvals<=.1)),
                'median_p':float(np.median(pvals)),
                'control_mean_z':float(df.loc[df.group=='control',f'D_{end}_{mode}_z'].mean()),
                'pd_mean_z':float(df.loc[df.group=='pd',f'D_{end}_{mode}_z'].mean())
            }
    raw_cols=[n+'_'+type for n in ('duration','RTO','RHS','LTO') for type in ('mean','sd','diff1','acf1')]
    configurations={
        'tempo_and_phase_summaries':[c for c in raw_cols if c.endswith(('_mean','_sd'))],
        'conventional_with_serial_order':raw_cols,
        'TMA_uncorrected':['D_64','H_64','multilevel_64'],
        'TMA_order_corrected':['D_64_order_z','D_zchange_order'],
        'conventional_plus_TMA_order':raw_cols+['D_64_order_z','D_zchange_order'],
        'conventional_plus_raw_TMA':raw_cols+['D_64','H_64','multilevel_64'],
    }
    classification={k:loocv_auc(df,cols) for k,cols in configurations.items()}
    # bootstrap 95% CI for *paired* empirical AUC difference
    rng=np.random.default_rng(SEED+3459)
    pred0=np.array(classification['conventional_with_serial_order']['predictions'])
    pred1=np.array(classification['conventional_plus_TMA_order']['predictions'])
    deltas=[]
    for i in range(9999):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))<2:continue
        deltas.append(float(roc_auc_score(y[ix],pred1[ix])-roc_auc_score(y[ix],pred0[ix])))
    auc_gain={'observed':classification['conventional_plus_TMA_order']['AUC']-classification['conventional_with_serial_order']['AUC'],
              'bootstrap_CI_95':np.quantile(deltas,[.025,.975]).tolist(),
              'n_bootstrap':len(deltas)}
    for config in classification:classification[config].pop('predictions',None)
    out={'B_per_subject_per_null':B,'n_subjects':len(df),'n_pd':int(y.sum()),
         'n_controls':int(len(y)-y.sum()),'fixed_origin':True,
         'checkpoints':CHECKPOINTS,'group_comparisons':fixed_tests,
         'within_subject_nulls':null_tests,
         'classification_LOOCV':classification,
         'AUC_gain_paired_bootstrap':auc_gain,
         'runtime_seconds':time.monotonic()-clock,
         'limitations':['Temporal baseline grounded to each subject 64-cycle event multiset',
                        'Within-cycle beat events fixed by left heel strikes; cycle order permutations retain the durations and phase distribution',
                        'Block null retains 4-cycle local patterns, rearranges four-cycle blocks',
                        'No longitudinal clinical progression labels',
                        'Multiple analyses exploratory; fixed ten-feature FDR reported',
                        'External validation required']}
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'group_comparisons':fixed_tests,'classification_LOOCV':classification,
                      'AUC_gain':auc_gain,'runtime_seconds':out['runtime_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
