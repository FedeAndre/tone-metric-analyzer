#!/usr/bin/env python3
"""Direct measurement economy and nonredundancy among exact homometric rhythms.

Pre-established population:
9,828 fixed-4/4 sixteenth-grid motifs with 4-7 variable attacks.
Restrict to distinct non-rotation, non-reversal patterns with same whole-bar
onset autocorrelation and same circular IOI histogram. Strong subset: same
isolated-event TMA density power. Compare online one-number and finite-vector
descriptors, full complex onset phase, full TMA wave, root-shift baseline.

A TMA difference that ordinary *phase-aware* signal already exposes
is not unique source information. Do not infer clinical utility.
"""
from fractions import Fraction
from pathlib import Path
from collections import defaultdict
import itertools,time,json
import numpy as np
import tma_variable_drums as tm
from tma_controlled_spectra import cyc_acf,intervals,make_events,components,tv,power,centered_power_key,N,N_BARS,BAR,GRID
from tma_nondihedral_controls import canonical,spectral_keys
OUT=Path('research/tma_online_measurement_results')
OUT.mkdir(parents=True,exist_ok=True)
EPS=1e-11
def extract_sequence(motif,preexisting):
    t=make_events(motif)
    L=tm.levels_full(t,N_BARS)
    S=[preexisting[int(float(q)*GRID)] for q in t]
    H=np.array([max(L[q]) for q in t],float)
    D=np.array([len(L[q]) for q in t],float)
    SH=np.array([max(x) for x in S],float)
    SD=np.array([len(x) for x in S],float)
    output={}
    # Smallest useful scalar TMA measurement versus ordinary statistics:
    output['H_mean']=float(H.mean())
    output['D_mean']=float(D.mean())
    output['H_sd']=float(H.std())
    output['D_sd']=float(D.std())
    output['context_D_mean']=float((D-SD).mean())
    output['context_H_mean']=float((H-SH).mean())
    output['context_D_sd']=float((D-SD).std())
    output['context_H_sd']=float((H-SH).std())
    # Distribution over Level values as a non-Fourier origin-anchored state
    output['D_hist']=tuple(int(x) for x in np.bincount(D.astype(int),minlength=30))
    output['H_hist']=tuple(int(x) for x in np.bincount(H.astype(int),minlength=30))
    output['context_D_hist']=tuple(int(x) for x in np.bincount((D-SD).astype(int)+30,minlength=70))
    # Phase aware conventional beat/grid and canonical integer gaps
    mask=np.zeros(BAR,int);mask[list(motif)]=1
    output['phase_vector']=tuple(int(x) for x in mask)
    output['ioi_ordered']=tuple((motif[(i+1)%len(motif)] +(BAR if i==len(motif)-1 else 0))-motif[i] for i in range(len(motif)))
    output['ioi_hist']=intervals(motif)
    output['onset_acf']=cyc_acf(motif)
    # Small scalar conventional summaries; intentionally weaker, not claimed general baseline
    IOI=np.array(output['ioi_ordered'])
    output['ioi_mean']=float(IOI.mean())
    output['ioi_std']=float(IOI.std())
    output['ioi_madiff']=float(np.mean(abs(np.diff(IOI))))
    output['beat_phase_R']=float(abs(np.mean(np.exp(2j*np.pi*(np.array(motif)%4)/4))))
    return output
def enumerate_strong(single):
    sh=np.array([max(x) for x in single],int)
    sd=np.array([len(x) for x in single],int)
    group=defaultdict(list)
    for k in range(4,8):
        for subset in itertools.combinations(range(1,BAR),k-1):
            motif=(0,*subset)
            group[(k,cyc_acf(motif),intervals(motif))].append(motif)
    matching=[]
    other=0
    for motifs in group.values():
        if len(motifs)<2:continue
        signatures={p:spectral_keys(p,sh,sd) for p in motifs}
        keys={p:canonical(p) for p in motifs}
        for i,p in enumerate(motifs):
            for q in motifs[i+1:]:
                if keys[p]==keys[q]:continue
                other+=1
                if signatures[p][1]==signatures[q][1]:
                    matching.append((p,q))
    assert other==1696 and len(matching)==40,(other,len(matching))
    return matching
def check(pair,single):
    a,b=pair
    ca=extract_sequence(a,single);cb=extract_sequence(b,single)
    ordinary=['phase_vector','ioi_ordered','ioi_hist','onset_acf','ioi_mean','ioi_std','ioi_madiff','beat_phase_R']
    tma_features=['H_mean','D_mean','H_sd','D_sd','context_D_mean','context_H_mean','context_D_sd','context_H_sd','D_hist','H_hist','context_D_hist']
    diff={x:ca[x]!=cb[x] if not isinstance(ca[x],float) else abs(ca[x]-cb[x])>EPS for x in ordinary+tma_features}
    metric=components(a,b,single)
    return {'motifA':a,'motifB':b,
      'distinguished_by':diff,
      'ordinary_onset_power_equal':metric['raw']['exact_power_equal'],
      'static_density_power_equal':metric['sD']['exact_power_equal'],
      'full_density_power_distinct':not metric['D']['exact_power_equal'],
      'context_density_power_distinct':not metric['cD']['exact_power_equal'],
      'full_density_TV':metric['D']['power_tv_distance'],
      'context_density_TV':metric['cD']['power_tv_distance'],
      'mean_H_abs_delta':abs(ca['H_mean']-cb['H_mean']),
      'mean_D_abs_delta':abs(ca['D_mean']-cb['D_mean'])}
def onlinetest(pair,nbars=16):
    # first seven identical bars, then A remains A versus switches to B.
    a,b=pair
    switch=7
    times1=tuple(Fraction(4*k)+Fraction(j,4) for k in range(nbars) for j in a)
    times2=tuple(Fraction(4*k)+Fraction(j,4) for k in range(nbars) for j in (a if k<switch else b))
    assert len(times1)==len(times2)
    # compare two expected full event streams, cumulative at every completed bar
    L1=tm.levels_full(times1,nbars)
    L2=tm.levels_full(times2,nbars)
    rows=[]
    for step in range(1,nbars+1):
        cutoff=4*step
        x=[q for q in times1 if q<cutoff]
        y=[q for q in times2 if q<cutoff]
        out={'bars_observed':step,'event_count':len(x)}
        out['onset_binned_hist_same']=tuple(np.bincount([int(q*4)%16 for q in x],minlength=16))==tuple(np.bincount([int(q*4)%16 for q in y],minlength=16))
        out['onset_exact_stream_same']=x==y
        out['tma_H_mean_same']=abs(np.mean([max(L1[q]) for q in x])-np.mean([max(L2[q]) for q in y]))<EPS
        out['tma_D_mean_same']=abs(np.mean([len(L1[q]) for q in x])-np.mean([len(L2[q]) for q in y]))<EPS
        out['tma_level_density_hist_same']=tuple(np.bincount([len(L1[q]) for q in x],minlength=30))==tuple(np.bincount([len(L2[q]) for q in y],minlength=30))
        rows.append(out)
    return rows
def main():
    tick=time.perf_counter()
    singleton=[tm.levels_full([Fraction(j,GRID)],N_BARS)[Fraction(j,GRID)] for j in range(N)]
    strong=enumerate_strong(singleton)
    rec=[check(x,singleton) for x in strong]
    features=rec[0]['distinguished_by'].keys()
    rates={x:{'n_distinct':sum(int(q['distinguished_by'][x]) for q in rec),
               'fraction':sum(int(q['distinguished_by'][x]) for q in rec)/len(rec)}
           for x in features}
    spectral={x:sum(int(q[x]) for q in rec) for x in
       ['full_density_power_distinct','context_density_power_distinct']}
    # Event economy of representative controlled example
    anchor=min(rec,key=lambda x:(-x['context_density_TV'],x['motifA'],x['motifB']))
    times=onlinetest((anchor['motifA'],anchor['motifB']))
    tm.check_engine(make_events(anchor['motifA']),N_BARS)
    tm.check_engine(make_events(anchor['motifB']),N_BARS)
    output={'population':'all non-dihedral homometric 4–7 attack rhythms with equal cyclic IOI histogram, subset also equal static-TMA D power',
      'n_strongly_matched_pairs':len(strong),
      'n_all_nondihedral_matched_pairs':1696,
      'n_distinguishable_by_each_measure':rates,
      'spectral_count':spectral,
      'full_TMA_density_TV_quantiles':np.quantile([r['full_density_TV'] for r in rec],[0,.25,.5,.75,1]).tolist(),
      'context_TMA_density_TV_quantiles':np.quantile([r['context_density_TV'] for r in rec],[0,.25,.5,.75,1]).tolist(),
      'max_context_pair':anchor,
      'max_context_pair_online_change_from_bar8':times,
      'exact_engine_gates_passed':True,
      'seconds':time.perf_counter()-tick,
      'limitations':['The synthetic task is a measurement-discrimination test; no hidden biology or external meaning',
        'Exact phase vector and full complex Fourier phase are invertible alternatives and will discriminate every non-identical pair',
        'No claim that a small set of TMA features is an injective representation of original rhythmic events',
        'Strongly matched pairs selected by TMA static-density power (pre-committed), which benefits residual TMA comparison',
        'Per-bar onset-phase histogram measures an ordinary beat-locked temporal feature and is explicitly compared',
        'The onset stream is beat-anchored and contains variable within-bar attack rhythms, satisfying TMA input requirements']}
    OUT.joinpath('controlled_nonredundancy.json').write_text(json.dumps(output,indent=2))
    print('PASS all 40 strict pairs frozen engine checked')
    print('FINAL',json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
