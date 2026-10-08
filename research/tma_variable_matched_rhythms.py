#!/usr/bin/env python3
"""Controlled *variable* rhythms: exact aperiodic homometric seed motifs
under irregular shared bar gating.

Every retained variant has fixed 4/4 metronomic beat, variable attacks within
active bars, irregular active/silent-bar allocation across 16 bars.
Aperiodic seed autocorrelation identity guarantees equality of entire
length-256 onset FFT magnitudes after convolving with same nonoverlapping gate.
Match circular motif IOI histograms, and explicitly exclude all rotations
and reversals. Optimize no clinical outcome: select minimum isolated-event
TMA density+height Fourier TV before measuring full TMA.
"""
import collections,itertools,json,time
from pathlib import Path
from fractions import Fraction
import numpy as np
import tma_variable_drums as t
from tma_controlled_spectra import power,tv
from tma_nondihedral_controls import canonical
OUT=Path('research/tma_controlled_spectra_results')
NB=16
GRID=4
BAR=16
N=NB*BAR
GATE=(0,1,3,4,5,7,8,10,11,12,14,15)
def diffs(p):
    return tuple(sorted(p[j]-p[i] for i in range(len(p)) for j in range(i+1,len(p))))
def ioihist(p):
    return tuple(sorted((p[(i+1)%len(p)] + (BAR if i==len(p)-1 else 0))-p[i] for i in range(len(p))))
def marks(p):
    return tuple(sorted(Fraction(4*b)+Fraction(k,4) for b in GATE for k in p))
def aperiodic_power(p):
    q=np.zeros(BAR,int);q[list(p)]=1
    # linear autocorrelation invariant
    v=np.correlate(q,q,mode='full')
    return tuple(int(z) for z in v)
def static_signatures(event,single):
    x=np.zeros(N);h=np.zeros(N);d=np.zeros(N)
    for tm in event:
        i=int(4*tm)
        x[i]=1
        s=single[i]
        h[i]=max(s);d[i]=len(s)
    return {'raw':power(x),'staticH':power(h),'staticD':power(d)}
def evaluate(event,single):
    lv=t.levels_full(event,NB)
    signals={k:np.zeros(N) for k in ('fullH','fullD','contextH','contextD')}
    changed=0
    for tm in event:
        i=int(4*tm)
        st=lv[tm];sg=single[i]
        if st!=sg:changed+=1
        signals['fullH'][i]=max(st)
        signals['fullD'][i]=len(st)
        signals['contextH'][i]=max(st)-max(sg)
        signals['contextD'][i]=len(st)-len(sg)
    return {k:power(v) for k,v in signals.items()},changed/len(event)
def main():
    tic=time.monotonic()
    patterns=collections.defaultdict(list)
    count=0
    for k in (6,7,8):
        for tail in itertools.combinations(range(1,BAR),k-1):
            p=(0,*tail)
            patterns[(k,diffs(p),ioihist(p))].append(p)
            count+=1
    pairs=[]
    for group in patterns.values():
        if len(group)<2:continue
        for i,p in enumerate(group):
            for q in group[i+1:]:
                if canonical(p)==canonical(q):continue
                assert aperiodic_power(p)==aperiodic_power(q)
                pairs.append((p,q))
    # independent exemplar secondary: all patterns dense enough to be rhythmic
    assert pairs
    singleton=[t.levels_full([Fraction(j,GRID)],NB)[Fraction(j,GRID)] for j in range(N)]
    candidates=[]
    for p,q in pairs:
        A=marks(p);B=marks(q)
        a=static_signatures(A,singleton);b=static_signatures(B,singleton)
        assert np.allclose(a['raw'],b['raw'],rtol=1e-10,atol=1e-7)
        d0=tv(a['staticD'],b['staticD'])
        h0=tv(a['staticH'],b['staticH'])
        candidates.append({'A':p,'B':q,'n_events':len(A),'staticH_TV':h0,
         'staticD_TV':d0,'static_TV_mean':.5*(d0+h0)})
    candidates.sort(key=lambda x:(x['staticD_TV'],x['static_TV_mean']))
    chosen=candidates[0]
    A=marks(chosen['A']);B=marks(chosen['B'])
    ta,ca=evaluate(A,singleton);tb,cb=evaluate(B,singleton)
    t.check_engine(A,NB);t.check_engine(B,NB)
    met={}
    for k in ('fullH','fullD','contextH','contextD'):
        met[k]={'power_TV':tv(ta[k],tb[k]),
          'exactly_equal':bool(np.allclose(ta[k],tb[k],atol=1e-7,rtol=1e-10))}
    output={'tested_patterns':count,'eligible_aperoidic_homometric_non_dihedral_pairs':len(pairs),
      'beat':'4/4 quarter note; 16-step each bar', 'irregular_gate_bar_indices':GATE,
      'n_active_bars':len(GATE),'n_silent_bars':NB-len(GATE),
      'selected_by_best_static_control_before_TMA_outcome':chosen,
      'actual_contextual_metrics':met,
      'fraction_contextual_stack':{'A':ca,'B':cb},
      'conventional_psd_equal':True,
      'linear_noncyclic_onset_autocorrelation_seed':aperiodic_power(chosen['A']),
      'comparisons':[{'A':q['A'],'B':q['B'],'sH':q['staticH_TV'],'sD':q['staticD_TV']} for q in candidates[:20]],
      'runtime_s':time.monotonic()-tic,
      'limitations':['No physiological response available; mathematical stimulus construction only',
         'Irregular bar gating creates active/silent breaks while constant metronome persists',
         'Within-beat phase histograms of selected motifs need not be equal',
         'Static singleton H/D spectral matching is not guaranteed even when onset PSD matches',
         'All Fourier complex phases and complete interval order may distinguish the motifs']}
    OUT.joinpath('variable_bar_results.json').write_text(json.dumps(output,indent=2))
    print('PASS exact aperiodic onset PSD and frozen TMA engine for irregular variable rhythms')
    print('RESULT',json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
