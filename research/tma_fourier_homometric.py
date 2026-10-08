#!/usr/bin/env python3
"""Exact homometric Fourier counterexample on 4/4 dyadic beat grid.

Two 16-step rhythmic onset patterns have identical cyclic autocorrelation
and onset Fourier power, while frozen TMA H/D wave power may differ.
Distinguishes different summary representations, *not new Shannon info*.
"""
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import tma_variable_drums as g

N_BARS=32
STEP_PER_QUARTER=4
PER_BAR=16
A=(0,1,4,9)
B=(0,1,5,8)
OUT=Path('research/tma_fourier_results/homometric_test.json')

def cyclic_counts(pattern):
    p=set(pattern)
    return [sum(((u+d)%PER_BAR) in p for u in pattern) for d in range(PER_BAR)]

def build(pattern):
    times=sorted(Fraction(4*bar)+Fraction(k,STEP_PER_QUARTER) for bar in range(N_BARS) for k in pattern)
    g.check_engine(times,16)
    g.check_engine(times,32)
    l=g.levels_full(times,N_BARS)
    N=N_BARS*PER_BAR
    raw=np.zeros(N)
    H=np.zeros(N);D=np.zeros(N)
    h0=np.zeros(N);d0=np.zeros(N)
    tmaH=[];tmaD=[];singleH=[];singleD=[]
    for t in times:
        j=int(t*STEP_PER_QUARTER)
        a=l[t]
        s=g.levels_full([t],N_BARS)[t]
        raw[j]=1
        tmaH.append(max(a));tmaD.append(len(a))
        singleH.append(max(s));singleD.append(len(s))
    for i,t in enumerate(times):
        j=int(t*STEP_PER_QUARTER)
        H[j]=tmaH[i]-np.mean(tmaH)
        D[j]=tmaD[i]-np.mean(tmaD)
        h0[j]=singleH[i]-np.mean(singleH)
        d0[j]=singleD[i]-np.mean(singleD)
    return {'onset':raw,'H':H,'D':D,'singleH':h0,'singleD':d0,
            'contextH':H-h0,'contextD':D-d0,
            'contextual_event_fraction':float(np.mean([a!=b for a,b in zip(tmaH,singleH)])),
            'mean_D':float(np.mean(tmaD)),'mean_H':float(np.mean(tmaH))}

def spectrum(x):
    w=np.abs(np.fft.rfft(x))**2
    return w
def divergence(p,q):
    p=p/np.sum(p);q=q/np.sum(q)
    return float(np.sum(np.abs(p-q))/2)
def main():
    assert cyclic_counts(A)==cyclic_counts(B)
    x=build(A);y=build(B)
    stats={}
    for name in ('onset','H','D','singleH','singleD','contextH','contextD'):
        sa=spectrum(x[name]);sb=spectrum(y[name])
        stats[name]={'maximum_absolute_power_difference':float(np.max(abs(sa-sb))),
                     'total_variation_between_normalized_power_spectra':divergence(sa,sb),
                     'same_power_to_1e-7':bool(np.allclose(sa,sb,atol=1e-7,rtol=1e-11)),
                     'total_power_A':float(np.sum(sa)),'total_power_B':float(np.sum(sb))}
    assert stats['onset']['same_power_to_1e-7']
    results={'grid':'4/4 meter, sixteenth-note onset grid, 32 repeated bars',
             'motif_A':A,'motif_B':B,'exact_cyclic_autocorrelation':cyclic_counts(A),
             'musical_onset_spectra_identical':True,
             'event_count_each':len(A)*N_BARS,'frozen_TMA_engine_matched':True,
             'statistics':stats,
             'mean_D':{'A':x['mean_D'],'B':y['mean_D']},
             'event_context_fraction':{'A':x['contextual_event_fraction'],'B':y['contextual_event_fraction']},
             'note':'Onset PSD matching does not imply equality of all conventional sequence descriptors. All transforms are deterministic.'}
    OUT.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
