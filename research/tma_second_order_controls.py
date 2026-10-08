#!/usr/bin/env python3
"""Strong controlled TMA motif pairs: identical onset PSD, circular IOI
histograms AND adjacent-IOI-pair histograms, not dihedral symmetries.

Uses frozen engine, singleton fractal control, pairwise matching of all 16th
onsets. Attempts strict matching of singleton D power. Tests ordinary
third-order circular interval statistics (not presumed equal).
"""
import json,collections,itertools,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import tma_variable_drums as t
from tma_controlled_spectra import N,N_BARS,GRID,BAR,cyc_acf,intervals,make_events,power,tv,centered_power_key,components
from tma_nondihedral_controls import canonical,spectral_keys
OUT=Path('research/tma_controlled_spectra_results')
def iois(p):
    x=list(p)+[p[0]+BAR]
    return tuple(y-x for x,y in zip(x,x[1:]))
def pair_hist(p,order):
    v=iois(p)
    return tuple(sorted(tuple(v[(i+j)%len(v)] for j in range(order)) for i in range(len(v))))
def main():
    st=time.monotonic()
    singleton=[t.levels_full([Fraction(j,GRID)],N_BARS)[Fraction(j,GRID)] for j in range(N)]
    SH=np.array([max(x) for x in singleton]);SD=np.array([len(x) for x in singleton])
    bykey=collections.defaultdict(list)
    for k in (4,5,6,7):
        for tail in itertools.combinations(range(1,BAR),k-1):
            p=(0,*tail)
            sig=(k,cyc_acf(p),intervals(p),pair_hist(p,2))
            bykey[sig].append(p)
    pairs=[]
    for motifs in bykey.values():
        if len(motifs)<2:continue
        cs={p:canonical(p) for p in motifs}
        S={p:spectral_keys(p,SH,SD) for p in motifs}
        for i,p in enumerate(motifs):
            for q in motifs[i+1:]:
                if cs[p]==cs[q]:continue
                x,y,hx,dx=S[p];u,v,hy,dy=S[q]
                pairs.append({
                  'A':p,'B':q,'staticH_equal':x==u,'staticD_equal':y==v,
                  'staticH_TV':tv(hx,hy),'staticD_TV':tv(dx,dy),
                  'same_gap_trigrams':pair_hist(p,3)==pair_hist(q,3)})
    # The choice of candidate is based ONLY on conventional and singleton null matching.
    selected=sorted(pairs,key=lambda p:(not p['staticD_equal'],p['staticH_TV']+p['staticD_TV']))[0]
    measured=components(selected['A'],selected['B'],singleton)
    others=[]
    for row in sorted(pairs,key=lambda p:(not p['staticD_equal'],p['staticH_TV']+p['staticD_TV']))[:15]:
        met=components(row['A'],row['B'],singleton)
        others.append({**row,'context_density_TV':met['cD']['power_tv_distance'],
           'full_density_TV':met['D']['power_tv_distance'],'max_abs_full_power':met['D']['maximum_absolute_power_difference'],
           'context_frac_A':met['context_affected_fraction']['A'],
           'context_frac_B':met['context_affected_fraction']['B']})
    t.check_engine(make_events(selected['A']),N_BARS)
    t.check_engine(make_events(selected['B']),N_BARS)
    report={
      'N_patterns':9828,'non_dihedral_same_power_first_order_and_second_order_IOI_pairs':len(pairs),
      'exact_singleton_D_matched_count':sum(row['staticD_equal'] for row in pairs),
      'exact_singleton_H_D_matched_count':sum(row['staticD_equal'] and row['staticH_equal'] for row in pairs),
      'same_gap_trigrams_count':sum(row['same_gap_trigrams'] for row in pairs),
      'first_by_conventional_and_static_null_matching':{**selected,'metrics':measured,
         'interval_sequence_A':iois(selected['A']),'interval_sequence_B':iois(selected['B'])},
      'best_15_static_control_pair_diagnostics':others,
      'runtime_s':time.monotonic()-st,
      'interpretation_limits':[
       'Pair has fixed 4/4 beat plus nonconstant sixteenth-grid attack rhythm',
       'Onset power and both one- and two-event gap occurrence distributions are matched exactly',
       'Higher order cyclic gap distributions or complex Fourier phase may still distinguish pairs',
       'Result is an exact mathematical counterexample against chosen *summaries*, not added information beyond full event timing',
       'No participant or biological system has been tested on these constructed stimuli']}
    OUT.joinpath('second_order_results.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
