#!/usr/bin/env python3
"""Predefined robustness of two controlled homometric TMA rhythm comparisons
across fixed-origin lengths 8,16,32 bars; no re-tuning motif based on result."""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import tma_variable_drums as t
from tma_controlled_spectra import tv,power
OUT=Path('research/tma_controlled_spectra_results')
P1=((0,2,3,6,9,11,15),(0,2,5,6,9,13,15))
P2=((0,1,3,6,8,12,15),(0,3,5,8,9,10,12))
def spectrum(p,n):
    events=tuple(Fraction(4*b)+Fraction(k,4) for b in range(n) for k in p)
    levels=t.levels_full(events,n)
    single={tm:t.levels_full([tm],n)[tm] for tm in events}
    vec={k:np.zeros(16*n,float) for k in ('raw','H','D','sH','sD','cH','cD')}
    for tm in events:
        j=int(4*tm);lv=levels[tm];sg=single[tm]
        vec['raw'][j]=1
        vec['H'][j]=max(lv);vec['D'][j]=len(lv)
        vec['sH'][j]=max(sg);vec['sD'][j]=len(sg)
        vec['cH'][j]=max(lv)-max(sg)
        vec['cD'][j]=len(lv)-len(sg)
    return {k:power(v) for k,v in vec.items()},events
def main():
    result={}
    for name,(a,b) in [('first_order_exact_staticD',P1),('second_order_matched',P2)]:
        result[name]={}
        for nb in (8,16,32):
            A,eventsA=spectrum(a,nb);B,eventsB=spectrum(b,nb)
            metrics={x:{'psd_TV':tv(A[x],B[x]),
                        'equal_to_1e_minus_7':bool(np.allclose(A[x],B[x],atol=1e-7,rtol=1e-10))}
                    for x in ('raw','H','D','sH','sD','cH','cD')}
            assert metrics['raw']['equal_to_1e_minus_7']
            t.check_engine(eventsA,nb)
            t.check_engine(eventsB,nb)
            result[name][str(nb)]={'n_events':len(eventsA),'metrics':metrics}
            print('PASS frozen exact',name,nb,flush=True)
    report={'pairs':{'first_order_exact_staticD':P1,'second_order_matched':P2},
            'recording_lengths_tested_bars':[8,16,32],
            'results':result,
            'limits':['Each bar repeats the identical motif; this tests nested origin scale robustness, not slowly changing real-world physiology',
                      'Binary 4/4 exact quantized sixteenth grid; no other time division inferred',
                      'A rhythm pair may cease to match singleton-template D power at different global recording lengths',
                      'No new information over complete event sequence; no biological response measured']}
    OUT.joinpath('length_robustness.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
