#!/usr/bin/env python3
"""Pre-biological falsification: exhaustive conventional-homometric short motifs.

Frozen binary TMA: 4/4, exact quarter-note tactus, sixteenth subdivisions.
Enumerate every 4-7-hit motif with onset at bar start, 16 repeated bars.
Group by exact conventional cyclic autocorrelation and circular interval histogram,
search matches of singleton-fractal Fourier power H AND D. Compare event-context
residual + Level spectra of the most stringently controlled available pair.
Never call difference unique beyond full onset time series or complex Fourier.
"""
from __future__ import annotations
import json,itertools,collections,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import tma_variable_drums as tma

N_BARS=16
GRID=4
BAR=16
N=N_BARS*BAR
OUT=Path("research/tma_controlled_spectra_results")
OUT.mkdir(parents=True,exist_ok=True)
EPS=1e-12
def cyc_acf(motif):
    s=set(motif)
    return tuple(sum(((x+d)%BAR) in s for x in motif) for d in range(BAR))
def intervals(motif):
    x=list(motif)+[motif[0]+BAR]
    return tuple(sorted(b-a for a,b in zip(x,x[1:])))
def make_events(pat):
    return tuple(Fraction(4*bar)+Fraction(i,GRID) for bar in range(N_BARS) for i in pat)
def waves(events,single):
    levels=tma.levels_full(events,N_BARS)
    waves={k:np.zeros(N,dtype=int) for k in ('raw','H','D','sH','sD','cH','cD')}
    lev=[]
    for tm in events:
        j=int(tm*GRID)
        s=levels[tm];l0=single[j]
        waves['raw'][j]=1
        waves['H'][j]=max(s)
        waves['D'][j]=len(s)
        waves['sH'][j]=max(l0)
        waves['sD'][j]=len(l0)
        waves['cH'][j]=max(s)-max(l0)
        waves['cD'][j]=len(s)-len(l0)
        lev.append(s)
    return waves,levels
def autocorr(x):
    # Integer periodic autocorrelation => EXACT equivalence to Fourier power.
    F=np.fft.fft(x)
    C=np.fft.ifft(F*np.conj(F)).real
    return tuple(np.rint(C).astype(np.int64))
def centered_power_key(x):
    n=int(np.count_nonzero(x))
    if n<=1: return autocorr(x)
    # center only at event locations, integer arithmetic
    z=x.astype(np.int64)
    m=int(z.sum())
    z=np.where(z!=0, n*z-m,0)
    return autocorr(z)
def power(x):
    F=np.fft.rfft(x.astype(float))
    q=np.abs(F)**2
    q[0]=0
    return q
def tv(p,q):
    if p.sum()<EPS or q.sum()<EPS:return 0. if p.sum()<EPS and q.sum()<EPS else 1.
    return float(.5*np.sum(abs(p/p.sum()-q/q.sum())))
def signature(pat,sh,sd):
    raw=np.zeros(N,int)
    for bar in range(N_BARS):
        raw[bar*BAR+np.array(pat,dtype=int)]=1
    return centered_power_key(raw),centered_power_key(sh*raw),centered_power_key(sd*raw)
def components(a,b,single):
    wa,la=waves(make_events(a),single)
    wb,lb=waves(make_events(b),single)
    results={}
    for key in ('raw','H','D','sH','sD','cH','cD'):
        xa=wa[key].astype(float);xb=wb[key].astype(float)
        if key!='raw':
            for arr in (xa,xb):
                arr[arr!=0]-=0 # preserve weights; DC excluded below
        pa=power(xa);pb=power(xb)
        results[key]={
          'power_tv_distance':tv(pa,pb),
          'maximum_absolute_power_difference':float(np.max(abs(pa-pb))),
          'exact_power_equal':bool(np.allclose(pa,pb,rtol=1e-10,atol=1e-7)),
          'power_A':float(pa.sum()),'power_B':float(pb.sum())}
    # count event contextual influence
    ca=np.mean([la[tm]!=single[int(tm*GRID)] for tm in la])
    cb=np.mean([lb[tm]!=single[int(tm*GRID)] for tm in lb])
    results['context_affected_fraction']={'A':float(ca),'B':float(cb)}
    # compare actual Level occupancy using full half-peak cancellation
    def level_fft(levels):
        m=max(max(v) for v in levels.values())
        ar=np.zeros((m,N))
        for tm,s in levels.items():
            j=int(tm*GRID)
            for k in s:ar[k-1,j]=1
        Z=np.fft.rfft(ar,axis=1)
        return Z
    za=level_fft(la);zb=level_fft(lb)
    m=max(len(za),len(zb))
    xa=np.pad(za,((0,m-len(za)),(0,0)))
    xb=np.pad(zb,((0,m-len(zb)),(0,0)))
    q1=np.sum(np.abs(xa)**2,axis=0);q2=np.sum(np.abs(xb)**2,axis=0)
    q1[0]=0;q2[0]=0
    intera=np.abs(xa.sum(axis=0))**2-q1
    interb=np.abs(xb.sum(axis=0))**2-q2
    results['level_individual_power_TV']=tv(q1,q2)
    results['level_interference_delta_norm']=float(np.linalg.norm(intera[1:]-interb[1:])/max(np.linalg.norm(intera[1:]),np.linalg.norm(interb[1:]),1e-9))
    return results
def main():
    start=time.monotonic()
    single=[]
    for j in range(N):
        tm=Fraction(j,GRID)
        single.append(tma.levels_full([tm],N_BARS)[tm])
    SH=np.array([max(s) for s in single],int)
    SD=np.array([len(s) for s in single],int)
    groups=collections.defaultdict(list)
    base_power_groups=collections.defaultdict(list)
    count_by_k=collections.Counter()
    for k in (4,5,6,7):
        for tail in itertools.combinations(range(1,BAR),k-1):
            pat=(0,*tail)
            count_by_k[k]+=1
            groupkey=(k,cyc_acf(pat),intervals(pat))
            groups[groupkey].append(pat)
            base_power_groups[(k,cyc_acf(pat))].append(pat)
    grouped=[s for s in groups.values() if len(s)>1]
    paired=sum(len(g)*(len(g)-1)//2 for g in grouped)
    stringent=[];candidates=[]
    # For every onset PSD & IOI-matched group, scan all pairs:
    # prioritize both singleton H/D equal, then near static spectra.
    for g in grouped:
        sigs=[]
        for pat in g:
            raw=np.zeros(N,int)
            for b in range(N_BARS):raw[b*BAR+np.array(pat,dtype=int)]=1
            pa=power(SH*raw);pb=power(SD*raw)
            # integer keys for exact singleton H & D powers, no mean removal
            sH=centered_power_key(SH*raw)
            sD=centered_power_key(SD*raw)
            sigs.append((pat,sH,sD,pa,pb))
        for i in range(len(sigs)):
            p,h,d,ph,pd=sigs[i]
            for j in range(i+1,len(sigs)):
                q,h2,d2,ph2,pd2=sigs[j]
                static_tv=.5*(tv(ph,ph2)+tv(pd,pd2))
                cands={"A":p,"B":q,"singleton_TV_mean":static_tv,
                  "static_H_power_equal":bool(h==h2),"static_D_power_equal":bool(d==d2)}
                if h==h2 and d==d2:
                    stringent.append(cands)
                candidates.append(cands)
    # Include the earlier pair if strict IOI not possible, but no hiding confounds
    candidates.sort(key=lambda x:(not(x['static_H_power_equal'] and x['static_D_power_equal']),x['singleton_TV_mean']))
    examined=min(25,len(candidates))
    best_context=None
    comparisons=[]
    for i,q in enumerate(candidates[:examined]):
        resp=components(q['A'],q['B'],single)
        q={**q,'metrics':resp}
        comparisons.append(q)
        val=resp['cD']['power_tv_distance']+resp['cH']['power_tv_distance']
        if best_context is None or val>best_context[0]:best_context=(val,q)
    # Audit canonical homometric and previous results
    prior_a=(0,1,4,9);prior_b=(0,1,5,8)
    assert cyc_acf(prior_a)==cyc_acf(prior_b)
    prior=components(prior_a,prior_b,single)
    assert prior['raw']['exact_power_equal']
    assert not prior['H']['exact_power_equal']
    if comparisons:
        a,b=comparisons[0]['A'],comparisons[0]['B']
        tma.check_engine(make_events(a),N_BARS)
        tma.check_engine(make_events(b),N_BARS)
    tma.check_engine(make_events(prior_a),N_BARS)
    tma.check_engine(make_events(prior_b),N_BARS)
    output={
      'fixed_beat':'16 bars of 4 quarter-note beats, sixteenth grid, exactly repeated motifs',
      'enumeration':dict(count_by_k),'n_patterns':sum(count_by_k.values()),
      'groups_with_at_least_two_rhythms_same_onset_power_and_circular_IOI_histogram':len(grouped),
      'pairs_matching_onset_Fourier_power_and_circular_IOI_histogram':paired,
      'pairs_also_matching_exact_singleton_H_and_D_Fourier_power':len(stringent),
      'candidate_count_analyzed_frozen_TMA':examined,
      'nearest_singleton_matched_pair':comparisons[0] if comparisons else None,
      'largest_context_difference_among_top_singleton_matched':best_context[1] if best_context else None,
      'prior_homometric_pair':{'A':prior_a,'B':prior_b,
        'same_circular_ioi_histogram':intervals(prior_a)==intervals(prior_b),
        'metrics':prior},
      'math_conclusions':['A complete onset complex DFT is invertible and therefore always determines TMA given beat, origin and rule',
        'PSD equality or autocorrelation equality does not imply onset time series equality',
        'TMA contextual Levels can introduce distinctions beyond a static singleton Level template, tested through explicit residual spectra',
        'No biological outcome is implied by the existence of a constructed pair'],
      'runtime_seconds':time.monotonic()-start}
    OUT.joinpath('results.json').write_text(json.dumps(output,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else x))
    print('RESULT',json.dumps(output,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else x),flush=True)
if __name__=='__main__': main()
