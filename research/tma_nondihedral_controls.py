#!/usr/bin/env python3
"""Independent structural test excluding every cyclic rotation/reflection.

Within all 9828 anchored 4-7 hit motifs on exact 16th grid:
 - same cyclic onset autocorrelation (equivalent Fourier power)
 - same circular interval histogram
 - not a rotation/reflection (dihedrally non-equivalent)
 - test equality of static singleton template H and D Fourier power
 - detect contextual TMA D/H differences and evaluate exact power effect sizes.

Frozen-engine check on selected patterns; no claims about clinical outcomes.
"""
import itertools,collections,json,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import tma_variable_drums as t
from tma_controlled_spectra import N,N_BARS,BAR,GRID,cyc_acf,intervals,make_events,waves,power,tv,centered_power_key,components
OUT=Path('research/tma_controlled_spectra_results')
OUT.mkdir(parents=True,exist_ok=True)
def canonical(p):
    return min(tuple(sorted((sg*x+d)%BAR for x in p)) for sg in (-1,1) for d in range(BAR))
def spectral_keys(p,SH,SD):
    mask=np.zeros(N,dtype=int)
    for b in range(N_BARS):mask[b*BAR+np.array(p)]=1
    h=SH*mask;d=SD*mask
    return centered_power_key(h),centered_power_key(d),power(h),power(d)
def main():
    tic=time.monotonic()
    single=[t.levels_full([Fraction(j,GRID)],N_BARS)[Fraction(j,GRID)] for j in range(N)]
    SH=np.array([max(q) for q in single]);SD=np.array([len(q) for q in single])
    grouped=collections.defaultdict(list)
    for k in (4,5,6,7):
        for tail in itertools.combinations(range(1,BAR),k-1):
            motif=(0,*tail)
            grouped[(k,cyc_acf(motif),intervals(motif))].append(motif)
    all_count=0;matched_D=0;matched_HD=0
    distinct=[]
    dist_all=[]
    for g in grouped.values():
        if len(g)<2:continue
        tagged=[(p,canonical(p)) for p in g]
        byclass=collections.defaultdict(list)
        for p,key in tagged:byclass[key].append(p)
        if len(byclass)<2:continue
        data={p:spectral_keys(p,SH,SD) for p in g}
        for i,p in enumerate(g):
            for q in g[i+1:]:
                if canonical(p)==canonical(q):continue
                all_count+=1
                h1,d1,ph1,pd1=data[p];h2,d2,ph2,pd2=data[q]
                exactH=(h1==h2);exactD=(d1==d2)
                matched_D+=exactD;matched_HD+=exactH and exactD
                Hdist=tv(ph1,ph2);Ddist=tv(pd1,pd2)
                dist_all.append((Hdist+Ddist)/2)
                distinct.append({'A':p,'B':q,'same_singleton_H_power':exactH,'same_singleton_D_power':exactD,
                  'static_H_TV':Hdist,'static_D_TV':Ddist,'mean_static_TV':(Hdist+Ddist)/2})
    assert all_count==1696,(all_count,'controlled enumeration changed')
    strict_D=[x for x in distinct if x['same_singleton_D_power']]
    if strict_D:
        near=min(strict_D,key=lambda x:x['static_H_TV'])
    else:
        near=min(distinct,key=lambda x:x['mean_static_TV'])
    # Also fixed first non-dihedral pair, no selection on TMA outcome
    fst=min(distinct,key=lambda x:(x['A'],x['B']))
    # Inspect 25 strongest *control-matched* pairs only, selection never clinical
    pool=sorted(strict_D if strict_D else distinct,key=lambda x:x['mean_static_TV'])[:25]
    tested=[]
    for q in pool:
        measured=components(q['A'],q['B'],single)
        tested.append({**q,'results':measured})
    candidate=max(tested,key=lambda x:x['results']['cD']['power_tv_distance']) if tested else None
    near_details=components(near['A'],near['B'],single)
    fst_details=components(fst['A'],fst['B'],single)
    t.check_engine(make_events(near['A']),N_BARS)
    t.check_engine(make_events(near['B']),N_BARS)
    t.check_engine(make_events(fst['A']),N_BARS)
    t.check_engine(make_events(fst['B']),N_BARS)
    output={
       'N_motifs':9828,'N_non_dihedrally_equivalent_pairs_matching_exact_onset_PSD_and_IOI_histogram':all_count,
       'n_matching_exact_singleton_density_Fourier_power':matched_D,
       'n_matching_both_singleton_H_and_D_Fourier_power':matched_HD,
       'static_template_TV_quantiles':np.quantile(dist_all,[0,.05,.25,.5,.75,1]).tolist(),
       'best_static_template_matched_nontrivial_pair':{**near,'metrics':near_details},
       'lexicographically_first_nontrivial_pair':{**fst,'metrics':fst_details},
       'most_context_discriminated_among_25_closest_static_matches':candidate,
       'first_25_control_matched_summary':[
          {'motifs':[x['A'],x['B']],
           'contextD_TV':x['results']['cD']['power_tv_distance'],
           'fullD_TV':x['results']['D']['power_tv_distance'],
           'staticH_TV':x['static_H_TV'],'staticD_TV':x['static_D_TV']}
          for x in tested],
       'runtime_seconds':time.monotonic()-tic,
       'guardrails':[
        'Dihedral equivalence excluded: motif rotations and reversals are not counted',
        'Circular IOI histogram and complete onset power exactly equal, complex Fourier phase differs',
        'Frozen-TMA contextual terms are deterministic of original event time series; no Shannon information gain',
        'If singleton H cannot be matched, full TMA differences are not uniquely attributable to contextual effects',
        'Picking maximum among 25 comparisons is mathematical construction, not an inferential or neural result']
    }
    OUT.joinpath('nondihedral_results.json').write_text(json.dumps(output,indent=2))
    print('RESULT',json.dumps(output,indent=2),flush=True)
if __name__=='__main__':main()
