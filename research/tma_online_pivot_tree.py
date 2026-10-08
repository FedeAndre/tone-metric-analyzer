#!/usr/bin/env python3
"""Causal status and 'number of events required' for full frozen TMA descriptors.
Tests H/D, confirmed pivots and nearest-admissible predecessor TREE edges
without future information. Completed-beat prefixes at a fixed 32-bar horizon
against complete 128-beat (316+ event) event sequence.

Important distinction: a confirmed pivot is knowable only on its recovery,
not at its earlier crest. Incoming TREE edges can finalize on target event;
outgoing branch counts of earlier nodes can evolve with later arrivals.
"""
from fractions import Fraction
from pathlib import Path
import numpy as np,json,time
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
import tma_variable_drums as tma
from tma_online_causality import events

OUT=Path('research/tma_online_measurement_results')
OUT.mkdir(parents=True,exist_ok=True)
def wave(times,levels):
    return [{'segment_index':0,'measure_index':int(q)//4,'measure_number':str(int(q)//4+1),
        'onset_quarter':str(q),'offset_in_measure_quarter':str(q%4),
        'levels':sorted(levels[q]),'height':max(levels[q]),'density':len(levels[q]),
        'lowest_level':min(levels[q]),'attack':True,'parenthetical':False}
        for q in times]
def maps(times,L):
    w=wave(times,L)
    pivots=build_pivot_profile(w)
    tree=build_tree_profile(w)
    # Identity by crest time; confirmed only at recovery time.
    piv={p['pivot_onset_quarter']:(p['recovery_onset_quarter'],p['kind'],p['drop_depth']) for p in pivots}
    incoming={b['target_onset_quarter']:(b['source_onset_quarter'],b['source_level'],b['target_level'])
              for b in tree['branches']}
    return piv,incoming,tree
def main():
    t0=time.perf_counter()
    alltime=events(128)
    nbar=32
    levels=tma.levels_full(alltime,nbar)
    final_piv,final_edges,final_tree=maps(alltime,levels)
    history=[]
    stable_levels=0
    total_old_levels=0
    tree_reassign=[]
    pivot_reassign=[]
    prev_piv={}
    prev_tree={}
    observed_first_pivot=None
    for beat in range(1,129):
        till=tuple(q for q in alltime if q<beat)
        lev=tma.levels_full(till,nbar)
        piv,edg,tree=maps(till,lev)
        if observed_first_pivot is None and piv:observed_first_pivot=(beat,len(till))
        for q in till:
            total_old_levels+=1
            stable_levels+=int(lev[q]==levels[q])
        for target,parent in edg.items():
            if final_edges.get(target)!=parent:
                tree_reassign.append({'asof_beat':beat,'target':target,'interim_source':parent,'final_source':final_edges.get(target)})
        for crest,value in piv.items():
            if final_piv.get(crest)!=value:
                pivot_reassign.append({'asof_beat':beat,'crest':crest,'interim':value,'final':final_piv.get(crest)})
        history.append({'beat':beat,'events':len(till),'confirmed_pivots':len(piv),'tree_branches':len(edg)})
    index={q:i for i,q in enumerate(alltime)}
    pivot_delays=[]
    for crest,(recovery,kind,depth) in final_piv.items():
        p=Fraction(crest);r=Fraction(recovery)
        pivot_delays.append({'crest_time':crest,'recovery_time':recovery,
          'wait_after_crest_events':index[r]-index[p],'wait_after_crest_beats':float(r-p),'compound':kind=='compound'})
    edgespan=[index[Fraction(t)]-index[Fraction(s)]
              for t,(s,_,_) in final_edges.items()]
    delays=np.array([x['wait_after_crest_events'] for x in pivot_delays],dtype=float)
    distances=np.array(edgespan,float)
    # frozen production check at 4, 16, 32 bars, exact same event series
    for nb in (4,16,32):
        subset=[q for q in alltime if q<nb*4]
        tma.check_engine(subset,nb)
    report={
      'n_events':len(alltime),'beats':128,'n_final_pivots':len(final_piv),
      'n_final_tree_branches':len(final_edges),
      'previous_levels_checked':total_old_levels,'previous_levels_equal_final':stable_levels,
      'retroactive_tree_edge_reassignments_at_completed_beats':tree_reassign[:10],
      'n_retroactive_tree_edge_reassignments':len(tree_reassign),
      'retroactive_confirmed_pivot_reassignments_at_completed_beats':pivot_reassign[:10],
      'n_retroactive_confirmed_pivot_reassignments':len(pivot_reassign),
      'first_confirmed_pivot_observed':observed_first_pivot,
      'pivot_confirmation_events_quantiles':np.quantile(delays,[0,.25,.5,.75,.90,1]).tolist(),
      'pivot_confirmation_beats_quantiles':np.quantile([x['wait_after_crest_beats'] for x in pivot_delays],[0,.25,.5,.75,.90,1]).tolist(),
      'fraction_pivots_confirmed_by_2_events_after_crest':float(np.mean(delays<=2)),
      'fraction_pivots_confirmed_by_4_events_after_crest':float(np.mean(delays<=4)),
      'fraction_pivots_confirmed_by_8_events_after_crest':float(np.mean(delays<=8)),
      'tree_branch_event_span_quantiles':np.quantile(distances,[0,.25,.5,.75,.9,1]).tolist(),
      'online_summary_milestones':[x for x in history if x['beat'] in [1,2,4,8,16,32,64,128]],
      'runtime_s':time.perf_counter()-t0,
      'notes':['Pivot is a crest identified only upon first renewed ascent; never assign its finalized value at crest during causal streaming',
       'Tree rule selects closest admissible earlier event, but predecessor assignments may change if underlying Level of target revised',
       'Completed-beat rather than mid-beat prefix to avoid provisional Level-stack revisions',
       'Number of observed events required for the FIRST pivot is descriptively informative, not evidence of superior inferential efficiency',
       'Only binary 4/4 quarter note tactus and synthetic variable rhythm tested']}
    OUT.joinpath('causal_pivot_tree.json').write_text(json.dumps(report,indent=2))
    print('PASS causal measured pivot/tree Level status',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
