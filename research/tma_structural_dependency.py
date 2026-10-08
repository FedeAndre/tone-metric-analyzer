#!/usr/bin/env python3
"""Structural-dependency audit: locate the source of event-conditioned TMA Levels.

Hold target beat events, meter, origin, and predeclared total horizon CONSTANT.
Perturb earlier and later beats independently, comparing target-beat Levels.
Also contrast with pivot and predecessor-tree context, whose definitions can
explicitly depend on preceding/following event history.

A stronger negative result would falsify simplistic 'long-memory TMA Levels'.
"""
from fractions import Fraction
from pathlib import Path
import json,random,numpy as np
import tma_variable_drums as tma
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tma_online_causality import PHASES

OUT=Path('research/tma_online_measurement_results');OUT.mkdir(parents=True,exist_ok=True)
NB=16
BEATS=NB*4
TARGET_BEATS=(8,17,29,42,55)
TRIALS=40 # fixed predeclared contexts, no outcome sampling
RNG=np.random.default_rng(20261008)
def generate(patterns):
    return tuple(sorted(Fraction(b)+x for b,s in enumerate(patterns) for x in s))
def wave(t,levels):
    return [{'segment_index':0,'measure_index':int(tm)//4,'measure_number':str(1+int(tm)//4),
      'onset_quarter':str(tm),'offset_in_measure_quarter':str(tm%4),
      'levels':sorted(levels[tm]),'height':max(levels[tm]),'density':len(levels[tm]),
      'lowest_level':min(levels[tm]),'attack':True,'parenthetical':False}
      for tm in t]
def summaries(t,lv):
    v=wave(t,lv)
    piv=build_pivot_profile(v)
    tree=build_tree_profile(v)['branches']
    return {'pivots':len(piv),'tree_branches':len(tree),
       'pivot_recovery_times':[x.get('recovery_onset_quarter') for x in piv],
       'tree_endpoints':[(b.get('source_node_index'),b.get('target_node_index')) for b in tree]}
def main():
    checked=0
    independence_earlier=0
    independence_later=0
    earliest=[]
    full_effects=[]
    future_effects=[]
    for target in TARGET_BEATS:
        for case in range(TRIALS):
            baseline=[PHASES[int(RNG.integers(len(PHASES)))] for _ in range(BEATS)]
            earlier=[p if j>=target else PHASES[int(RNG.integers(len(PHASES)))]
                      for j,p in enumerate(baseline)]
            later=[p if j<=target else PHASES[int(RNG.integers(len(PHASES)))]
                      for j,p in enumerate(baseline)]
            t=generate(baseline);e=generate(earlier);f=generate(later)
            X=tma.levels_full(t,NB);Y=tma.levels_full(e,NB);Z=tma.levels_full(f,NB)
            subject=sorted(q for q in t if target<=q<target+1)
            assert subject==[q for q in e if target<=q<target+1]
            assert subject==[q for q in f if target<=q<target+1]
            checked+=1
            independence_earlier+=int(all(X[q]==Y[q] for q in subject))
            independence_later+=int(all(X[q]==Z[q] for q in subject))
            if len(earliest)<3:
                earliest.append({'beat':target,'held_phase_pattern':[str(x) for x in baseline[target]],
                 'levels':[sorted(X[q]) for q in subject]})
            if case<4 and target==TARGET_BEATS[2]:
                w1=summaries(t,X);w2=summaries(e,Y);w3=summaries(f,Z)
                full_effects.append({'pivot_count_base':w1['pivots'],'pivot_count_earlier_shuffled':w2['pivots'],
                     'tree_count_base':w1['tree_branches'],'tree_count_earlier_shuffled':w2['tree_branches'],
                     'different_pivot_recovery':w1['pivot_recovery_times']!=w2['pivot_recovery_times'],
                     'different_tree_endpoints':w1['tree_endpoints']!=w2['tree_endpoints']})
                future_effects.append({'pivot_count_base':w1['pivots'],'pivot_count_later_shuffled':w3['pivots'],
                     'tree_count_base':w1['tree_branches'],'tree_count_later_shuffled':w3['tree_branches'],
                     'different_pivot_recovery':w1['pivot_recovery_times']!=w3['pivot_recovery_times'],
                     'different_tree_endpoints':w1['tree_endpoints']!=w3['tree_endpoints']})
    assert independence_earlier==checked and independence_later==checked, (
        'Frozen Level assignment has cross-beat event-history dependence')
    report={'n_independent_target_beat_controls':checked,
        'frozen_horizon_bars':NB,
        'target_beat_earlier_perturbation_unchanged_count':independence_earlier,
        'target_beat_later_perturbation_unchanged_count':independence_later,
        'example_unchanged_levels':earliest,
        'whole_record_pivot_tree_sensitivity_to_earlier_history':full_effects,
        'whole_record_pivot_tree_sensitivity_to_later_history':future_effects,
        'interpretation':[
          'Under fixed origin+meter+horizon, event Level membership in target beat depends on that beat own attacks, not earlier or later beat event contents',
          'TMA Level membership is not an event-conditioned long-memory operator across beats in this frozen implementation',
          'Changes in global pivots and predecessor trees are a separate layer that can retain temporal history',
          'Whole-record pivot or tree summaries may change when later events arrive; causality must be tested separately',
          'Boundary scale and parent Level are defined by metric scaffold which is origin dependent'],
        'limits':['Synthetic 4/4 binary quarter-note reference; no triplets or mixed arities',
             'TMA invariance demonstrated for chosen beat subdivisions, beyond mathematical proof by implementation source',
             'Pivots and trees compared as whole-record summaries, not local retroactive change counts']}
    OUT.joinpath('structural_dependency.json').write_text(json.dumps(report,indent=2))
    print('PASS fixed-origin within-beat Level independence',checked)
    print('RESULT',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
