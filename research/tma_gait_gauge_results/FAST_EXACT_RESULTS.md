# Fast exact TMA gait gauge validation

Repository-engine equivalence checks: 12 / 12 passed.
Subjects: 47 (18 controls, 29 PD).

## Frozen-population regression
- mean_D: current C=2.1230, frozen C=2.0400; current PD=1.9297, frozen PD=1.8830.
- RHS_mean_D: current C=2.0408, frozen C=1.7130; current PD=1.5178, frozen PD=1.3430.
- RHS_multilevel: current C=0.4661, frozen C=0.3180; current PD=0.2565, frozen PD=0.1690.
- mean_lambda: current C=6.7678, frozen C=7.0670; current PD=7.1664, frozen PD=7.3410.
- RHS_mean_lambda: current C=6.9106, frozen C=7.8200; current PD=7.9106, frozen PD=8.4640.
- tree_span_events: current C=2.6344, frozen C=1.2610; current PD=2.6846, frozen PD=1.3120.

## Split-half
- Top-1 person identification: 0.106; chance=0.021; permutation p=0.0023995.
- Within-group top-1: 0.128.
- Median true-match rank: 7.00.
- Same/different distance ratio: 0.666.

## Scrambling
- cycle_order: top-1 mean=0.121, 95%=0.064–0.170, p vs real=0.83582.
- phase_label: top-1 mean=0.109, 95%=0.064–0.170, p vs real=0.60697.

## Core ICCs
- mean_D: ICC=0.852; Pearson r=0.852.
- mean_lambda: ICC=0.699; Pearson r=0.702.
- tree_span_events: ICC=0.365; Pearson r=0.372.
- pivot_rate: ICC=0.605; Pearson r=0.608.
- RHS_mean_D: ICC=0.774; Pearson r=0.774.
- RHS_mean_lambda: ICC=0.683; Pearson r=0.683.
- RHS_multilevel: ICC=0.724; Pearson r=0.724.

## Strongest current group differences
- multilevel: C=0.3789, PD=0.3242, g=-1.834, p=3.1797e-08.
- tree_span_time: C=1.2378, PD=1.2969, g=+1.675, p=2.4813e-07.
- RHS_mean_D: C=2.0408, PD=1.5178, g=-1.808, p=3.2244e-07.
- mean_D: C=2.1230, PD=1.9297, g=-2.097, p=4.6165e-07.
- projection_error_ms_max: C=5.0000, PD=5.0000, g=-1.513, p=5.9081e-07.
- RHS_multilevel: C=0.4661, PD=0.2565, g=-1.580, p=1.2674e-06.
- projection_depth_mean: C=4.0185, PD=4.5499, g=+1.769, p=4.2678e-06.
- mean_lambda: C=6.7678, PD=7.1664, g=+1.769, p=4.2678e-06.
- RHS_mean_lambda: C=6.9106, PD=7.9106, g=+1.453, p=7.9168e-06.
- tree_span_events: C=2.6344, PD=2.6846, g=+1.336, p=3.9295e-05.
