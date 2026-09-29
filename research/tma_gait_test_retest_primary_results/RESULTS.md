# Primary independent-session gait TMA test-retest

Regression gate: PASS; max abs diff 0.
Candidate pairs: 13; analyzed 01↔02 pairs: 13.

## Multivariate
- Same-person distance: 0.9397
- Different-person distance: 1.3139
- Ratio: 0.7152
- p(same-distance): 0.00138999
- Top-1: 0.2308; chance 0.0769; p=0.0772292

## Core features
- mean_H: ICC(A,1)=0.586; ICC(3,1)=0.567; MDC95=0.4066; bias=+0.0108; q(bias)=1
- mean_D: ICC(A,1)=0.447; ICC(3,1)=0.433; MDC95=0.2299; bias=+0.0168; q(bias)=1
- mean_lambda: ICC(A,1)=0.602; ICC(3,1)=0.583; MDC95=0.5448; bias=-0.0060; q(bias)=1
- pivot_rate: ICC(A,1)=0.477; ICC(3,1)=0.464; MDC95=0.0807; bias=-0.0066; q(bias)=1
- pivot_depth: ICC(A,1)=0.088; ICC(3,1)=0.082; MDC95=0.3750; bias=-0.0075; q(bias)=1
- tree_span_events: ICC(A,1)=0.620; ICC(3,1)=0.614; MDC95=0.1084; bias=-0.0127; q(bias)=1
- tree_span_time: ICC(A,1)=0.603; ICC(3,1)=0.585; MDC95=0.0853; bias=-0.0021; q(bias)=1

## Within-recording versus independent-session MDC
- mean_H: 0.4167 -> 0.4066 (x0.98)
- mean_D: 0.1466 -> 0.2299 (x1.57)
- mean_lambda: 0.4852 -> 0.5448 (x1.12)
- pivot_rate: 0.0694 -> 0.0807 (x1.16)
- pivot_depth: 0.4317 -> 0.3750 (x0.87)
- tree_span_events: 0.1183 -> 0.1084 (x0.92)
- tree_span_time: 0.0671 -> 0.0853 (x1.27)
