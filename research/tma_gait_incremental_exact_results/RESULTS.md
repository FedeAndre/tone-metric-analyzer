# Exact-engine incremental-information validation

Criterion I verdict: **INCONCLUSIVE**

All models use participant-level LOOCV with within-fold standardization and fixed ridge C=1.
GaPt03 lacks Speed_01, leaving 46 participants for the speed-containing comparison.

## Primary comparisons

- core_plus_full: conventional AUC=0.728; +TMA AUC=0.937; ΔAUC=+0.208; bootstrap 95% CI +0.087 to +0.349; p(Δ<=0)=0.00039992.
- expanded_plus_full: conventional AUC=0.865; +TMA AUC=0.948; ΔAUC=+0.083; bootstrap 95% CI -0.010 to +0.192; p(Δ<=0)=0.040592.
- core_plus_compact: conventional AUC=0.728; +TMA AUC=0.946; ΔAUC=+0.218; bootstrap 95% CI +0.099 to +0.365; p(Δ<=0)=0.00019996.
- expanded_plus_compact: conventional AUC=0.865; +TMA AUC=0.956; ΔAUC=+0.091; bootstrap 95% CI +0.008 to +0.198; p(Δ<=0)=0.014997.

## Interpretation rule

PASS requires positive ΔAUC with the full exact-engine TMA block for both the five-variable core conventional baseline and the broader seven-variable baseline, with both paired-bootstrap lower 95% bounds above zero.

The historical 0.770→0.893 result is treated as context only because its exact conventional-feature formulas and ridge implementation were not preserved.
