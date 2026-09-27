# Exact TMA ON/OFF levodopa clinical validation

Dataset: Shida et al. public ON/OFF overground walking dataset (Figshare 14896881).
Primary fixed sequence: 24 complete annotated gait cycles per medication state.
Primary paired cohort: 19 patients.
Repository-engine equivalence gate: 8/8 passed.

## Criterion 2: Does TMA move with medication?
A-priori TMA normalization composite mean = 0.2400 OFF-SD units (positive = movement toward the independent control direction).
One-sample t=1.370, p=0.18754; Wilcoxon p=0.072834.

Core TMA features:
- mean_D: OFF=1.9978, ON=2.0510, ON-OFF=+0.0532, dz=+0.511, p=0.038974, FDR q=0.27282; normalizing-direction change=+0.0532.
- multilevel: OFF=0.2993, ON=0.3026, ON-OFF=+0.0033, dz=+0.141, p=0.5461, FDR q=0.63712; normalizing-direction change=+0.0033.
- mean_lambda: OFF=6.5005, ON=6.4781, ON-OFF=-0.0225, dz=-0.104, p=0.65434, FDR q=0.65434; normalizing-direction change=+0.0225.
- RHS_mean_D: OFF=1.3750, ON=1.4583, ON-OFF=+0.0833, dz=+0.271, p=0.25278, FDR q=0.39336; normalizing-direction change=+0.0833.
- RHS_multilevel: OFF=0.1491, ON=0.1754, ON-OFF=+0.0263, dz=+0.255, p=0.28097, FDR q=0.39336; normalizing-direction change=+0.0263.
- RHS_mean_lambda: OFF=7.7281, ON=7.5899, ON-OFF=-0.1382, dz=-0.257, p=0.27748, FDR q=0.39336; normalizing-direction change=+0.1382.
- tree_span_time: OFF=1.1603, ON=1.1467, ON-OFF=-0.0137, dz=-0.293, p=0.21847, FDR q=0.39336; normalizing-direction change=+0.0137.

## Does TMA change track clinical improvement?
- UPDRS_II: Spearman rho=+0.085, permutation p=0.73021, FDR q=0.87626; partial rank r=+0.013 controlling Δspeed, Δstride length, Δcycle time (p=0.95669).
- UPDRS_II_walking: Spearman rho=-0.025, permutation p=0.9221, FDR q=0.9221; partial rank r=-0.009 controlling Δspeed, Δstride length, Δcycle time (p=0.97172).
- UPDRS_III: Spearman rho=+0.188, permutation p=0.43438, FDR q=0.78716; partial rank r=+0.157 controlling Δspeed, Δstride length, Δcycle time (p=0.52183).
- UPDRS_III_walking: Spearman rho=+0.155, permutation p=0.52477, FDR q=0.78716; partial rank r=+0.131 controlling Δspeed, Δstride length, Δcycle time (p=0.59325).
- miniBEST: Spearman rho=+0.442, permutation p=0.059597, FDR q=0.17879; partial rank r=+0.388 controlling Δspeed, Δstride length, Δcycle time (p=0.10046).
- FESI: Spearman rho=+0.483, permutation p=0.040948, FDR q=0.17879; partial rank r=+0.432 controlling Δspeed, Δstride length, Δcycle time (p=0.064593).

## Criterion 3: incremental prediction beyond conventional gait change
- UPDRS_II: conventional LOOCV MAE=2.490, +TMA=2.455; MAE improvement=+0.035, sign-flip p=0.43075; CV R² -0.162 -> -0.143.
- UPDRS_III: conventional LOOCV MAE=9.417, +TMA=9.241; MAE improvement=+0.177, sign-flip p=0.90714; CV R² -0.140 -> -0.129.
- UPDRS_III_walking: conventional LOOCV MAE=0.698, +TMA=0.708; MAE improvement=-0.010, sign-flip p=0.7915; CV R² -0.182 -> -0.239.
- FESI: conventional LOOCV MAE=9.117, +TMA=6.791; MAE improvement=+2.326, sign-flip p=0.31529; CV R² -1.068 -> +0.106.

## Sensitivity to sequence length
- N=24: n=19, TMA normalization mean=+0.2400, p=0.18754; UPDRS-III rho=+0.188.
- N=16: n=20, TMA normalization mean=-0.0057, p=0.97519; UPDRS-III rho=+0.100.
- N=32: n=13, TMA normalization mean=+0.1209, p=0.45842; UPDRS-III rho=+0.096.

## Interpretation rule
A positive normalization score means ON-medication moved the TMA profile in the direction previously observed in healthy controls: higher D/multilevel occupancy and/or lower lambda/tree span. The direction was fixed before examining this ON/OFF dataset.

These analyses test medication responsiveness within the same patients. They do not establish diagnostic specificity or clinical utility unless the effect is reproducible and incremental beyond conventional gait measures.
