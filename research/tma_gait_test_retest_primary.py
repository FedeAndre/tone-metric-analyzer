#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.tma_gait_test_retest import (
    PAIR_IDS_WITH_02, PRIMARY_N_CYCLES, CANONICAL, CORE, OUT,
    analyze_record, paired_feature_stats, standardized_distance_metrics,
    regression_gate, compare_to_short_term,
)

PRIMARY_OUT = Path("research/tma_gait_test_retest_primary_results")
PRIMARY_OUT.mkdir(parents=True, exist_ok=True)

def main():
    gate = regression_gate()
    rows = []
    failures = {}
    records = []
    for sid in PAIR_IDS_WITH_02:
        records.extend([f"{sid}_01", f"{sid}_02"])
    for i, rec in enumerate(records, 1):
        print(f"[primary] {i:02d}/{len(records)} {rec}", flush=True)
        try:
            row = analyze_record(rec, PRIMARY_N_CYCLES)
            sid, suffix = rec.rsplit("_", 1)
            row.update({"subject": sid, "suffix": suffix})
            rows.append(row)
        except Exception as exc:
            failures[rec] = repr(exc)

    rdf = pd.DataFrame(rows)
    rdf.to_csv(PRIMARY_OUT / "record_features.csv", index=False)
    A = rdf[rdf.suffix == "01"].set_index("subject")
    B = rdf[rdf.suffix == "02"].set_index("subject")
    ids = sorted(set(A.index) & set(B.index))
    A = A.loc[ids]
    B = B.loc[ids]
    st = paired_feature_stats(A, B, "01-vs-02")
    st.to_csv(PRIMARY_OUT / "paired_stats.csv", index=False)
    multi = standardized_distance_metrics(A, B, CANONICAL, Bperm=100000, seed=20260929)

    # Recalculate the existing Ga split-half MDC directly for comparison.
    gaA = pd.read_csv("research/tma_gait_gauge_results/fast_exact_split_half_A.csv").set_index("record")
    gaB = pd.read_csv("research/tma_gait_gauge_results/fast_exact_split_half_B.csv").set_index("record")
    common = sorted(set(gaA.index) & set(gaB.index))
    comp = []
    for feat in CORE:
        d = (gaB.loc[common, feat] - gaA.loc[common, feat]).dropna().astype(float)
        short_mdc = float(1.96 * d.std(ddof=1))
        q = st[st.feature == feat]
        if len(q) == 1:
            session_mdc = float(q.iloc[0].mdc95)
            comp.append({
                "feature": feat,
                "within_recording_mdc95": short_mdc,
                "independent_session_mdc95": session_mdc,
                "ratio": session_mdc / short_mdc if short_mdc > 0 else None,
            })
    cdf = pd.DataFrame(comp)
    cdf.to_csv(PRIMARY_OUT / "mdc_comparison.csv", index=False)

    summary = {
        "analysis_date": "2026-09-29",
        "dataset": "PhysioNet gaitpdb Ju cohort",
        "mapping_caveat": "01->02 is inferred as first-session baseline to first second-session no-RAS bout from the record pattern plus the source-study design; the PhysioNet landing page does not explicitly define suffix semantics.",
        "implementation": "frozen 20-N LHS/RTO/RHS/LTO encoder; 32 cycles; exact-equivalent specialized binary TMA evaluator; GaCo01_01 regression gate",
        "regression_gate": gate,
        "n_candidate_pairs": len(PAIR_IDS_WITH_02),
        "n_analyzed_pairs": len(ids),
        "pair_ids": ids,
        "failures": failures,
        "multivariate": multi,
        "core": st[st.feature.isin(CORE)].to_dict(orient="records"),
        "mdc_comparison": comp,
    }
    (PRIMARY_OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Primary independent-session gait TMA test-retest",
        "",
        f"Regression gate: {'PASS' if gate['passed'] else 'FAIL'}; max abs diff {gate['max_abs_diff']:.3g}.",
        f"Candidate pairs: {len(PAIR_IDS_WITH_02)}; analyzed 01↔02 pairs: {len(ids)}.",
        "",
        "## Multivariate",
        f"- Same-person distance: {multi.get('same_person_distance_mean', float('nan')):.4f}",
        f"- Different-person distance: {multi.get('different_person_distance_mean', float('nan')):.4f}",
        f"- Ratio: {multi.get('same_over_different_ratio', float('nan')):.4f}",
        f"- p(same-distance): {multi.get('p_same_distance', float('nan')):.6g}",
        f"- Top-1: {multi.get('top1_identification', float('nan')):.4f}; chance {multi.get('chance_top1', float('nan')):.4f}; p={multi.get('p_top1', float('nan')):.6g}",
        "",
        "## Core features",
    ]
    for r in summary["core"]:
        lines.append(
            f"- {r['feature']}: ICC(A,1)={r['iccA_1_absolute']:.3f}; ICC(3,1)={r['icc3_1_consistency']:.3f}; "
            f"MDC95={r['mdc95']:.4f}; bias={r['bias_visit2_minus_visit1']:+.4f}; q(bias)={r.get('paired_t_q_BH', float('nan')):.4g}"
        )
    lines += ["", "## Within-recording versus independent-session MDC"]
    for r in comp:
        lines.append(f"- {r['feature']}: {r['within_recording_mdc95']:.4f} -> {r['independent_session_mdc95']:.4f} (x{r['ratio']:.2f})")
    (PRIMARY_OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((PRIMARY_OUT / "RESULTS.md").read_text(), flush=True)

if __name__ == "__main__":
    main()
