#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.tma_gait_gauge_validation import (
    download_record,
    transitions,
    first_between,
    project_phase,
    icc3_1,
)
from research.tma_gait_fast_exact import fast_analyze_cycles

SEED = 20260929
THRESHOLD_N = 20.0
PRIMARY_N_CYCLES = 32
SENSITIVITY_N_CYCLES = 24
PAIR_IDS_WITH_02 = [
    "JuPt01", "JuPt03", "JuPt06", "JuPt09", "JuPt10", "JuPt11",
    "JuPt15", "JuPt20", "JuPt21", "JuPt23", "JuPt24", "JuPt28", "JuPt29",
]
# Availability inferred from the public PhysioNet record listing. Avoid probing
# known-absent files, which otherwise incurs long HTTP timeouts.
AVAILABLE_SUFFIXES = {
    "JuPt01": ("01","02","03","04","05","06"),
    "JuPt03": ("01","02","03","04","05","06","07"),
    "JuPt06": ("01","02","03","04","05","06","07"),
    "JuPt09": ("01","02","03","04","05"),
    "JuPt10": ("01","02","03","04","05","06","07"),
    "JuPt11": ("01","02","03","04","05","06","07"),
    "JuPt15": ("01","02","03","04","05","06","07"),
    "JuPt20": ("01","02","03","04","05","06","07"),
    "JuPt21": ("01","02","03","04","05","06","07"),
    "JuPt23": ("01","02","03","04","05","06","07"),
    "JuPt24": ("01","02"),
    "JuPt28": ("01","02","03","04","05","06","07"),
    "JuPt29": ("01","02","03","04","05","06","07"),
}
REPEAT_SUFFIXES = ("02", "03", "04", "05", "06", "07")
CANONICAL = [
    "mean_H", "mean_D", "mean_lambda", "multilevel",
    "pivot_rate", "compound_pivot_fraction", "pivot_depth",
    "tree_span_events", "tree_span_time", "tree_root_fraction",
]
CORE = [
    "mean_H", "mean_D", "mean_lambda", "pivot_rate",
    "pivot_depth", "tree_span_events", "tree_span_time",
]
OUT = Path("research/tma_gait_test_retest_results")
OUT.mkdir(parents=True, exist_ok=True)


def extract_cycles_n(arr: np.ndarray, record: str, n_cycles: int):
    t = arr[:, 0]
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    if not (95.0 <= fs <= 105.0):
        raise RuntimeError(f"{record}: unexpected sampling rate {fs:.3f} Hz")

    left_total = arr[:, 17]
    right_total = arr[:, 18]
    lhs, lto = transitions(left_total, THRESHOLD_N)
    rhs, rto = transitions(right_total, THRESHOLD_N)

    cycles = []
    for a, b in zip(lhs[:-1], lhs[1:]):
        dur_s = (b - a) / fs
        if not (0.55 <= dur_s <= 2.5):
            continue
        e_rto = first_between(rto, a, b)
        if e_rto is None:
            continue
        e_rhs = first_between(rhs, e_rto, b)
        if e_rhs is None:
            continue
        e_lto = first_between(lto, e_rhs, b)
        if e_lto is None:
            continue
        if not (a < e_rto < e_rhs < e_lto < b):
            continue

        events = []
        for label, s in (("LHS", a), ("RTO", e_rto), ("RHS", e_rhs), ("LTO", e_lto)):
            q, depth, err_ms, raw_phase = project_phase(s, a, b, fs)
            events.append({
                "label": label,
                "sample": int(s),
                "raw_phase": raw_phase,
                "phase_num": int(q.numerator),
                "phase_den": int(q.denominator),
                "projection_depth": int(depth),
                "projection_error_ms": float(err_ms),
            })
        cycles.append({
            "start_sample": int(a),
            "end_sample": int(b),
            "duration_s": float(dur_s),
            "events": events,
        })
        if len(cycles) >= n_cycles:
            break

    if len(cycles) < n_cycles:
        raise RuntimeError(f"{record}: only {len(cycles)} valid complete cycles; need {n_cycles}")
    return cycles[:n_cycles], fs


def analyze_record(record: str, n_cycles: int):
    arr = download_record(record)
    cycles, fs = extract_cycles_n(arr, record, n_cycles)
    feat, events, pivots, tree = fast_analyze_cycles(cycles)
    durs = np.asarray([c["duration_s"] for c in cycles], float)
    feat = dict(feat)
    feat.update({
        "record": record,
        "n_cycles_fixed": int(n_cycles),
        "sampling_hz": float(fs),
        "stride_time_mean_s": float(np.mean(durs)),
        "stride_time_sd_s": float(np.std(durs, ddof=1)),
        "stride_time_cv": float(np.std(durs, ddof=1) / np.mean(durs)),
    })
    return feat


def icc_absolute_1(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    x = np.column_stack([a, b])
    n, k = x.shape
    if n < 2:
        return np.nan
    grand = x.mean()
    rowmean = x.mean(axis=1)
    colmean = x.mean(axis=0)
    msr = k * np.sum((rowmean - grand) ** 2) / (n - 1)
    msc = n * np.sum((colmean - grand) ** 2) / (k - 1)
    mse = np.sum((x - rowmean[:, None] - colmean[None, :] + grand) ** 2) / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den != 0 else np.nan


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    q = np.empty(len(vals), float)
    last = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        last = min(last, vals[idx] * len(vals) / rank)
        q[idx] = last
    out[np.flatnonzero(ok)] = q
    return out


def safe_corr(a, b, method="pearson"):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    r = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def bootstrap_pair_metric(a, b, fn, B=4000, seed=SEED):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            v = float(fn(a[idx], b[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 100:
        return (np.nan, np.nan)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_feature_stats(A: pd.DataFrame, B: pd.DataFrame, label: str):
    rows = []
    features = CANONICAL + ["stride_time_mean_s", "stride_time_cv"]
    for feat in features:
        q = pd.DataFrame({"a": A[feat], "b": B[feat]}).dropna()
        a = q.a.to_numpy(float)
        b = q.b.to_numpy(float)
        n = len(a)
        if n < 3:
            continue
        d = b - a
        bias = float(np.mean(d))
        sd_diff = float(np.std(d, ddof=1))
        sw = float(sd_diff / math.sqrt(2.0))
        mdc = float(1.96 * sd_diff)
        pearson_r, pearson_p = safe_corr(a, b, "pearson")
        spearman_r, spearman_p = safe_corr(a, b, "spearman")
        icc_c = icc3_1(a, b)
        icc_a = icc_absolute_1(a, b)
        try:
            paired_t_p = float(stats.ttest_rel(a, b).pvalue)
        except Exception:
            paired_t_p = np.nan
        try:
            wilcoxon_p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue) if not np.allclose(d, 0) else 1.0
        except Exception:
            wilcoxon_p = np.nan
        icc_lo, icc_hi = bootstrap_pair_metric(a, b, icc_absolute_1, B=4000, seed=SEED + len(rows))
        mdc_lo, mdc_hi = bootstrap_pair_metric(
            a, b,
            lambda x, y: 1.96 * float(np.std(y - x, ddof=1)),
            B=4000, seed=SEED + 100 + len(rows),
        )
        rows.append({
            "contrast": label,
            "feature": feat,
            "n_pairs": n,
            "visit1_mean": float(np.mean(a)),
            "visit2_mean": float(np.mean(b)),
            "bias_visit2_minus_visit1": bias,
            "sd_difference": sd_diff,
            "within_subject_sd": sw,
            "mdc95": mdc,
            "mdc95_pct_of_pooled_mean": float(100 * mdc / abs((np.mean(a) + np.mean(b)) / 2))
                if (np.mean(a) + np.mean(b)) != 0 else np.nan,
            "loa95_low": float(bias - 1.96 * sd_diff),
            "loa95_high": float(bias + 1.96 * sd_diff),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p,
            "icc3_1_consistency": icc_c,
            "iccA_1_absolute": icc_a,
            "iccA_1_boot_lo": icc_lo,
            "iccA_1_boot_hi": icc_hi,
            "mdc95_boot_lo": mdc_lo,
            "mdc95_boot_hi": mdc_hi,
            "paired_t_p_bias": paired_t_p,
            "wilcoxon_p_bias": wilcoxon_p,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["paired_t_q_BH"] = bh_adjust(out["paired_t_p_bias"].values)
    return out


def standardized_distance_metrics(A: pd.DataFrame, B: pd.DataFrame, features, Bperm=50000, seed=SEED):
    ids = sorted(set(A.index) & set(B.index))
    A = A.loc[ids, features].astype(float)
    B = B.loc[ids, features].astype(float)
    mask = A.notna().all(axis=1) & B.notna().all(axis=1)
    A = A.loc[mask]
    B = B.loc[mask]
    ids = list(A.index)
    n = len(ids)
    if n < 3:
        return {}
    pooled = pd.concat([A, B], axis=0)
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0, ddof=1).replace(0, np.nan)
    keep = [c for c in features if np.isfinite(sd[c]) and sd[c] > 0]
    Az = ((A[keep] - mu[keep]) / sd[keep]).to_numpy(float)
    Bz = ((B[keep] - mu[keep]) / sd[keep]).to_numpy(float)
    D = np.sqrt(np.nanmean((Az[:, None, :] - Bz[None, :, :]) ** 2, axis=2))
    same = np.diag(D)
    different = D[~np.eye(n, dtype=bool)]
    nearest = np.argmin(D, axis=1)
    top1 = float(np.mean(nearest == np.arange(n)))
    ratio = float(np.mean(same) / np.mean(different))

    rng = np.random.default_rng(seed)
    count_mean = 0
    count_top = 0
    obs_same_mean = float(np.mean(same))
    for _ in range(Bperm):
        perm = rng.permutation(n)
        m = float(np.mean(D[np.arange(n), perm]))
        if m <= obs_same_mean + 1e-15:
            count_mean += 1
        # Relabel B rows under permutation; ask whether nearest row still bears correct subject label.
        hits = np.mean(perm[nearest] == np.arange(n))
        if hits >= top1 - 1e-15:
            count_top += 1

    return {
        "n_pairs": n,
        "features": keep,
        "same_person_distance_mean": obs_same_mean,
        "different_person_distance_mean": float(np.mean(different)),
        "same_over_different_ratio": ratio,
        "top1_identification": top1,
        "chance_top1": float(1.0 / n),
        "permutations": int(Bperm),
        "p_same_distance": float((1 + count_mean) / (Bperm + 1)),
        "p_top1": float((1 + count_top) / (Bperm + 1)),
        "ids": ids,
    }


def regression_gate():
    # Re-run the first 32 cycles of the original frozen GaCo01_01 record and
    # require exact agreement with the already committed split-half-A result.
    expected_path = Path("research/tma_gait_gauge_results/fast_exact_split_half_A.csv")
    if not expected_path.exists():
        raise RuntimeError("Frozen split-half reference table not found.")
    exp = pd.read_csv(expected_path)
    exp = exp[exp.record == "GaCo01_01"]
    if len(exp) != 1:
        raise RuntimeError("GaCo01_01 frozen reference row missing or duplicated.")
    got = analyze_record("GaCo01_01", PRIMARY_N_CYCLES)
    check = {}
    max_abs = 0.0
    for feat in CANONICAL:
        if feat not in exp.columns or feat not in got:
            continue
        e = float(exp.iloc[0][feat])
        g = float(got[feat])
        if np.isnan(e) and np.isnan(g):
            diff = 0.0
        else:
            diff = abs(e - g)
        check[feat] = {"expected": e, "current": g, "abs_diff": diff}
        max_abs = max(max_abs, diff)
    passed = bool(max_abs < 1e-12)
    (OUT / "regression_gate.json").write_text(json.dumps({
        "record": "GaCo01_01",
        "cycles": PRIMARY_N_CYCLES,
        "passed": passed,
        "max_abs_diff": max_abs,
        "features": check,
    }, indent=2))
    if not passed:
        raise RuntimeError(f"Frozen regression gate failed; max difference={max_abs}")
    return {"passed": passed, "max_abs_diff": max_abs}


def run_n_cycles(n_cycles: int):
    record_rows = []
    failures = {}
    records = [
        f"{sid}_{sfx}"
        for sid in PAIR_IDS_WITH_02
        for sfx in AVAILABLE_SUFFIXES[sid]
    ]

    for i, rec in enumerate(records, 1):
        print(f"[{n_cycles} cycles] {i:03d}/{len(records)} {rec}", flush=True)
        try:
            row = analyze_record(rec, n_cycles)
            sid, suffix = rec.rsplit("_", 1)
            row.update({"subject": sid, "suffix": suffix})
            record_rows.append(row)
        except Exception as exc:
            failures[rec] = repr(exc)

    rdf = pd.DataFrame(record_rows)
    rdf.to_csv(OUT / f"record_features_N{n_cycles}.csv", index=False)

    # Primary independent-session comparison: record 01 vs record 02.
    A = rdf[rdf.suffix == "01"].set_index("subject")
    B = rdf[rdf.suffix == "02"].set_index("subject")
    ids = sorted(set(A.index) & set(B.index))
    A2, B2 = A.loc[ids], B.loc[ids]
    primary = paired_feature_stats(A2, B2, "01-vs-02")
    primary.to_csv(OUT / f"paired_stats_01_vs_02_N{n_cycles}.csv", index=False)
    multi_primary = standardized_distance_metrics(A2, B2, CANONICAL, Bperm=50000, seed=SEED + n_cycles)

    # Secondary: baseline 01 vs average of all available repeat records 02..07.
    reps = rdf[rdf.suffix.isin(REPEAT_SUFFIXES)].copy()
    numeric_cols = [c for c in CANONICAL + ["stride_time_mean_s", "stride_time_cv"] if c in reps.columns]
    Bavg = reps.groupby("subject")[numeric_cols].mean()
    repeat_counts = reps.groupby("subject").size().rename("n_repeat_records")
    ids_avg = sorted(set(A.index) & set(Bavg.index))
    Aavg = A.loc[ids_avg]
    Bavg2 = Bavg.loc[ids_avg]
    avgstats = paired_feature_stats(Aavg, Bavg2, "01-vs-repeat-average")
    avgstats.to_csv(OUT / f"paired_stats_01_vs_repeat_average_N{n_cycles}.csv", index=False)
    multi_avg = standardized_distance_metrics(Aavg, Bavg2, CANONICAL, Bperm=50000, seed=SEED + 1000 + n_cycles)

    # Each repeat suffix separately vs baseline, to reveal systematic drift/order effects.
    suffix_tables = []
    for sfx in REPEAT_SUFFIXES:
        Bs = rdf[rdf.suffix == sfx].set_index("subject")
        ids_s = sorted(set(A.index) & set(Bs.index))
        if len(ids_s) < 3:
            continue
        st = paired_feature_stats(A.loc[ids_s], Bs.loc[ids_s], f"01-vs-{sfx}")
        suffix_tables.append(st)
    suffix_stats = pd.concat(suffix_tables, ignore_index=True) if suffix_tables else pd.DataFrame()
    suffix_stats.to_csv(OUT / f"paired_stats_by_suffix_N{n_cycles}.csv", index=False)

    # Subject-level QC / available repeat records.
    qc_rows = []
    for sid in PAIR_IDS_WITH_02:
        available = sorted(rdf.loc[rdf.subject == sid, "suffix"].astype(str).tolist())
        qc_rows.append({
            "subject": sid,
            "baseline_01_ok": "01" in available,
            "repeat_02_ok": "02" in available,
            "available_suffixes": ";".join(available),
            "n_repeat_records": int(sum(s in REPEAT_SUFFIXES for s in available)),
        })
    qcdf = pd.DataFrame(qc_rows)
    qcdf.to_csv(OUT / f"qc_subjects_N{n_cycles}.csv", index=False)

    return {
        "n_cycles": n_cycles,
        "n_records_analyzed": int(len(rdf)),
        "n_failures": int(len(failures)),
        "failures": failures,
        "primary_01_vs_02_n": int(len(ids)),
        "primary_ids": ids,
        "primary_multivariate": multi_primary,
        "repeat_average_n": int(len(ids_avg)),
        "repeat_average_multivariate": multi_avg,
        "repeat_record_counts": {str(k): int(v) for k, v in repeat_counts.to_dict().items()},
        "primary_core": primary[primary.feature.isin(CORE)].to_dict(orient="records") if len(primary) else [],
        "repeat_average_core": avgstats[avgstats.feature.isin(CORE)].to_dict(orient="records") if len(avgstats) else [],
    }


def compare_to_short_term(primary_stats: pd.DataFrame):
    # Existing split-half 32-cycle reference from the frozen Ga cohort.
    A = pd.read_csv("research/tma_gait_gauge_results/fast_exact_split_half_A.csv").set_index("record")
    B = pd.read_csv("research/tma_gait_gauge_results/fast_exact_split_half_B.csv").set_index("record")
    common = sorted(set(A.index) & set(B.index))
    rows = []
    for feat in CORE:
        if feat not in A or feat not in B:
            continue
        d = (B.loc[common, feat] - A.loc[common, feat]).dropna().to_numpy(float)
        short_mdc = float(1.96 * np.std(d, ddof=1))
        z = primary_stats[primary_stats.feature == feat]
        if len(z) != 1:
            continue
        long_mdc = float(z.iloc[0].mdc95)
        rows.append({
            "feature": feat,
            "within_recording_mdc95_32cycles_Ga": short_mdc,
            "independent_session_mdc95_32cycles_Ju": long_mdc,
            "session_to_within_ratio": float(long_mdc / short_mdc) if short_mdc > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mdc_short_vs_independent_session.csv", index=False)
    return out


def write_results_md(summary, mdc_compare):
    p32 = summary["N32"]
    p24 = summary["N24"]
    lines = [
        "# Independent-session TMA gait test–retest",
        "",
        "Protocol: same frozen 20-N LHS/RTO/RHS/LTO event encoder, two-tactus cycle normalization,",
        "shallowest dyadic projection within ±5 ms, repository TMA Levels → H/D/lambda → pivots → trees.",
        "Primary unit: first 32 complete cycles. Sensitivity unit: first 24 complete cycles.",
        "",
        "Study mapping note: the Ju cohort matches the 29-PD/26-control rhythmic-auditory-stimulation",
        "study. The paper reports that an arbitrarily selected subset of 15 PD participants returned",
        "approximately two weeks later for six 100-m no-RAS bouts. PhysioNet does not explicitly label",
        "the suffix-to-condition mapping on the landing page; this analysis therefore treats 01→02 as",
        "the baseline-to-first-repeat mapping supported by the record pattern and study design, and",
        "reports the mapping as an inference rather than a documented file-label definition.",
        "",
        "## Regression gate",
        f"- Frozen 32-cycle GaCo01_01 engine/protocol regression: {'PASS' if summary['regression_gate']['passed'] else 'FAIL'}.",
        f"- Maximum absolute feature difference: {summary['regression_gate']['max_abs_diff']:.3g}.",
        "",
        "## Primary 32-cycle result",
        f"- Eligible 01↔02 pairs: {p32['primary_01_vs_02_n']}.",
        f"- Multivariate same-person / different-person distance ratio: {p32.get('primary_multivariate',{}).get('same_over_different_ratio',float('nan')):.3f}.",
        f"- Same-person distance permutation p: {p32.get('primary_multivariate',{}).get('p_same_distance',float('nan')):.5g}.",
        f"- Top-1 subject identification: {p32.get('primary_multivariate',{}).get('top1_identification',float('nan')):.3f} "
        f"(chance {p32.get('primary_multivariate',{}).get('chance_top1',float('nan')):.3f}; "
        f"permutation p={p32.get('primary_multivariate',{}).get('p_top1',float('nan')):.5g}).",
        "",
        "Core features:",
    ]
    for r in p32["primary_core"]:
        lines.append(
            f"- {r['feature']}: ICC(A,1)={r['iccA_1_absolute']:.3f}, "
            f"ICC(3,1)={r['icc3_1_consistency']:.3f}, MDC95={r['mdc95']:.4f}, "
            f"bias={r['bias_visit2_minus_visit1']:+.4f}, q_bias={r.get('paired_t_q_BH',float('nan')):.4g}."
        )
    lines += [
        "",
        "## 24-cycle sensitivity result",
        f"- Eligible 01↔02 pairs: {p24['primary_01_vs_02_n']}.",
        f"- Multivariate same/different ratio: {p24.get('primary_multivariate',{}).get('same_over_different_ratio',float('nan')):.3f}.",
        f"- Same-person distance permutation p: {p24.get('primary_multivariate',{}).get('p_same_distance',float('nan')):.5g}.",
        "",
        "## 32-cycle within-recording versus independent-session MDC",
    ]
    if len(mdc_compare):
        for _, r in mdc_compare.iterrows():
            lines.append(
                f"- {r['feature']}: within-recording {r['within_recording_mdc95_32cycles_Ga']:.4f}; "
                f"independent-session {r['independent_session_mdc95_32cycles_Ju']:.4f}; "
                f"ratio {r['session_to_within_ratio']:.2f}."
            )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    regression = regression_gate()
    n32 = run_n_cycles(PRIMARY_N_CYCLES)
    n24 = run_n_cycles(SENSITIVITY_N_CYCLES)

    primary32 = pd.read_csv(OUT / "paired_stats_01_vs_02_N32.csv")
    mdc_compare = compare_to_short_term(primary32)

    summary = {
        "analysis_date": "2026-09-29",
        "dataset": "PhysioNet Gait in Parkinson's Disease — Ju cohort",
        "source_study": "Hausdorff et al. 2007 rhythmic auditory stimulation study",
        "source_study_design": {
            "first_session": "six ordered conditions including baseline/RAS/carryover",
            "second_session": "subset of 15 PD approximately two weeks later; six 100-m bouts without RAS",
            "record_mapping_status": "01-to-02 baseline/repeat interpretation inferred from cohort/record pattern; not explicitly documented on PhysioNet landing page",
        },
        "frozen_protocol": {
            "threshold_N": THRESHOLD_N,
            "event_order": ["LHS", "RTO", "RHS", "LTO"],
            "cycle_normalization_tactus_units": 2,
            "projection": "shallowest binary grid point within ±5 ms sampled-time tolerance",
            "primary_cycles": PRIMARY_N_CYCLES,
            "sensitivity_cycles": SENSITIVITY_N_CYCLES,
        },
        "implementation": "specialized binary evaluator previously gated to exact repository-engine equality; current run additionally checks frozen GaCo01_01 32-cycle regression",
        "regression_gate": regression,
        "N32": n32,
        "N24": n24,
        "mdc_short_vs_session": mdc_compare.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_results_md(summary, mdc_compare)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
