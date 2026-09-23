#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

from tone_metric.engine import analyze
from tone_metric.models import Hit, MeasureInfo
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tone_metric.waves import build_wave_profile

SEED = 20260923
FS_NOMINAL = 100.0
THRESHOLD_N = 20.0
PROJECTION_TOL_S = 0.005
N_CYCLES = 64
HALF_CYCLES = 32
SCRAMBLE_B = 80
PERM_B = 5000

OUT = Path("research/tma_gait_gauge_results")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("/tmp/tma_gaitpdb")
CACHE.mkdir(parents=True, exist_ok=True)

CONTROL_IDS = [f"GaCo{i:02d}_01" for i in range(1, 18)] + ["GaCo22_01"]
PD_IDS = [f"GaPt{i:02d}_01" for i in range(3, 10)] + [f"GaPt{i:02d}_01" for i in range(12, 34)]
RECORDS = [(r, "control") for r in CONTROL_IDS] + [(r, "pd") for r in PD_IDS]

EVENT_TYPES = ("LHS", "RTO", "RHS", "LTO")

# Previously reported frozen-Ga population summaries. These are regression
# targets only: the current paper-aligned engine is not forced to match them.
FROZEN_TARGETS = {
    "mean_D": {"control": 2.040, "pd": 1.883},
    "RHS_mean_D": {"control": 1.713, "pd": 1.343},
    "RHS_multilevel": {"control": 0.318, "pd": 0.169},
    "mean_lambda": {"control": 7.067, "pd": 7.341},
    "RHS_mean_lambda": {"control": 7.820, "pd": 8.464},
    "tree_span_events": {"control": 1.261, "pd": 1.312},
}

BASE_URLS = (
    "https://physionet.org/files/gaitpdb/1.0.0/{record}.txt",
    "https://archive.physionet.org/pn3/gaitpdb/{record}.txt",
)


def download_record(record: str) -> np.ndarray:
    path = CACHE / f"{record}.txt"
    if not path.exists():
        last_error = None
        for tmpl in BASE_URLS:
            url = tmpl.format(record=record)
            try:
                rr = requests.get(url, timeout=90)
                rr.raise_for_status()
                if len(rr.content) < 10000:
                    raise RuntimeError(f"implausibly short response ({len(rr.content)} bytes)")
                path.write_bytes(rr.content)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not download {record}: {last_error}")
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 19:
        raise RuntimeError(f"{record}: expected >=19 columns, got {arr.shape}")
    return arr[:, :19]


def transitions(force: np.ndarray, threshold: float = THRESHOLD_N):
    contact = np.asarray(force, float) >= threshold
    hs = np.flatnonzero((~contact[:-1]) & contact[1:]) + 1
    to = np.flatnonzero(contact[:-1] & (~contact[1:])) + 1
    return hs.astype(int), to.astype(int)


def first_between(values: np.ndarray, lo: int, hi: int):
    i = np.searchsorted(values, lo + 1, side="left")
    if i < len(values) and values[i] < hi:
        return int(values[i])
    return None


def project_phase(sample: int, start: int, end: int, fs: float):
    if sample == start:
        return Fraction(0), 0, 0.0, 0.0
    raw = Fraction(2 * (sample - start), end - start)
    raw_float = float(raw)
    cycle_s = (end - start) / fs
    for depth in range(0, 21):
        den = 2 ** depth
        k = int(math.floor(raw_float * den + 0.5))
        q = Fraction(k, den)
        err_s = abs(float(q - raw) * cycle_s / 2.0)
        if err_s <= PROJECTION_TOL_S + 1e-12:
            return q, depth, 1000.0 * err_s, raw_float
    # This should never be needed at 100 Hz, but fail closed to the exact
    # sampled phase rather than inventing a coarser position.
    return raw, -1, 0.0, raw_float


def extract_cycles(arr: np.ndarray, record: str):
    t = arr[:, 0]
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    if not (95.0 <= fs <= 105.0):
        raise RuntimeError(f"{record}: unexpected sampling rate {fs:.3f} Hz")

    left_sensor_sum = np.sum(arr[:, 1:9], axis=1)
    right_sensor_sum = np.sum(arr[:, 9:17], axis=1)
    left_total = arr[:, 17]
    right_total = arr[:, 18]
    force_integrity = {
        "left_sum_median_abs_error": float(np.median(np.abs(left_sensor_sum - left_total))),
        "right_sum_median_abs_error": float(np.median(np.abs(right_sensor_sum - right_total))),
    }

    lhs, lto = transitions(left_total)
    rhs, rto = transitions(right_total)

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
        if len(cycles) >= N_CYCLES:
            break

    if len(cycles) < N_CYCLES:
        raise RuntimeError(f"{record}: only {len(cycles)} valid complete cycles; need {N_CYCLES}")
    return cycles[:N_CYCLES], fs, force_integrity


def frac_from_event(event):
    return Fraction(int(event["phase_num"]), int(event["phase_den"]))


def clone_cycles(cycles):
    return [
        {
            "start_sample": c["start_sample"],
            "end_sample": c["end_sample"],
            "duration_s": c["duration_s"],
            "events": [dict(e) for e in c["events"]],
        }
        for c in cycles
    ]


def cycle_order_scramble(cycles, rng):
    idx = rng.permutation(len(cycles))
    return [clone_cycles([cycles[i]])[0] for i in idx]


def phase_label_scramble(cycles, rng):
    out = clone_cycles(cycles)
    for label in ("RTO", "RHS", "LTO"):
        vals = []
        for c in out:
            e = next(e for e in c["events"] if e["label"] == label)
            vals.append((e["phase_num"], e["phase_den"], e["projection_depth"], e["projection_error_ms"], e["raw_phase"]))
        vals = [vals[i] for i in rng.permutation(len(vals))]
        for c, val in zip(out, vals):
            e = next(e for e in c["events"] if e["label"] == label)
            e["phase_num"], e["phase_den"], e["projection_depth"], e["projection_error_ms"], e["raw_phase"] = val
    return out


def build_hits_measures(cycles):
    hits = []
    labels = []
    cycles_for_event = []
    measures = []
    for i, c in enumerate(cycles):
        start = Fraction(2 * i)
        measures.append(
            MeasureInfo(
                index=i,
                number=str(i + 1),
                start=start,
                full_duration=Fraction(2),
                actual_duration=Fraction(2),
                pickup_shift=Fraction(0),
                numerator=2,
                denominator=4,
                implicit=False,
                opening_anacrusis=False,
            )
        )
        for event in c["events"]:
            q = frac_from_event(event)
            onset = start + q
            hits.append(
                Hit(
                    onset=onset,
                    duration=Fraction(0),
                    measure_index=i,
                    measure_number=str(i + 1),
                    offset_in_measure=q,
                    sources=[],
                    canonical_recovered=False,
                )
            )
            labels.append(event["label"])
            cycles_for_event.append(i)
    order = sorted(range(len(hits)), key=lambda j: (hits[j].onset, EVENT_TYPES.index(labels[j])))
    hits = [hits[j] for j in order]
    labels = [labels[j] for j in order]
    cycles_for_event = [cycles_for_event[j] for j in order]
    return hits, measures, labels, cycles_for_event


def analyze_cycles(cycles):
    hits, measures, labels, event_cycles = build_hits_measures(cycles)
    result = analyze(hits, measures)
    rows = [e for seg in result["segments"] for e in seg["events"]]
    if len(rows) != len(labels):
        raise RuntimeError(f"event count mismatch: engine {len(rows)} vs input {len(labels)}")

    event_df = pd.DataFrame({
        "label": labels,
        "cycle_index": event_cycles,
        "H": [int(r["tone_metric_height"]) for r in rows],
        "D": [int(r["tone_metric_density"]) for r in rows],
        "lambda": [int(r["lowest_tone_metric_level"]) for r in rows],
        "levels": [";".join(str(x) for x in r["tone_metric_levels"]) for r in rows],
        "onset": [r["onset_quarter"] for r in rows],
    })
    if (event_df["D"] <= 0).any():
        raise RuntimeError(f"{int((event_df['D'] <= 0).sum())} event(s) were not resolved by the TMA engine")

    wave = build_wave_profile(result)
    pivots = build_pivot_profile(wave)
    tree = build_tree_profile(wave)
    branches = tree.get("branches", [])

    feat = {
        "n_cycles": len(cycles),
        "n_events": len(event_df),
        "mean_H": float(event_df.H.mean()),
        "mean_D": float(event_df.D.mean()),
        "mean_lambda": float(event_df["lambda"].mean()),
        "multilevel": float((event_df.D > 1).mean()),
        "pivot_rate": float(len(pivots) / len(event_df)),
        "compound_pivot_fraction": float(np.mean([p["compound"] for p in pivots])) if pivots else np.nan,
        "pivot_depth": float(np.mean([p["drop_depth"] for p in pivots])) if pivots else np.nan,
        "tree_span_events": float(np.mean([
            int(b["target_node_index"]) - int(b["source_node_index"]) for b in branches
        ])) if branches else np.nan,
        "tree_span_time": float(np.mean([
            float(Fraction(str(b["distance_quarter"]))) for b in branches
        ])) if branches else np.nan,
        "tree_root_fraction": float(np.mean([bool(n["root"]) for n in tree.get("nodes", [])])) if tree.get("nodes") else np.nan,
    }
    for label in EVENT_TYPES:
        z = event_df[event_df.label == label]
        feat[f"{label}_mean_H"] = float(z.H.mean())
        feat[f"{label}_mean_D"] = float(z.D.mean())
        feat[f"{label}_mean_lambda"] = float(z["lambda"].mean())
        feat[f"{label}_multilevel"] = float((z.D > 1).mean())

    depths = [e["projection_depth"] for c in cycles for e in c["events"] if e["label"] != "LHS"]
    errors = [e["projection_error_ms"] for c in cycles for e in c["events"] if e["label"] != "LHS"]
    feat["projection_depth_mean"] = float(np.mean(depths))
    feat["projection_error_ms_mean"] = float(np.mean(errors))
    feat["projection_error_ms_max"] = float(np.max(errors))
    return feat, event_df, pivots, tree


def hedges_g(x0, x1):
    x0 = np.asarray(x0, float)
    x1 = np.asarray(x1, float)
    n0, n1 = len(x0), len(x1)
    sp2 = ((n0 - 1) * np.var(x0, ddof=1) + (n1 - 1) * np.var(x1, ddof=1)) / (n0 + n1 - 2)
    if sp2 <= 0:
        return np.nan
    d = (np.mean(x1) - np.mean(x0)) / math.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (n0 + n1) - 9.0)
    return float(J * d)


def icc3_1(a, b):
    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    n, k = x.shape
    grand = x.mean()
    rowmean = x.mean(axis=1)
    colmean = x.mean(axis=0)
    msr = k * np.sum((rowmean - grand) ** 2) / (n - 1)
    mse = np.sum((x - rowmean[:, None] - colmean[None, :] + grand) ** 2) / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse
    return float((msr - mse) / den) if den > 0 else np.nan


def corr_safe(a, b, method="pearson"):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan, np.nan
    if method == "pearson":
        r = stats.pearsonr(a, b)
    else:
        r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def select_id_columns(df_a, df_b):
    preferred = [
        "mean_H", "mean_D", "mean_lambda", "multilevel",
        "pivot_rate", "compound_pivot_fraction", "pivot_depth",
        "tree_span_events",
        "RTO_mean_D", "RTO_mean_lambda", "RTO_multilevel",
        "RHS_mean_D", "RHS_mean_lambda", "RHS_multilevel",
        "LTO_mean_D", "LTO_mean_lambda", "LTO_multilevel",
    ]
    return [c for c in preferred if c in df_a.columns and c in df_b.columns
            and np.isfinite(df_a[c]).all() and np.isfinite(df_b[c]).all()
            and np.nanstd(np.r_[df_a[c].values, df_b[c].values]) > 0]


def identification_metrics(df_a, df_b, cols, rng=None):
    A = df_a[cols].to_numpy(float)
    B = df_b[cols].to_numpy(float)
    pooled = np.vstack([A, B])
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    A = (A - mu) / sd
    B = (B - mu) / sd
    dist = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    nearest = np.argmin(dist, axis=1)
    ranks = np.array([1 + np.sum(dist[i] < dist[i, i]) for i in range(len(A))])
    same_dist = np.diag(dist)
    off = dist[~np.eye(len(A), dtype=bool)]
    groups = df_a["group"].to_numpy()
    within_group_nearest = []
    for i in range(len(A)):
        candidates = np.flatnonzero(groups == groups[i])
        j = candidates[np.argmin(dist[i, candidates])]
        within_group_nearest.append(int(j == i))
    return {
        "top1_rate": float(np.mean(nearest == np.arange(len(A)))),
        "within_group_top1_rate": float(np.mean(within_group_nearest)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "same_person_distance_mean": float(np.mean(same_dist)),
        "different_person_distance_mean": float(np.mean(off)),
        "distance_ratio_same_over_different": float(np.mean(same_dist) / np.mean(off)),
    }, dist


def permutation_top1_p(dist, rng, B=PERM_B):
    n = dist.shape[0]
    nearest = np.argmin(dist, axis=1)
    obs = np.mean(nearest == np.arange(n))
    vals = np.empty(B)
    for b in range(B):
        perm = rng.permutation(n)
        vals[b] = np.mean(perm[nearest] == np.arange(n))
    return {
        "observed": float(obs),
        "null_mean": float(vals.mean()),
        "null_q025": float(np.quantile(vals, 0.025)),
        "null_q975": float(np.quantile(vals, 0.975)),
        "p_ge_observed": float((1 + np.sum(vals >= obs)) / (B + 1)),
        "chance": float(1.0 / n),
    }


def full_population(subjects):
    rows = []
    event_tables = {}
    for i, sub in enumerate(subjects, 1):
        print(f"[full] {i:02d}/{len(subjects)} {sub['record']}", flush=True)
        feat, events, pivots, tree = analyze_cycles(sub["cycles"])
        rows.append({"record": sub["record"], "group": sub["group"], **feat})
        event_tables[sub["record"]] = events
    df = pd.DataFrame(rows).sort_values("record").reset_index(drop=True)
    df.to_csv(OUT / "full_subject_features.csv", index=False)
    return df, event_tables


def group_table(df):
    feature_cols = [c for c in df.columns if c not in {"record", "group"} and pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for c in feature_cols:
        x0 = df.loc[df.group == "control", c].dropna().to_numpy(float)
        x1 = df.loc[df.group == "pd", c].dropna().to_numpy(float)
        if len(x0) < 3 or len(x1) < 3:
            continue
        tt = stats.ttest_ind(x0, x1, equal_var=False)
        out.append({
            "feature": c,
            "control_mean": float(np.mean(x0)),
            "pd_mean": float(np.mean(x1)),
            "difference_pd_minus_control": float(np.mean(x1) - np.mean(x0)),
            "welch_p": float(tt.pvalue),
            "hedges_g": hedges_g(x0, x1),
        })
    z = pd.DataFrame(out).sort_values("welch_p")
    z.to_csv(OUT / "group_comparisons.csv", index=False)
    return z


def regression_targets(df):
    rows = []
    for feature, target in FROZEN_TARGETS.items():
        if feature not in df.columns:
            continue
        c = float(df.loc[df.group == "control", feature].mean())
        p = float(df.loc[df.group == "pd", feature].mean())
        rows.append({
            "feature": feature,
            "current_control": c,
            "frozen_control": target["control"],
            "abs_diff_control": abs(c - target["control"]),
            "current_pd": p,
            "frozen_pd": target["pd"],
            "abs_diff_pd": abs(p - target["pd"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "frozen_regression_check.csv", index=False)
    return out


def split_half(subjects):
    arows, brows = [], []
    for i, sub in enumerate(subjects, 1):
        print(f"[split] {i:02d}/{len(subjects)} {sub['record']}", flush=True)
        fa, *_ = analyze_cycles(sub["cycles"][:HALF_CYCLES])
        fb, *_ = analyze_cycles(sub["cycles"][HALF_CYCLES:N_CYCLES])
        arows.append({"record": sub["record"], "group": sub["group"], **fa})
        brows.append({"record": sub["record"], "group": sub["group"], **fb})
    a = pd.DataFrame(arows).sort_values("record").reset_index(drop=True)
    b = pd.DataFrame(brows).sort_values("record").reset_index(drop=True)
    a.to_csv(OUT / "split_half_A.csv", index=False)
    b.to_csv(OUT / "split_half_B.csv", index=False)

    reliability = []
    numeric = [c for c in a.columns if c not in {"record", "group"} and c in b.columns
               and pd.api.types.is_numeric_dtype(a[c])]
    for c in numeric:
        mask = np.isfinite(a[c].to_numpy(float)) & np.isfinite(b[c].to_numpy(float))
        if mask.sum() < 10:
            continue
        x, y = a.loc[mask, c].to_numpy(float), b.loc[mask, c].to_numpy(float)
        pr, pp = corr_safe(x, y, "pearson")
        sr, sp = corr_safe(x, y, "spearman")
        reliability.append({
            "feature": c,
            "n": int(mask.sum()),
            "half_A_mean": float(x.mean()),
            "half_B_mean": float(y.mean()),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp,
            "icc3_1": icc3_1(x, y),
        })
    rel = pd.DataFrame(reliability).sort_values("icc3_1", ascending=False)
    rel.to_csv(OUT / "split_half_reliability.csv", index=False)

    cols = select_id_columns(a, b)
    ident, dist = identification_metrics(a, b, cols)
    rng = np.random.default_rng(SEED + 100)
    perm = permutation_top1_p(dist, rng)
    return a, b, rel, cols, ident, perm


def scramble_validation(subjects, id_cols, real_ident):
    rng = np.random.default_rng(SEED + 200)
    modes = ("cycle_order", "phase_label")
    results = {}
    for mode in modes:
        top1 = []
        within = []
        ratio = []
        median_rank = []
        core_iccs = {"mean_D": [], "mean_lambda": [], "tree_span_events": [], "pivot_rate": []}
        for rep in range(SCRAMBLE_B):
            arows, brows = [], []
            for sub in subjects:
                ca = sub["cycles"][:HALF_CYCLES]
                cb = sub["cycles"][HALF_CYCLES:N_CYCLES]
                if mode == "cycle_order":
                    ca = cycle_order_scramble(ca, rng)
                    cb = cycle_order_scramble(cb, rng)
                else:
                    ca = phase_label_scramble(ca, rng)
                    cb = phase_label_scramble(cb, rng)
                fa, *_ = analyze_cycles(ca)
                fb, *_ = analyze_cycles(cb)
                arows.append({"record": sub["record"], "group": sub["group"], **fa})
                brows.append({"record": sub["record"], "group": sub["group"], **fb})
            a = pd.DataFrame(arows).sort_values("record").reset_index(drop=True)
            b = pd.DataFrame(brows).sort_values("record").reset_index(drop=True)
            cols = [c for c in id_cols if c in a.columns and c in b.columns
                    and np.isfinite(a[c]).all() and np.isfinite(b[c]).all()
                    and np.nanstd(np.r_[a[c].values, b[c].values]) > 0]
            im, _ = identification_metrics(a, b, cols)
            top1.append(im["top1_rate"])
            within.append(im["within_group_top1_rate"])
            ratio.append(im["distance_ratio_same_over_different"])
            median_rank.append(im["median_rank"])
            for c in core_iccs:
                if c in a.columns and np.isfinite(a[c]).all() and np.isfinite(b[c]).all():
                    core_iccs[c].append(icc3_1(a[c], b[c]))
            if (rep + 1) % 10 == 0:
                print(f"[scramble:{mode}] {rep+1}/{SCRAMBLE_B}", flush=True)

        arr = np.asarray(top1)
        results[mode] = {
            "B": SCRAMBLE_B,
            "top1_mean": float(arr.mean()),
            "top1_q025": float(np.quantile(arr, 0.025)),
            "top1_q975": float(np.quantile(arr, 0.975)),
            "p_ge_real_top1": float((1 + np.sum(arr >= real_ident["top1_rate"])) / (SCRAMBLE_B + 1)),
            "within_group_top1_mean": float(np.mean(within)),
            "distance_ratio_mean": float(np.mean(ratio)),
            "median_rank_mean": float(np.mean(median_rank)),
            "core_icc_mean": {k: float(np.nanmean(v)) if v else np.nan for k, v in core_iccs.items()},
            "core_icc_q025": {k: float(np.nanquantile(v, 0.025)) if v else np.nan for k, v in core_iccs.items()},
            "core_icc_q975": {k: float(np.nanquantile(v, 0.975)) if v else np.nan for k, v in core_iccs.items()},
        }
    return results


def write_markdown(summary, group, regression, rel):
    lines = [
        "# TMA gait gauge validation — frozen Ga cohort",
        "",
        f"Subjects: {summary['n_subjects']} ({summary['n_controls']} controls, {summary['n_pd']} Parkinson).",
        f"Protocol: first {N_CYCLES} complete LHS→LHS cycles; 20-N contact threshold; LHS→RTO→RHS→LTO; "
        f"each cycle normalized to two tactus units; shallowest dyadic projection within ±{PROJECTION_TOL_S*1000:.0f} ms.",
        "Engine: repository Tone-Metric Levels → H,D,lambda → paper-aligned pivots → nearest-admissible-predecessor trees.",
        "",
        "## Regression against earlier frozen population summaries",
        "",
    ]
    for _, r in regression.iterrows():
        lines.append(
            f"- {r['feature']}: current control {r['current_control']:.4f} vs frozen {r['frozen_control']:.4f}; "
            f"current PD {r['current_pd']:.4f} vs frozen {r['frozen_pd']:.4f}."
        )
    lines += [
        "",
        "## Split-half gauge result",
        "",
        f"- Multivariate top-1 person identification: {summary['split_half']['identification']['top1_rate']:.3f} "
        f"(chance {1/summary['n_subjects']:.3f}; permutation p={summary['split_half']['permutation_top1']['p_ge_observed']:.5g}).",
        f"- Within-diagnostic-group top-1 identification: {summary['split_half']['identification']['within_group_top1_rate']:.3f}.",
        f"- Median true-match rank: {summary['split_half']['identification']['median_rank']:.2f} of {summary['n_subjects']}.",
        f"- Same/different-person distance ratio: {summary['split_half']['identification']['distance_ratio_same_over_different']:.3f}.",
        "",
        "Core feature ICC(3,1):",
    ]
    core = ["mean_D", "mean_lambda", "tree_span_events", "pivot_rate", "RHS_mean_D", "RHS_mean_lambda", "RHS_multilevel"]
    for c in core:
        rr = rel[rel.feature == c]
        if len(rr):
            q = rr.iloc[0]
            lines.append(f"- {c}: ICC={q['icc3_1']:.3f}, Pearson r={q['pearson_r']:.3f}.")
    lines += ["", "## Temporal scrambling", ""]
    for mode, z in summary["scrambling"].items():
        lines.append(
            f"- {mode}: mean top-1 {z['top1_mean']:.3f} "
            f"(95% surrogate range {z['top1_q025']:.3f}–{z['top1_q975']:.3f}); "
            f"surrogate tail p vs real={z['p_ge_real_top1']:.5g}."
        )
    lines += ["", "## Strongest population differences under current engine", ""]
    for _, r in group.head(10).iterrows():
        lines.append(
            f"- {r['feature']}: control {r['control_mean']:.4f}, PD {r['pd_mean']:.4f}, "
            f"Hedges g={r['hedges_g']:+.3f}, Welch p={r['welch_p']:.5g}."
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    subjects = []
    raw_qc = []
    for i, (record, group) in enumerate(RECORDS, 1):
        print(f"[download/extract] {i:02d}/{len(RECORDS)} {record}", flush=True)
        arr = download_record(record)
        cycles, fs, integrity = extract_cycles(arr, record)
        subjects.append({"record": record, "group": group, "cycles": cycles})
        raw_qc.append({
            "record": record,
            "group": group,
            "n_rows": len(arr),
            "fs": fs,
            "n_cycles": len(cycles),
            "mean_stride_s": float(np.mean([c["duration_s"] for c in cycles])),
            "stride_cv": float(np.std([c["duration_s"] for c in cycles], ddof=1) / np.mean([c["duration_s"] for c in cycles])),
            **integrity,
        })
    pd.DataFrame(raw_qc).to_csv(OUT / "raw_qc.csv", index=False)

    full, _ = full_population(subjects)
    group = group_table(full)
    regression = regression_targets(full)
    half_a, half_b, rel, id_cols, ident, perm = split_half(subjects)

    summary = {
        "seed": SEED,
        "dataset": "PhysioNet Gait in Parkinson's Disease (Ga *_01 frozen cohort)",
        "n_subjects": len(subjects),
        "n_controls": len(CONTROL_IDS),
        "n_pd": len(PD_IDS),
        "protocol": {
            "threshold_N": THRESHOLD_N,
            "first_complete_cycles": N_CYCLES,
            "event_order": list(EVENT_TYPES),
            "cycle_normalization_tactus_units": 2,
            "projection": "shallowest binary grid point within ±5 ms sampled-time tolerance",
            "projection_tolerance_s": PROJECTION_TOL_S,
        },
        "implementation": {
            "engine": "repository tone_metric.engine/analyze",
            "wave": "tone_metric.waves/build_wave_profile",
            "pivots": "tone_metric.pivots/build_pivot_profile",
            "trees": "tone_metric.trees/build_tree_profile",
            "branch_base_commit": "b6908ddd0173d97080fb97fea7e086e9781c1cc7",
        },
        "split_half": {
            "half_cycles": HALF_CYCLES,
            "id_feature_columns": id_cols,
            "identification": ident,
            "permutation_top1": perm,
        },
    }

    summary["scrambling"] = scramble_validation(subjects, id_cols, ident)
    summary["runtime_s"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, group, regression, rel)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
