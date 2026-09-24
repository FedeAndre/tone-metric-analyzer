#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tma_gait_gauge_validation import RECORDS, download_record, extract_cycles

SEED = 20260924
OUT = Path("research/tma_gait_incremental_exact_results")
OUT.mkdir(parents=True, exist_ok=True)

TMA_FEATURE_PATH = Path("research/tma_gait_gauge_results/fast_exact_full_subject_features.csv")

CONVENTIONAL_CORE = [
    "speed_m_s",
    "stride_cv_pct",
    "swing_cv_pct",
    "bilateral_swing_asym_log_pct",
    "pci_pct",
]

CONVENTIONAL_EXPANDED = [
    "speed_m_s",
    "mean_stride_s",
    "stride_cv_pct",
    "swing_cv_pct",
    "bilateral_swing_asym_log_pct",
    "contralateral_hs_phase_error_pct",
    "pci_pct",
]

# This is the exact-engine multivariate block used for split-half identification
# in the completed validation. It excludes projection-error diagnostics.
TMA_FULL = [
    "mean_H", "mean_D", "mean_lambda", "multilevel",
    "pivot_rate", "compound_pivot_fraction", "pivot_depth",
    "tree_span_events",
    "RTO_mean_D", "RTO_mean_lambda", "RTO_multilevel",
    "RHS_mean_D", "RHS_mean_lambda", "RHS_multilevel",
    "LTO_mean_D", "LTO_mean_lambda", "LTO_multilevel",
]

# Compact, theory-facing subset matching the earlier gait-validation summaries.
TMA_COMPACT = [
    "mean_D", "mean_lambda", "multilevel",
    "pivot_rate", "compound_pivot_fraction", "pivot_depth",
    "tree_span_events",
    "RHS_mean_D", "RHS_mean_lambda", "RHS_multilevel",
]

C_PRIMARY = 1.0
C_SENSITIVITY = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
BOOT_B = 20000

DEMOGRAPHICS_URL = "https://physionet.org/files/gaitpdb/1.0.0/demographics.txt"


def cv_pct(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    m = float(np.mean(x))
    if not np.isfinite(m) or abs(m) < 1e-12:
        return np.nan
    return float(100.0 * np.std(x, ddof=1) / m)


def conventional_from_cycles(cycles, fs: float) -> dict:
    stride = np.asarray([float(c["duration_s"]) for c in cycles], float)
    right_swing = []
    left_swing = []
    phase_deg = []

    for c in cycles:
        ev = {e["label"]: int(e["sample"]) for e in c["events"]}
        a = int(c["start_sample"])
        b = int(c["end_sample"])
        if not (a < ev["RTO"] < ev["RHS"] < ev["LTO"] < b):
            continue
        right_swing.append((ev["RHS"] - ev["RTO"]) / fs)
        left_swing.append((b - ev["LTO"]) / fs)
        phase_deg.append(360.0 * (ev["RHS"] - a) / (b - a))

    right_swing = np.asarray(right_swing, float)
    left_swing = np.asarray(left_swing, float)
    phase_deg = np.asarray(phase_deg, float)

    if min(len(right_swing), len(left_swing), len(phase_deg)) < 10:
        raise RuntimeError("Insufficient conventional-gait events")

    lmean = float(np.mean(left_swing))
    rmean = float(np.mean(right_swing))
    swing_cv = float(np.mean([cv_pct(left_swing), cv_pct(right_swing)]))

    # Standard logarithmic gait-asymmetry magnitude; zero means perfect symmetry.
    asym_log = float(100.0 * abs(math.log(lmean / rmean)))
    # Also retained as a sensitivity diagnostic, not used in the primary block.
    asym_norm = float(100.0 * abs(lmean - rmean) / ((lmean + rmean) / 2.0))

    # Phase Coordination Index (PCI): accuracy + consistency of antiphase stepping.
    phase_accuracy = float(100.0 * np.mean(np.abs(phase_deg - 180.0)) / 180.0)
    phase_cv = cv_pct(phase_deg)
    pci = float(phase_accuracy + phase_cv)

    return {
        "mean_stride_s": float(np.mean(stride)),
        "stride_cv_pct": cv_pct(stride),
        "left_swing_mean_s": lmean,
        "right_swing_mean_s": rmean,
        "left_swing_cv_pct": cv_pct(left_swing),
        "right_swing_cv_pct": cv_pct(right_swing),
        "swing_cv_pct": swing_cv,
        "bilateral_swing_asym_log_pct": asym_log,
        "bilateral_swing_asym_norm_pct": asym_norm,
        "contralateral_hs_phase_error_pct": phase_accuracy,
        "phase_cv_pct": phase_cv,
        "pci_pct": pci,
    }


def load_speed_table() -> dict[str, float]:
    rr = requests.get(DEMOGRAPHICS_URL, timeout=90)
    rr.raise_for_status()
    df = pd.read_csv(io.StringIO(rr.text), sep="\t")
    out = {}
    for _, row in df.iterrows():
        sid = str(row.get("ID", "")).strip()
        try:
            speed = float(row.get("Speed_01"))
        except Exception:
            speed = np.nan
        out[sid] = speed
    return out


def loocv_probabilities(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n = len(y)
    pred = np.full(n, np.nan)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=float(C),
                solver="liblinear",
                max_iter=10000,
                random_state=SEED,
            )),
        ])
        model.fit(X[train], y[train])
        pred[i] = model.predict_proba(X[i:i+1])[:, 1][0]
    return pred


def paired_bootstrap_auc(y, p_base, p_aug, B=BOOT_B, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    p_base = np.asarray(p_base, float)
    p_aug = np.asarray(p_aug, float)
    n = len(y)
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(
            roc_auc_score(yy, p_aug[idx]) - roc_auc_score(yy, p_base[idx])
        )
    d = np.asarray(deltas, float)
    return {
        "B_requested": int(B),
        "B_valid": int(len(d)),
        "delta_mean": float(np.mean(d)),
        "ci95_low": float(np.quantile(d, 0.025)),
        "ci95_high": float(np.quantile(d, 0.975)),
        "p_delta_le_0": float((1 + np.sum(d <= 0)) / (len(d) + 1)),
    }


def evaluate_pair(df, conventional, tma, label, C=C_PRIMARY):
    needed = conventional + tma
    use = df.dropna(subset=needed).copy().reset_index(drop=True)
    y = (use["group"] == "pd").astype(int).to_numpy()
    X0 = use[conventional].to_numpy(float)
    X1 = use[conventional + tma].to_numpy(float)
    p0 = loocv_probabilities(X0, y, C)
    p1 = loocv_probabilities(X1, y, C)
    a0 = float(roc_auc_score(y, p0))
    a1 = float(roc_auc_score(y, p1))
    boot = paired_bootstrap_auc(y, p0, p1, seed=SEED + sum(map(ord, label)))
    pred = use[["record", "group"]].copy()
    pred["y"] = y
    pred["p_conventional"] = p0
    pred["p_augmented"] = p1
    pred.to_csv(OUT / f"predictions_{label}.csv", index=False)
    return {
        "label": label,
        "n": int(len(use)),
        "n_control": int(np.sum(y == 0)),
        "n_pd": int(np.sum(y == 1)),
        "C": float(C),
        "conventional_features": conventional,
        "tma_features": tma,
        "auc_conventional": a0,
        "auc_conventional_plus_tma": a1,
        "delta_auc": float(a1 - a0),
        "bootstrap": boot,
    }


def sensitivity_table(df, conventional, tma, label):
    rows = []
    for C in C_SENSITIVITY:
        needed = conventional + tma
        use = df.dropna(subset=needed).copy().reset_index(drop=True)
        y = (use["group"] == "pd").astype(int).to_numpy()
        p0 = loocv_probabilities(use[conventional].to_numpy(float), y, C)
        p1 = loocv_probabilities(use[conventional + tma].to_numpy(float), y, C)
        a0 = float(roc_auc_score(y, p0))
        a1 = float(roc_auc_score(y, p1))
        rows.append({
            "label": label,
            "C": C,
            "n": len(use),
            "auc_conventional": a0,
            "auc_augmented": a1,
            "delta_auc": a1 - a0,
        })
    return rows


def main():
    tma = pd.read_csv(TMA_FEATURE_PATH)
    speeds = load_speed_table()

    rows = []
    for i, (record, group) in enumerate(RECORDS, 1):
        print(f"[conventional] {i:02d}/{len(RECORDS)} {record}", flush=True)
        arr = download_record(record)
        cycles, fs, _ = extract_cycles(arr, record)
        feat = conventional_from_cycles(cycles, fs)
        sid = record.split("_")[0]
        rows.append({
            "record": record,
            "group": group,
            "speed_m_s": speeds.get(sid, np.nan),
            **feat,
        })

    conv = pd.DataFrame(rows).sort_values("record").reset_index(drop=True)
    conv.to_csv(OUT / "conventional_features.csv", index=False)

    merged = conv.merge(
        tma.drop(columns=["group"], errors="ignore"),
        on="record",
        how="inner",
        validate="one_to_one",
    )
    merged.to_csv(OUT / "analysis_matrix.csv", index=False)

    comparisons = [
        evaluate_pair(merged, CONVENTIONAL_CORE, TMA_FULL, "core_plus_full"),
        evaluate_pair(merged, CONVENTIONAL_EXPANDED, TMA_FULL, "expanded_plus_full"),
        evaluate_pair(merged, CONVENTIONAL_CORE, TMA_COMPACT, "core_plus_compact"),
        evaluate_pair(merged, CONVENTIONAL_EXPANDED, TMA_COMPACT, "expanded_plus_compact"),
    ]

    sens = []
    sens += sensitivity_table(merged, CONVENTIONAL_CORE, TMA_FULL, "core_plus_full")
    sens += sensitivity_table(merged, CONVENTIONAL_EXPANDED, TMA_FULL, "expanded_plus_full")
    sens += sensitivity_table(merged, CONVENTIONAL_CORE, TMA_COMPACT, "core_plus_compact")
    sens += sensitivity_table(merged, CONVENTIONAL_EXPANDED, TMA_COMPACT, "expanded_plus_compact")
    pd.DataFrame(sens).to_csv(OUT / "ridge_C_sensitivity.csv", index=False)

    by_label = {x["label"]: x for x in comparisons}
    primary = by_label["expanded_plus_full"]
    corroborating = by_label["core_plus_full"]
    if (
        primary["delta_auc"] > 0
        and corroborating["delta_auc"] > 0
        and primary["bootstrap"]["ci95_low"] > 0
        and corroborating["bootstrap"]["ci95_low"] > 0
    ):
        verdict = "PASS"
    elif primary["delta_auc"] <= 0 and corroborating["delta_auc"] <= 0:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "seed": SEED,
        "criterion": "I — incremental information beyond conventional gait variables",
        "verdict": verdict,
        "cohort_note": "Speed_01 is missing for GaPt03, so speed-containing models use 46 participants.",
        "model": {
            "type": "standardized L2/ridge logistic regression",
            "validation": "participant-level leave-one-out cross-validation",
            "primary_C": C_PRIMARY,
            "scaling": "fit within each training fold only",
            "bootstrap": f"paired subject bootstrap of fixed LOOCV predictions, B={BOOT_B}",
        },
        "comparisons": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Exact-engine incremental-information validation",
        "",
        f"Criterion I verdict: **{verdict}**",
        "",
        "All models use participant-level LOOCV with within-fold standardization and fixed ridge C=1.",
        "GaPt03 lacks Speed_01, leaving 46 participants for the speed-containing comparison.",
        "",
        "## Primary comparisons",
        "",
    ]
    for z in comparisons:
        b = z["bootstrap"]
        lines.append(
            f"- {z['label']}: conventional AUC={z['auc_conventional']:.3f}; "
            f"+TMA AUC={z['auc_conventional_plus_tma']:.3f}; "
            f"ΔAUC={z['delta_auc']:+.3f}; "
            f"bootstrap 95% CI {b['ci95_low']:+.3f} to {b['ci95_high']:+.3f}; "
            f"p(Δ<=0)={b['p_delta_le_0']:.5g}."
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "PASS requires positive ΔAUC with the full exact-engine TMA block for both the five-variable core conventional baseline and the broader seven-variable baseline, with both paired-bootstrap lower 95% bounds above zero.",
        "",
        "The historical 0.770→0.893 result is treated as context only because its exact conventional-feature formulas and ridge implementation were not preserved.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
