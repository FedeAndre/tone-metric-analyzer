#!/usr/bin/env python3
from __future__ import annotations

import argparse, io, json, math, os, re, tempfile, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.tma_clinical_exact_utils import analyze_event_frames, bh_adjust, group_contrast
from research.tma_mri_exact_utils import S3, BUCKET, math_audit

DS = "ds001907"\n# Frozen task-fMRI validation branch; push triggers the analysis workflow.
OUT = Path("research/tma_taskfmri_ds001907_results")
TMA_CORE = [
    "mean_H","sd_H","mean_D","sd_D","mean_lambda","sd_lambda","multilevel",
    "pivot_rate","compound_pivot_fraction","pivot_depth",
    "tree_span_events","tree_span_time","tree_root_fraction",
]
SURR_BASE = ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","tree_span_events","tree_span_time"]
CONV = [
    "rt_mean","rt_sd","accuracy","target_interval_mean","target_interval_cv",
    "peak_rate_per_min","coactivation_mean","coactivation_sd",
    "global_sd","global_lag1","spectral_entropy","dvars_pct","tsnr",
]

def list_ant_bold(run: int):
    pat = re.compile(rf"/ses-[12]/func/.*_task-ANT_run-{run}_bold\.nii\.gz$")
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": f"{DS}/"}
        if token:
            kw["ContinuationToken"] = token
        r = S3.list_objects_v2(**kw)
        for x in r.get("Contents", []):
            k = x["Key"]
            if pat.search(k):
                out.append({"key": k, "size": int(x["Size"])})
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    return sorted(out, key=lambda x: x["key"])

def download_temp(key: str):
    fd, p = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    S3.download_file(BUCKET, key, p)
    return Path(p)

def read_events_for_bold(key: str):
    ekey = key.replace("_bold.nii.gz", "_events.tsv")
    b = S3.get_object(Bucket=BUCKET, Key=ekey)["Body"].read()
    df = pd.read_csv(io.BytesIO(b), sep="\t")
    df.columns = [str(x).strip() for x in df.columns]
    return df

def _sample_indices(mask, maxn=3000):
    idx = np.flatnonzero(mask.ravel())
    if len(idx) <= maxn:
        return idx
    pos = np.linspace(0, len(idx)-1, maxn).round().astype(int)
    return idx[pos]

def _quadratic_peak_time(y, i, tr):
    if i <= 0 or i >= len(y)-1:
        return float(i*tr)
    y0, y1, y2 = float(y[i-1]), float(y[i]), float(y[i+1])
    den = y0 - 2*y1 + y2
    if abs(den) < 1e-12:
        d = 0.0
    else:
        d = 0.5 * (y0-y2) / den
        d = float(np.clip(d, -0.5, 0.5))
    return float((i+d)*tr)

def _build_frames(targets, peak_times):
    frames = []
    p = np.asarray(sorted(peak_times), float)
    for a, b in zip(targets[:-1], targets[1:]):
        dur = float(b-a)
        if not (2.0 <= dur <= 25.0):
            continue
        use = p[(p >= a) & (p < b)]
        events = [{"label":"BOLD","offset_s":float(t-a)} for t in use]
        frames.append({"duration_s":dur, "events":events})
    return frames

def _safe_analyze(frames, tol):
    if sum(len(fr["events"]) for fr in frames) < 8:
        raise RuntimeError("too few task-locked BOLD events")
    return analyze_event_frames(frames, tol_s=tol, label_features=False)

def _surrogate_z(targets, peak_times, observed, tol, run_duration, key, n_surr=20):
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vals = {c: [] for c in SURR_BASE}
    p = np.asarray(peak_times, float)
    if run_duration <= 10 or len(p) < 8:
        return {f"surz_{c}": np.nan for c in SURR_BASE}
    for _ in range(n_surr):
        shift = float(rng.uniform(5.0, max(5.01, run_duration-5.0)))
        q = np.mod(p + shift, run_duration)
        fr = _build_frames(targets, q)
        try:
            feat, _, _, _ = _safe_analyze(fr, tol)
        except Exception:
            continue
        for c in SURR_BASE:
            if c in feat and np.isfinite(feat[c]):
                vals[c].append(float(feat[c]))
    out = {}
    for c in SURR_BASE:
        x = np.asarray(vals[c], float)
        if len(x) >= 5 and np.std(x, ddof=1) > 1e-12:
            out[f"surz_{c}"] = float((observed[c]-np.mean(x))/np.std(x, ddof=1))
        else:
            out[f"surz_{c}"] = np.nan
    return out

def preprocess_task_bold(path: Path, events: pd.DataFrame, key: str):
    img = nib.load(str(path))
    if len(img.shape) != 4 or img.shape[3] < 80:
        raise RuntimeError(f"unexpected BOLD shape {img.shape}")
    tr = float(img.header.get_zooms()[3])
    if not (0.2 < tr < 5.0):
        raise RuntimeError(f"invalid TR {tr}")
    data = img.get_fdata(dtype=np.float32)
    T = data.shape[-1]
    run_duration = T * tr

    mean = np.nanmean(data[..., ::max(1, T//30)], axis=-1)
    pos = mean[np.isfinite(mean) & (mean > 0)]
    if len(pos) < 1000:
        raise RuntimeError("insufficient positive voxels")
    thr = 0.20 * np.nanpercentile(pos, 98)
    mask = np.isfinite(mean) & (mean > thr)
    if mask.sum() < 3000:
        raise RuntimeError(f"small brain mask {int(mask.sum())}")

    coords = np.argwhere(mask)
    mids = np.median(coords, axis=0)
    flat = data.reshape((-1, T))
    grid = np.indices(mask.shape)
    sectors = []
    for bx in (0,1):
        for by in (0,1):
            for bz in (0,1):
                m = mask.copy()
                m &= (grid[0] > mids[0]) if bx else (grid[0] <= mids[0])
                m &= (grid[1] > mids[1]) if by else (grid[1] <= mids[1])
                m &= (grid[2] > mids[2]) if bz else (grid[2] <= mids[2])
                idx = _sample_indices(m, 2500)
                if len(idx) >= 100:
                    sectors.append(np.nanmean(flat[idx, :], axis=0))
    if len(sectors) < 6:
        raise RuntimeError(f"only {len(sectors)} brain sectors")
    sec = np.asarray(sectors, float)
    sec = signal.detrend(sec, axis=-1, type="linear")
    sec = (sec - sec.mean(axis=1, keepdims=True)) / (sec.std(axis=1, keepdims=True) + 1e-9)
    coact = np.sqrt(np.mean(sec**2, axis=0))
    global_sig = np.mean(sec, axis=0)

    dist = max(1, int(round(3.0/tr)))
    prom = max(0.15*np.std(coact), 1e-6)
    peaks, _ = signal.find_peaks(coact, distance=dist, prominence=prom)
    if len(peaks) < 25:
        peaks, _ = signal.find_peaks(coact, distance=dist, prominence=max(0.07*np.std(coact),1e-6))
    if len(peaks) < 20:
        peaks, _ = signal.find_peaks(coact, distance=dist)
    if len(peaks) < 15:
        raise RuntimeError(f"only {len(peaks)} coactivation peaks")
    peak_times = np.array([_quadratic_peak_time(coact, int(i), tr) for i in peaks], float)

    if "trial_type" not in events.columns:
        raise RuntimeError(f"events columns missing trial_type: {list(events.columns)}")
    tt = events["trial_type"].astype(str).str.strip()
    target_rows = events[tt.isin(["Congruent","Incongruent"])].copy()
    targets = pd.to_numeric(target_rows["onset"], errors="coerce").dropna().to_numpy(float)
    targets = np.sort(targets[(targets >= 0) & (targets < run_duration)])
    if len(targets) < 25:
        raise RuntimeError(f"only {len(targets)} target trials")
    intervals = np.diff(targets)
    valid = (intervals >= 2.0) & (intervals <= 25.0)
    if valid.sum() < 20:
        raise RuntimeError("insufficient valid inter-target intervals")

    frames = _build_frames(targets, peak_times)
    tol = max(0.25, tr/2.0)
    feat, evdf, piv, tree = _safe_analyze(frames, tol)
    surr = _surrogate_z(targets, peak_times, feat, tol, run_duration, key)

    # Behavioral columns differ slightly across dataset revisions; use them when present.
    rtv = pd.to_numeric(target_rows["response_time"], errors="coerce").dropna().to_numpy(float) if "response_time" in target_rows.columns else np.array([],float)
    # Dataset response_time is in milliseconds when present.
    acc = pd.to_numeric(target_rows["accuracy"], errors="coerce").dropna().to_numpy(float) if "accuracy" in target_rows.columns else np.array([],float)
    iti = np.diff(targets)
    iti = iti[(iti >= 2.0) & (iti <= 25.0)]

    idx = _sample_indices(mask, 3500)
    vox = flat[idx, :].astype(float)
    vm = np.mean(vox, axis=1, keepdims=True)
    rel = (vox-vm)/(np.abs(vm)+1e-9)*100.0
    dvars = float(np.median(np.sqrt(np.mean(np.diff(rel, axis=1)**2, axis=0))))
    tsnr = float(np.median(np.abs(vm[:,0])/(np.std(vox, axis=1, ddof=1)+1e-9)))

    f, powr = signal.welch(global_sig, fs=1/tr, nperseg=min(128, len(global_sig)))
    band = (f >= .01) & (f <= min(.15, f.max()))
    pb = powr[band]
    pn = pb/pb.sum() if pb.sum() > 0 else np.ones_like(pb)/max(1,len(pb))
    sent = float(-np.sum(pn*np.log(pn+1e-15))/np.log(len(pn))) if len(pn)>1 else np.nan

    conv = {
        "tr": tr, "n_volumes": T, "n_targets": int(len(targets)), "n_peaks": int(len(peaks)),
        "rt_mean": float(np.mean(rtv)) if len(rtv) else np.nan,
        "rt_sd": float(np.std(rtv, ddof=1)) if len(rtv)>1 else np.nan,
        "accuracy": float(np.mean(acc)) if len(acc) else np.nan,
        "target_interval_mean": float(np.mean(iti)),
        "target_interval_cv": float(np.std(iti, ddof=1)/np.mean(iti)) if len(iti)>1 else np.nan,
        "peak_rate_per_min": float(len(peaks)/(run_duration/60.0)),
        "coactivation_mean": float(np.mean(coact)),
        "coactivation_sd": float(np.std(coact, ddof=1)),
        "global_sd": float(np.std(global_sig, ddof=1)),
        "global_lag1": float(np.corrcoef(global_sig[:-1], global_sig[1:])[0,1]),
        "spectral_entropy": sent, "dvars_pct": dvars, "tsnr": tsnr,
    }
    example = {
        "first_target_onsets_s": targets[:8].tolist(),
        "first_peak_times_s": peak_times[:12].tolist(),
        "first_nonempty_frames": [fr for fr in frames if fr["events"]][:3],
        "first_events": evdf.head(12).to_dict(orient="records"),
    }
    return {**conv, **feat, **surr}, example

def parse_ids(key):
    sm = re.search(r"/(sub-[^/]+)/", key)
    se = re.search(r"/(ses-[^/]+)/", key)
    ru = re.search(r"_run-(\d+)_", key)
    subject = sm.group(1) if sm else None
    session = se.group(1) if se else None
    run = int(ru.group(1)) if ru else None
    group = "Control" if subject and subject.startswith("sub-RC41") else ("PD" if subject and subject.startswith("sub-RC42") else None)
    return subject, session, run, group

def run_one(run: int):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list_ant_bold(run)
    rows, fails, worked = [], {}, {}
    for i, item in enumerate(keys, 1):
        key = item["key"]
        print(f"[{i}/{len(keys)}] {key}", flush=True)
        p = None
        try:
            p = download_temp(key)
            ev = read_events_for_bold(key)
            feat, ex = preprocess_task_bold(p, ev, key)
            subject, session, run_no, group = parse_ids(key)
            rows.append({"key":key,"subject":subject,"session":session,"run":run_no,"group":group,**feat})
            if len(worked) < 4:
                worked[key] = ex
        except Exception as e:
            fails[key] = repr(e)
            print("FAIL", key, repr(e), flush=True)
        finally:
            if p and p.exists():
                p.unlink()
    tag = f"run-{run:02d}"
    pd.DataFrame(rows).to_csv(OUT/f"{tag}_scan_features.csv", index=False)
    (OUT/f"{tag}_failures.json").write_text(json.dumps(fails, indent=2, default=str))
    (OUT/f"{tag}_worked_examples.json").write_text(json.dumps(worked, indent=2, default=str))
    print(json.dumps({"run":run,"n_scans":len(rows),"failures":len(fails)}, indent=2))

def loocv_auc(df, features):
    q = df[["group"]+features].replace([np.inf,-np.inf],np.nan).dropna().copy()
    q["y"] = (q["group"]=="PD").astype(int)
    pred = np.full(len(q), np.nan)
    for i in range(len(q)):
        tr = np.arange(len(q)) != i
        if q.iloc[tr].y.nunique() < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000, class_weight="balanced"))
        m.fit(q.iloc[tr][features], q.iloc[tr].y)
        pred[i] = m.predict_proba(q.iloc[[i]][features])[0,1]
    ok = np.isfinite(pred)
    if ok.sum() < 10:
        return np.nan, int(ok.sum())
    return float(roc_auc_score(q.loc[ok,"y"], pred[ok])), int(ok.sum())

def aggregate():
    files = sorted(OUT.glob("run-*_scan_features.csv"))
    if len(files) < 6:
        raise RuntimeError(f"expected 6 run files, found {len(files)}")
    scan = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    scan.to_csv(OUT/"all_scan_features.csv", index=False)

    nums = [c for c in scan.select_dtypes(include=[np.number]).columns if c not in ["run"]]
    subj = scan.groupby(["subject","session","group"], as_index=False)[nums].mean()
    subj.to_csv(OUT/"subject_session_features.csv", index=False)

    tma = [c for c in TMA_CORE if c in subj and subj[c].notna().all() and np.nanstd(subj[c])>1e-12]
    surz = [f"surz_{c}" for c in SURR_BASE if f"surz_{c}" in subj and subj[f"surz_{c}"].notna().all() and np.nanstd(subj[f"surz_{c}"])>1e-12]
    tested = tma + surz

    contrasts = {}
    tables = {}
    for ses in ["ses-1","ses-2"]:
        z = subj[subj.session==ses].copy()
        tab = group_contrast(z, "group", "Control", "PD", tested)
        tab.to_csv(OUT/f"{ses}_PD_vs_Control.csv", index=False)
        tables[ses] = tab
        contrasts[ses] = tab.head(12).to_dict(orient="records")

    a = tables["ses-1"][["feature","hedges_g","welch_p","q_BH"]].rename(columns={"hedges_g":"g_ses1","welch_p":"p_ses1","q_BH":"q_ses1"})
    b = tables["ses-2"][["feature","hedges_g","welch_p","q_BH"]].rename(columns={"hedges_g":"g_ses2","welch_p":"p_ses2","q_BH":"q_ses2"})
    rep = a.merge(b,on="feature",how="inner")
    rep["same_direction"] = np.sign(rep.g_ses1)==np.sign(rep.g_ses2)
    rep["nominal_both"] = rep.same_direction & (rep.p_ses1<0.05) & (rep.p_ses2<0.05)
    rep["fdr_both"] = rep.same_direction & (rep.q_ses1<0.05) & (rep.q_ses2<0.05)
    rep["max_p"] = rep[["p_ses1","p_ses2"]].max(axis=1)
    rep = rep.sort_values(["fdr_both","nominal_both","max_p"], ascending=[False,False,True])
    rep.to_csv(OUT/"session_replication.csv", index=False)

    if len(rep) >= 3:
        pr = stats.pearsonr(rep.g_ses1, rep.g_ses2)
        sr = stats.spearmanr(rep.g_ses1, rep.g_ses2)
        same = int(rep.same_direction.sum())
        sign_p = float(stats.binomtest(same, len(rep), 0.5).pvalue)
        effect_rep = {"n_features":int(len(rep)),"pearson_r":float(pr.statistic),"pearson_p":float(pr.pvalue),
                      "spearman_rho":float(sr.statistic),"spearman_p":float(sr.pvalue),
                      "same_direction":same,"same_direction_fraction":float(same/len(rep)),"sign_test_p":sign_p}
    else:
        effect_rep = {}

    auc = {}
    compact_tma = [c for c in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction","pivot_depth","tree_span_events","tree_span_time"] if c in subj]
    base = [c for c in CONV if c in subj and subj[c].notna().all() and np.nanstd(subj[c])>1e-12]
    for ses in ["ses-1","ses-2"]:
        z=subj[subj.session==ses]
        a0,n0=loocv_auc(z,base)
        a1,n1=loocv_auc(z,base+compact_tma)
        auc[ses]={"n":n0,"auc_conventional":a0,"auc_plus_tma":a1,"delta_auc":float(a1-a0) if np.isfinite(a0) and np.isfinite(a1) else np.nan}

    # Task-locking itself: observed TMA vs circular-shift surrogates.
    lock = {}
    for ses in ["ses-1","ses-2"]:
        z=subj[subj.session==ses]
        vals={}
        for c in surz:
            x=z[c].dropna().to_numpy(float)
            if len(x)>=10:
                tt=stats.ttest_1samp(x,0.0)
                vals[c]={"n":int(len(x)),"mean_z":float(np.mean(x)),"p_vs_shift_null":float(tt.pvalue)}
        if vals:
            ps=np.array([v["p_vs_shift_null"] for v in vals.values()])
            qs=bh_adjust(ps)
            for qv,(c,v) in zip(qs,vals.items()):
                v["q_BH"]=float(qv)
        lock[ses]=vals

    summary = {
        "dataset":DS,
        "description":"Task-locked exact TMA of whole-brain fMRI coactivation peaks during the Attention Network Test",
        "analysis_unit":"Subject-session means across six ANT runs",
        "groups":{"Control":int((subj[subj.session=="ses-1"].group=="Control").sum()),"PD":int((subj[subj.session=="ses-1"].group=="PD").sum())},
        "n_scans":int(len(scan)),"n_subject_session_rows":int(len(subj)),
        "math_audit":math_audit(),
        "event_definition":"Whole-brain coactivation peaks positioned inside consecutive target-to-target task intervals; exact TMA engine applied after interval normalization.",
        "surrogate":"20 deterministic circular shifts of BOLD peak times relative to the unchanged task schedule per scan.",
        "top_contrasts":contrasts,
        "effect_replication":effect_rep,
        "replicated_nominal":rep[rep.nominal_both].head(20).to_dict(orient="records"),
        "replicated_fdr":rep[rep.fdr_both].head(20).to_dict(orient="records"),
        "incremental_classification":auc,
        "task_locking":lock,
        "failures_total":int(sum(len(json.loads(p.read_text())) for p in OUT.glob("run-*_failures.json"))),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run",type=int,choices=[1,2,3,4,5,6])
    g.add_argument("--aggregate",action="store_true")
    args=ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_one(args.run)

if __name__=="__main__":
    main()
