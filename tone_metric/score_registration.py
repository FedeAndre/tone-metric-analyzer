from __future__ import annotations

"""Register recursive tone-metric results to the canonical score layout.

v0.11.0 keeps the v0.10.0 attack registrar unchanged in principle: every normal
(non-parenthetical) level label must return to the exact canonical attack column
that generated that sonic event.  No visual approximation is permitted for attacks.

The new dissertation-faithful layer adds *structural articulations*: recursive
metric positions at which the tone-metric grid exists but no new sonic event
occurs.  These are displayed parenthetically.  Because a dot continuation or
empty metric position has no notehead to anchor to, its drawing x-position is
computed only *after* musical time is fixed, from the canonical measure layout.
This layout coordinate is presentation metadata and can never create, move, or
remove an attack or alter the Levels engine.
"""

from collections import defaultdict
from fractions import Fraction


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _event_targets(analysis_result: dict) -> list[dict]:
    """Return actual sonic events only, preserving the v0.10.0 attack path."""
    rows = []
    for si, segment in enumerate(analysis_result.get("segments", [])):
        for event in segment.get("events", []):
            levels = sorted({int(x) for x in event.get("tone_metric_levels", []) if int(x) > 0})
            if not levels:
                continue
            onset = _frac(event.get("onset_quarter"))
            if onset is None:
                continue
            rows.append({"segment_index": si, "event": event, "onset": onset, "levels": levels})
    return rows


def _analysis_measures(analysis_result: dict) -> list[dict]:
    out = []
    for i, row in enumerate(analysis_result.get("measures", [])):
        start = _frac(row.get("start_quarter"))
        end = _frac(row.get("end_quarter"))
        full = _frac(row.get("full_duration_quarter"))
        shift = _frac(row.get("pickup_shift_quarter"), Fraction(0))
        if start is None or end is None or full is None:
            continue
        out.append({
            "index": int(row.get("index", i)),
            "number": row.get("number", str(i + 1)),
            "start": start,
            "end": end,
            "full": full,
            "shift": shift,
        })
    return out


def _structural_targets(analysis_result: dict) -> list[dict]:
    """Return recursive metric articulations at which no new sonic event occurs.

    These are the parenthetical points described in the dissertation.  Segment-end
    boundary points are excluded because they belong to the following segment or to
    the end of the piece and would otherwise be duplicated.
    """
    event_times = {
        _frac(event.get("onset_quarter"))
        for segment in analysis_result.get("segments", [])
        for event in segment.get("events", [])
        if _frac(event.get("onset_quarter")) is not None
    }
    measures = _analysis_measures(analysis_result)
    rows = []
    seen = set()
    for si, segment in enumerate(analysis_result.get("segments", [])):
        seg_end = _frac(segment.get("segment_end_quarter"))
        for point in segment.get("structural_points", []):
            t = _frac(point.get("time_quarter"))
            if t is None or seg_end is None or t >= seg_end or t in event_times:
                continue
            levels = sorted({int(x) for x in point.get("levels", []) if int(x) > 0})
            if not levels:
                continue
            measure = next((m for m in measures if m["start"] <= t < m["end"]), None)
            if measure is None:
                continue
            offset = t - measure["start"]
            key = (measure["index"], offset, tuple(levels))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "segment_index": si,
                "time": t,
                "levels": levels,
                "measure": measure,
                "offset": offset,
            })
    return rows


def _canonical_measure_map(meta: dict) -> dict[int, dict]:
    return {
        int(m.get("global_measure_index", -1)): m
        for m in meta.get("measures", [])
        if int(m.get("global_measure_index", -1)) >= 0
    }


def _structural_reason(measure_meta: dict, local_time: Fraction, exact_col: dict | None) -> str:
    if exact_col is not None:
        if exact_col.get("tie_continuation_only"):
            return "tie-continuation"
        if exact_col.get("rest") and not exact_col.get("attack"):
            return "rest"
        return "recognized-no-attack-column"

    # A parenthetical point can fall inside a longer sounding object without a
    # separately printed notehead there.  Identify the notation semantics when the
    # saved OMR contains enough duration information to do so.
    reasons = []
    for col in measure_meta.get("columns", []):
        start = _frac(col.get("onset_quarter"))
        if start is None or not (start < local_time):
            continue
        for event in col.get("notation_events", []) or []:
            dur = _frac(event.get("duration_quarter"), Fraction(0))
            if dur is None or dur <= 0 or not (local_time < start + dur):
                continue
            if event.get("rest"):
                reasons.append("rest-continuation")
            elif int(event.get("dot_count", 0) or 0) > 0:
                reasons.append("dotted-note-continuation")
            elif event.get("tie_continuation"):
                reasons.append("tie-continuation")
            elif event.get("kind") == "head":
                reasons.append("sustained-note")
    priority = (
        "dotted-note-continuation",
        "tie-continuation",
        "rest-continuation",
        "sustained-note",
    )
    for reason in priority:
        if reason in reasons:
            return reason
    return "metric-position-without-new-attack"


def _structural_layout_position(measure_meta: dict, local_time: Fraction) -> tuple[float | None, str, float, dict | None]:
    """Locate an already-known structural score time on the printed measure.

    Exact non-attack notation columns (rests/tie continuations) are preferred.  A
    dot continuation normally has no glyph at the continuation point, so its visual
    x is derived piecewise from surrounding canonical score-time columns.  This is
    intentionally a one-way *rendering* transform: it receives a fixed score time
    and returns an x-coordinate; it never feeds back into musical timing.
    """
    columns = []
    exact_col = None
    for col in measure_meta.get("columns", []):
        t = _frac(col.get("onset_quarter"))
        if t is None:
            continue
        x = float(col.get("x_abs", 0.0))
        columns.append((t, x, col))
        if t == local_time:
            exact_col = col
    columns.sort(key=lambda row: (row[0], row[1]))

    if exact_col is not None:
        return float(exact_col["x_abs"]), "canonical-no-attack-column-exact", 0.96, exact_col

    left = float(measure_meta.get("stack_left", 0.0) or 0.0)
    right = float(measure_meta.get("stack_right", left + 1.0) or (left + 1.0))
    full = _frac(measure_meta.get("full_duration_quarter"), Fraction(0))
    if full is None or full <= 0 or right <= left:
        return None, "structural-layout-unavailable", 0.0, None

    # Between two canonical columns: the strongest layout estimate for a position
    # that has no physical glyph of its own (most commonly a dotted continuation).
    for (ta, xa, _ca), (tb, xb, _cb) in zip(columns, columns[1:]):
        if ta < local_time < tb and tb > ta and xb > xa:
            ratio = float((local_time - ta) / (tb - ta))
            x = xa + ratio * (xb - xa)
            return x, "structural-score-time-between-canonical-columns", 0.86, None

    # Outside the first/last printed event, use the local score-spacing slope when
    # two columns are available.  Clamp inside the measure stack; the coordinate is
    # still visualization-only and is explicitly marked as estimated.
    if len(columns) >= 2:
        if local_time < columns[0][0]:
            (ta, xa, _), (tb, xb, _) = columns[0], columns[1]
        else:
            (ta, xa, _), (tb, xb, _) = columns[-2], columns[-1]
        if tb > ta and xb > xa:
            slope = (xb - xa) / float(tb - ta)
            x = xa + float(local_time - ta) * slope
            x = max(left + 1.0, min(right - 1.0, x))
            return x, "structural-score-time-local-extrapolation", 0.72, None

    # A completely silent/under-recognized measure can contain no usable notation
    # columns.  The metric time is already fixed, so proportional placement within
    # Audiveris' measured stack is safe as a display-only last resort.
    ratio = max(0.0, min(1.0, float(local_time / full)))
    x = left + ratio * (right - left)
    source = "structural-score-time-measure-layout" if not columns else "structural-score-time-single-column-layout"
    return x, source, 0.58 if not columns else 0.64, None


def build_layer_anchors_from_canonical_score(analysis_result: dict, page_dimensions: dict[int, tuple[float, float]]):
    """Build exact attack anchors plus dissertation-style parenthetical anchors."""
    meta = analysis_result.get("canonical_score_meta") or {}
    measures = analysis_result.get("measures", [])
    by_key = {}
    measure_shift = {
        int(m.get("index", i)): _frac(m.get("pickup_shift_quarter"), Fraction(0))
        for i, m in enumerate(measures)
    }
    for measure in meta.get("measures", []):
        mi = int(measure.get("global_measure_index", -1))
        shift = measure_shift.get(mi, Fraction(0))
        for col in measure.get("columns", []):
            if not col.get("attack"):
                continue
            onset_local = _frac(col.get("onset_quarter"))
            if onset_local is not None:
                by_key[(mi, shift + onset_local)] = col

    anchors_by_page = defaultdict(list)
    per_level = defaultdict(lambda: {"expected": 0, "mapped": 0, "missing": 0})
    missing = defaultdict(list)
    manufactured_attack_x = 0

    # --- v0.10.0 exact attack-registration path (kept semantically unchanged). ---
    for target in _event_targets(analysis_result):
        event = target["event"]
        mi = int(event.get("measure_index", -1))
        offset = _frac(event.get("offset_in_measure_quarter"))
        for level in target["levels"]:
            per_level[level]["expected"] += 1
        col = by_key.get((mi, offset))
        if col is None:
            for level in target["levels"]:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": "canonical-attack-column-not-found",
                })
            continue

        page = int(col["page_index"])
        system = int(col["system_index"])
        dims = page_dimensions.get(page)
        if not dims or float(dims[0]) <= 0:
            for level in target["levels"]:
                per_level[level]["missing"] += 1
                missing[level].append({"measure_index": mi, "reason": "page-dimensions-unavailable"})
            continue
        pw, ph = float(dims[0]), float(dims[1])
        x_abs = float(col["x_abs"])
        y_abs = float(col["y_abs"])
        row = {
            "page_index": page,
            "system_index": system,
            "physical_system_index": system,
            "segment_index": int(target["segment_index"]),
            "event_index": int(event.get("event_index", -1)),
            "measure_index": mi,
            "measure_number": event.get("measure_number", str(mi + 1)),
            "onset_quarter": event.get("onset_quarter"),
            "offset_in_measure_quarter": event.get("offset_in_measure_quarter"),
            "attack_key": event.get("attack_key", f"{mi}:{offset}"),
            "duration_quarter": event.get("duration_quarter", ""),
            "cx_norm": x_abs / pw,
            "cy_norm": y_abs / ph if ph > 0 else 0.0,
            "height": max(target["levels"]),
            "lowest_level": min(target["levels"]),
            "levels": target["levels"],
            "parenthetical": False,
            "registration_source": "canonical-score-event-exact-origin-column",
            "timing_source": col.get("timing_source"),
            "timing_confidence": col.get("confidence"),
            "recovered_chord_ids": list(col.get("chord_ids", [])),
            "recovered_x_abs": x_abs,
        }
        anchors_by_page[page].append(row)
        for level in target["levels"]:
            per_level[level]["mapped"] += 1

    # --- v0.11.0 structural no-attack display path. ---
    structural_by_page = defaultdict(list)
    structural_per_level = defaultdict(lambda: {"expected": 0, "mapped": 0, "missing": 0})
    structural_missing = defaultdict(list)
    measure_meta = _canonical_measure_map(meta)
    structural_source_counts = defaultdict(int)
    structural_reason_counts = defaultdict(int)

    for target in _structural_targets(analysis_result):
        m = target["measure"]
        mi = int(m["index"])
        offset = target["offset"]
        local_time = offset - m["shift"]
        for level in target["levels"]:
            structural_per_level[level]["expected"] += 1

        cm = measure_meta.get(mi)
        if cm is None or local_time < 0 or local_time >= m["full"]:
            for level in target["levels"]:
                structural_per_level[level]["missing"] += 1
                structural_missing[level].append({
                    "measure_index": mi,
                    "measure_number": m["number"],
                    "offset_in_measure_quarter": str(offset),
                    "reason": "canonical-measure-layout-not-found" if cm is None else "structural-time-outside-printable-measure",
                })
            continue

        page = int(cm.get("page_index", -1))
        system = int(cm.get("system_index", -1))
        dims = page_dimensions.get(page)
        if not dims or float(dims[0]) <= 0:
            for level in target["levels"]:
                structural_per_level[level]["missing"] += 1
                structural_missing[level].append({
                    "measure_index": mi,
                    "measure_number": m["number"],
                    "offset_in_measure_quarter": str(offset),
                    "reason": "page-dimensions-unavailable",
                })
            continue

        x_abs, source, confidence, exact_col = _structural_layout_position(cm, local_time)
        if x_abs is None:
            for level in target["levels"]:
                structural_per_level[level]["missing"] += 1
                structural_missing[level].append({
                    "measure_index": mi,
                    "measure_number": m["number"],
                    "offset_in_measure_quarter": str(offset),
                    "reason": source,
                })
            continue

        reason = _structural_reason(cm, local_time, exact_col)
        pw, ph = float(dims[0]), float(dims[1])
        y_abs = float(exact_col.get("y_abs", 0.0)) if exact_col is not None else 0.0
        row = {
            "page_index": page,
            "system_index": system,
            "physical_system_index": system,
            "segment_index": int(target["segment_index"]),
            "event_index": -1,
            "measure_index": mi,
            "measure_number": m["number"],
            "onset_quarter": str(target["time"]),
            "offset_in_measure_quarter": str(offset),
            "cx_norm": float(x_abs) / pw,
            "cy_norm": y_abs / ph if ph > 0 else 0.0,
            "height": max(target["levels"]),
            "lowest_level": min(target["levels"]),
            "levels": target["levels"],
            "parenthetical": True,
            "structural_reason": reason,
            "registration_source": source,
            "timing_source": "recursive-structural-grid",
            "timing_confidence": float(confidence),
            "recovered_chord_ids": list(exact_col.get("chord_ids", [])) if exact_col is not None else [],
            "recovered_x_abs": float(x_abs),
            "visual_position_is_estimated": source != "canonical-no-attack-column-exact",
        }
        structural_by_page[page].append(row)
        structural_source_counts[source] += 1
        structural_reason_counts[reason] += 1
        for level in target["levels"]:
            structural_per_level[level]["mapped"] += 1

    for rows in anchors_by_page.values():
        rows.sort(key=lambda a: (
            int(a.get("system_index", 0)),
            int(a.get("measure_index", -1)),
            _frac(a.get("offset_in_measure_quarter"), Fraction(0)),
        ))
    for rows in structural_by_page.values():
        rows.sort(key=lambda a: (
            int(a.get("system_index", 0)),
            int(a.get("measure_index", -1)),
            _frac(a.get("offset_in_measure_quarter"), Fraction(0)),
        ))

    warnings = list(meta.get("warnings", []))
    missing_total = sum(v["missing"] for v in per_level.values())
    if missing_total:
        warnings.append(f"{missing_total} event-level anchor(s) could not be returned to their originating canonical attack column.")
    structural_missing_total = sum(v["missing"] for v in structural_per_level.values())
    if structural_missing_total:
        warnings.append(f"{structural_missing_total} parenthetical structural level label(s) could not be placed on the canonical measure layout.")

    attack_levels = sorted(per_level)
    structural_levels = sorted(structural_per_level)
    all_levels = sorted(set(attack_levels) | set(structural_levels))
    stats = {
        "final_layer_anchors": sum(len(v) for v in anchors_by_page.values()),
        "layer_anchor_registration": "canonical-score-event-exact-origin-column",
        "levels_enabled": all_levels,
        "max_layer_level": max(all_levels, default=0),
        "layer_hits_expected": sum(v["expected"] for v in per_level.values()),
        "layer_hits_mapped": sum(v["mapped"] for v in per_level.values()),
        "layer_hits_without_visual_attack": missing_total,
        "canonical_column_count": int(meta.get("column_count", 0) or 0),
        "canonical_attack_column_count": int(meta.get("attack_column_count", 0) or 0),
        "manufactured_x_coordinates": manufactured_attack_x,
        "per_level": {str(k): dict(v) for k, v in sorted(per_level.items())},
        "missing_targets_by_level": {str(k): v for k, v in sorted(missing.items())},
        "structural_parenthetical_positions": sum(len(v) for v in structural_by_page.values()),
        "structural_parenthetical_labels_expected": sum(v["expected"] for v in structural_per_level.values()),
        "structural_parenthetical_labels_mapped": sum(v["mapped"] for v in structural_per_level.values()),
        "structural_parenthetical_labels_missing": structural_missing_total,
        "structural_per_level": {str(k): dict(v) for k, v in sorted(structural_per_level.items())},
        "structural_missing_targets_by_level": {str(k): v for k, v in sorted(structural_missing.items())},
        "structural_registration_sources": dict(sorted(structural_source_counts.items())),
        "structural_reasons": dict(sorted(structural_reason_counts.items())),
    }
    for level, values in per_level.items():
        stats[f"level{level}_anchors_mapped"] = values["mapped"]
        stats[f"level{level}_hits_expected"] = values["expected"]
        stats[f"level{level}_hits_without_visual_attack"] = values["missing"]
    return dict(anchors_by_page), dict(structural_by_page), stats, warnings
