from __future__ import annotations

"""Register Tone-Metric attacks to the canonical score layout.

Only actual sonic attacks may receive analytical Level labels.  Every label is
registered to an attacking notehead belonging to the exact canonical score-time
column that generated the event.  Rests, sustained notes, augmentation-dot
continuations, ties, accidentals, clefs, and barlines can never supply an x-anchor.
"""

from collections import defaultdict
from fractions import Fraction
from statistics import median


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _event_targets(analysis_result: dict) -> list[dict]:
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


def _attack_head_position(col: dict) -> tuple[float, float] | None:
    """Return the physical center of the attacking notehead(s) in one exact column.

    Canonical columns can also contain rests or continuation material.  We refuse to
    fall back to a generic column/chord box because that box can visually include an
    accidental, stem, rest, or nearby barline.  If no attacking head geometry is
    available, the attack is left unmapped rather than guessed.
    """
    heads = [
        e for e in (col.get("notation_events") or [])
        if e.get("kind") == "head" and bool(e.get("attack"))
    ]
    xs = []
    ys = []
    for head in heads:
        try:
            xs.append(float(head.get("x_abs")))
            ys.append(float(head.get("y_abs")))
        except Exception:
            continue
    if not xs or not ys:
        return None
    return float(median(xs)), float(median(ys))


def build_layer_anchors_from_canonical_score(
    analysis_result: dict,
    page_dimensions: dict[int, tuple[float, float]],
):
    """Build attack-only PDF anchors from exact canonical score-time columns."""
    meta = analysis_result.get("canonical_score_meta") or {}
    measures = analysis_result.get("measures", [])
    measure_shift = {
        int(m.get("index", i)): _frac(m.get("pickup_shift_quarter"), Fraction(0))
        for i, m in enumerate(measures)
    }

    by_key = {}
    for measure in meta.get("measures", []):
        mi = int(measure.get("global_measure_index", -1))
        shift = measure_shift.get(mi, Fraction(0))
        for col in measure.get("columns", []):
            if not col.get("attack"):
                continue
            onset_local = _frac(col.get("onset_quarter"))
            if onset_local is None:
                continue
            by_key[(mi, shift + onset_local)] = col

    anchors_by_page = defaultdict(list)
    per_level = defaultdict(lambda: {"expected": 0, "mapped": 0, "missing": 0})
    missing = defaultdict(list)
    missing_notehead_geometry = 0

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

        head_pos = _attack_head_position(col)
        if head_pos is None:
            missing_notehead_geometry += 1
            for level in target["levels"]:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": "attacking-notehead-geometry-not-found",
                })
            continue

        page = int(col.get("page_index", -1))
        system = int(col.get("system_index", -1))
        dims = page_dimensions.get(page)
        if not dims or float(dims[0]) <= 0:
            for level in target["levels"]:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": "page-dimensions-unavailable",
                })
            continue

        pw, ph = float(dims[0]), float(dims[1])
        x_abs, y_abs = head_pos
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
            "registration_source": "canonical-score-exact-attacking-notehead-center",
            "timing_source": col.get("timing_source"),
            "timing_confidence": col.get("confidence"),
            "recovered_chord_ids": list(col.get("chord_ids", [])),
            "recovered_x_abs": x_abs,
            "visual_position_is_estimated": False,
        }
        anchors_by_page[page].append(row)
        for level in target["levels"]:
            per_level[level]["mapped"] += 1

    for rows in anchors_by_page.values():
        rows.sort(key=lambda a: (
            int(a.get("system_index", 0)),
            int(a.get("measure_index", -1)),
            _frac(a.get("offset_in_measure_quarter"), Fraction(0)),
        ))

    warnings = list(meta.get("warnings", []))
    missing_total = sum(v["missing"] for v in per_level.values())
    if missing_total:
        warnings.append(
            f"{missing_total} event-level anchor(s) could not be returned to an exact attacking notehead."
        )
    if missing_notehead_geometry:
        warnings.append(
            f"{missing_notehead_geometry} attack column(s) lacked explicit attacking-notehead geometry and were deliberately not guessed."
        )

    attack_levels = sorted(per_level)
    stats = {
        "final_layer_anchors": sum(len(v) for v in anchors_by_page.values()),
        "layer_anchor_registration": "canonical-score-exact-attacking-notehead-center",
        "analysis_position_policy": "actual-attacks-only",
        "levels_enabled": attack_levels,
        "max_layer_level": max(attack_levels, default=0),
        "layer_hits_expected": sum(v["expected"] for v in per_level.values()),
        "layer_hits_mapped": sum(v["mapped"] for v in per_level.values()),
        "layer_hits_without_visual_attack": missing_total,
        "canonical_column_count": int(meta.get("column_count", 0) or 0),
        "canonical_attack_column_count": int(meta.get("attack_column_count", 0) or 0),
        "manufactured_x_coordinates": 0,
        "attacking_notehead_geometry_missing": missing_notehead_geometry,
        "per_level": {str(k): dict(v) for k, v in sorted(per_level.items())},
        "missing_targets_by_level": {str(k): v for k, v in sorted(missing.items())},
        # Compatibility keys: the obsolete parenthetical path is intentionally empty.
        "structural_parenthetical_positions": 0,
        "structural_parenthetical_labels_expected": 0,
        "structural_parenthetical_labels_mapped": 0,
        "structural_parenthetical_labels_missing": 0,
        "structural_per_level": {},
        "structural_missing_targets_by_level": {},
        "structural_registration_sources": {},
        "structural_reasons": {},
    }
    for level, values in per_level.items():
        stats[f"level{level}_anchors_mapped"] = values["mapped"]
        stats[f"level{level}_hits_expected"] = values["expected"]
        stats[f"level{level}_hits_without_visual_attack"] = values["missing"]

    # Signature retained for physical.py compatibility; there are no structural anchors.
    return dict(anchors_by_page), {}, stats, warnings
