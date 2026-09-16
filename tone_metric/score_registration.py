from __future__ import annotations

"""Register Tone-Metric attacks to the canonical score layout.

Only actual sonic attacks may receive analytical Level labels.  Every label is
registered to exact attacking-notehead geometry belonging to the canonical score-time
column that generated the event.  Rests, sustained notes, augmentation-dot
continuations, ties, accidentals, clefs, and barlines can never supply an x-anchor.
If Audiveris does not preserve exact attacking-head geometry, the label is left
unmapped rather than guessed from a chord/column box or nearby notation.
"""

from collections import defaultdict
from fractions import Fraction


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


def _attack_anchor_position(col: dict) -> tuple[float, float, str] | None:
    """Return one exact attacking-notehead coordinate for this canonical attack.

    A global rhythmic hit may contain simultaneous chords in several voices/staves.
    ``canonical_score.py`` retains an exact notehead-derived center for each attacking
    chord and also chooses one of those attacks as the column representative.  Reuse
    the attacking chord whose exact x-coordinate is closest to that representative;
    never average toward or fall back to non-note notation.
    """
    if not bool(col.get("attack")):
        return None

    points = []
    for event in col.get("notation_events") or []:
        if event.get("kind") != "head" or not bool(event.get("attack")):
            continue
        if not bool(event.get("head_geometry_exact")):
            continue
        try:
            x = event.get("visual_x_abs")
            y = event.get("visual_y_abs")
            if x is None or y is None:
                continue
            points.append((float(x), float(y), str(event.get("chord_id") or "")))
        except Exception:
            continue
    if not points:
        return None

    try:
        representative_x = float(col.get("x_abs"))
    except Exception:
        representative_x = points[0][0]
    x_abs, y_abs, _cid = min(
        points,
        key=lambda p: (abs(p[0] - representative_x), p[0], p[1], p[2]),
    )
    return x_abs, y_abs, "canonical-score-exact-attacking-notehead-center"


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
    missing_attack_anchor = 0
    exact_notehead_anchors = 0

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

        anchor_pos = _attack_anchor_position(col)
        if anchor_pos is None:
            missing_attack_anchor += 1
            for level in target["levels"]:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": "exact-attacking-notehead-geometry-not-found",
                })
            continue

        x_abs, y_abs, registration_source = anchor_pos
        exact_notehead_anchors += 1

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
            "registration_source": registration_source,
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
            f"{missing_total} event-level anchor(s) could not be returned to exact attacking-notehead geometry."
        )
    if missing_attack_anchor:
        warnings.append(
            f"{missing_attack_anchor} attack column(s) lacked exact attacking-notehead geometry and were deliberately not guessed."
        )

    attack_levels = sorted(per_level)
    stats = {
        "final_layer_anchors": sum(len(v) for v in anchors_by_page.values()),
        "layer_anchor_registration": "canonical-score-exact-attacking-notehead-centers-only",
        "analysis_position_policy": "actual-attacks-only",
        "levels_enabled": attack_levels,
        "max_layer_level": max(attack_levels, default=0),
        "layer_hits_expected": sum(v["expected"] for v in per_level.values()),
        "layer_hits_mapped": sum(v["mapped"] for v in per_level.values()),
        "layer_hits_without_visual_attack": missing_total,
        "canonical_column_count": int(meta.get("column_count", 0) or 0),
        "canonical_attack_column_count": int(meta.get("attack_column_count", 0) or 0),
        "canonical_exact_attack_geometry_column_count": int(meta.get("exact_attack_geometry_column_count", 0) or 0),
        "manufactured_x_coordinates": 0,
        "semantic_attack_anchor_missing": missing_attack_anchor,
        "exact_attacking_notehead_anchors": exact_notehead_anchors,
        "semantic_attack_column_fallbacks": 0,
        "per_level": {str(k): dict(v) for k, v in sorted(per_level.items())},
        "missing_targets_by_level": {str(k): v for k, v in sorted(missing.items())},
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

    return dict(anchors_by_page), {}, stats, warnings
