from __future__ import annotations

"""Dissertation-faithful tone-metric wave derivation.

The recursive Levels engine remains authoritative.  This module does not create,
move, delete, or reinterpret any musical event.  It reads the already-computed
``structural_points`` and converts their top occupied recursive level into the
wave envelope described by the dissertation's superposition of Levels.

Registration is intentionally separate: score-time wave points are matched only
to an already-validated attack anchor or parenthetical structural anchor.  No
new PDF x-coordinate is synthesized here.
"""

from collections import defaultdict
from fractions import Fraction


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _measure_rows(analysis_result: dict) -> list[dict]:
    rows = []
    for i, m in enumerate(analysis_result.get("measures", [])):
        start = _frac(m.get("start_quarter"))
        end = _frac(m.get("end_quarter"))
        if start is None or end is None:
            continue
        rows.append({
            "index": int(m.get("index", i)),
            "number": m.get("number", str(i + 1)),
            "start": start,
            "end": end,
        })
    return rows


def build_wave_profile(analysis_result: dict) -> list[dict]:
    """Return the tone-metric wave in score time, before PDF registration.

    The wave height at a structural time is the highest occupied recursive
    Level at that point.  This is the top boundary of the superimposed Levels,
    not a new analysis.  Segment-end sentinels are excluded because they are
    structural recursion boundaries rather than positions belonging to the
    half-open musical segment rendered on the score.
    """
    measures = _measure_rows(analysis_result)
    event_keys = {
        (int(e.get("measure_index", -1)), str(e.get("offset_in_measure_quarter")))
        for seg in analysis_result.get("segments", [])
        for e in seg.get("events", [])
    }
    out = []
    seen = set()
    for si, seg in enumerate(analysis_result.get("segments", [])):
        seg_start = _frac(seg.get("segment_start_quarter"))
        seg_end = _frac(seg.get("segment_end_quarter"))
        if seg_start is None or seg_end is None:
            continue
        for p in seg.get("structural_points", []):
            t = _frac(p.get("time_quarter"))
            if t is None or not (seg_start <= t < seg_end):
                continue
            levels = sorted({int(x) for x in p.get("levels", []) if int(x) > 0})
            if not levels:
                continue
            m = next((m for m in measures if m["start"] <= t < m["end"]), None)
            if m is None:
                continue
            offset = t - m["start"]
            key = (m["index"], offset)
            if key in seen:
                continue
            seen.add(key)
            is_attack = (m["index"], str(offset)) in event_keys
            out.append({
                "segment_index": si,
                "measure_index": m["index"],
                "measure_number": m["number"],
                "onset_quarter": str(t),
                "offset_in_measure_quarter": str(offset),
                "levels": levels,
                "height": max(levels),
                "density": len(levels),
                "lowest_level": min(levels),
                "attack": bool(is_attack),
                "parenthetical": not bool(is_attack),
                "wave_source": "recursive-structural-level-envelope",
            })
    out.sort(key=lambda r: (
        int(r["segment_index"]),
        _frac(r["onset_quarter"], Fraction(0)),
        int(r["measure_index"]),
    ))
    return out


def register_wave_profile(
    wave_profile: list[dict],
    attack_anchors_by_page: dict[int, list[dict]],
    structural_anchors_by_page: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict, list[str]]:
    """Attach fixed score-time wave points to existing validated PDF anchors.

    Attack points reuse exact canonical attack anchors.  Non-attack wave points
    reuse the already-created parenthetical structural anchors.  This
    function has no coordinate-estimation path of its own.
    """
    lookup = {}
    duplicate_keys = []
    for kind, pages in (("attack", attack_anchors_by_page), ("structural", structural_anchors_by_page)):
        for page, rows in pages.items():
            for r in rows:
                mi = int(r.get("measure_index", -1))
                off = _frac(r.get("offset_in_measure_quarter"))
                if off is None:
                    continue
                key = (mi, off)
                if key in lookup:
                    duplicate_keys.append(key)
                    continue
                lookup[key] = (kind, int(page), r)

    by_page = defaultdict(list)
    missing = []
    attack_count = 0
    parenthetical_count = 0
    for p in wave_profile:
        mi = int(p.get("measure_index", -1))
        off = _frac(p.get("offset_in_measure_quarter"))
        if off is None:
            missing.append({**p, "reason": "invalid-wave-offset"})
            continue
        found = lookup.get((mi, off))
        if found is None:
            missing.append({**p, "reason": "validated-level-anchor-not-found"})
            continue
        kind, page, anchor = found
        if bool(p.get("attack")) != (kind == "attack"):
            missing.append({**p, "reason": "wave-anchor-kind-mismatch", "found_kind": kind})
            continue
        # D(t) and lambda(t) remain available in the score-time wave profile and
        # event table, but are deliberately not copied into the PDF anchor payload.
        # This preserves the validated visual wave-anchor contract.
        visual_point = {k: v for k, v in p.items() if k not in {"density", "lowest_level"}}
        row = {
            **visual_point,
            "page_index": int(anchor.get("page_index", page)),
            "system_index": int(anchor.get("system_index", 0)),
            "physical_system_index": int(anchor.get("physical_system_index", anchor.get("system_index", 0))),
            "cx_norm": float(anchor.get("cx_norm")),
            "cy_norm": float(anchor.get("cy_norm", 0.0)),
            "recovered_x_abs": float(anchor.get("recovered_x_abs")),
            "registration_source": "existing-exact-attack-anchor" if kind == "attack" else "existing-parenthetical-structural-anchor",
            "anchor_registration_source": anchor.get("registration_source"),
            "visual_position_is_estimated": bool(anchor.get("visual_position_is_estimated", False)),
        }
        by_page[row["page_index"]].append(row)
        if kind == "attack":
            attack_count += 1
        else:
            parenthetical_count += 1

    for rows in by_page.values():
        rows.sort(key=lambda r: (
            int(r.get("system_index", 0)),
            int(r.get("measure_index", -1)),
            _frac(r.get("offset_in_measure_quarter"), Fraction(0)),
        ))

    warnings = []
    if duplicate_keys:
        warnings.append(f"{len(duplicate_keys)} duplicate pre-existing anchor key(s) were found while registering the wave profile.")
    if missing:
        warnings.append(f"{len(missing)} score-time wave point(s) could not be matched to an existing validated Level anchor.")

    heights = [int(p.get("height", 0)) for p in wave_profile]
    stats = {
        "wave_profile_points_expected": len(wave_profile),
        "wave_profile_points_mapped": sum(len(v) for v in by_page.values()),
        "wave_profile_points_missing": len(missing),
        "wave_attack_points_mapped": attack_count,
        "wave_parenthetical_points_mapped": parenthetical_count,
        "wave_max_height": max(heights, default=0),
        "wave_registration": "score-time-profile->existing-level-anchors-only",
        "wave_coordinate_synthesis": False,
        "wave_missing_points": missing,
        "wave_duplicate_existing_anchor_keys": [f"{mi}:{off}" for mi, off in duplicate_keys],
    }
    return dict(by_page), stats, warnings
