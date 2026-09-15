from __future__ import annotations

"""Paper-aligned pivot derivation from the validated wave profile.

A pivot is derived *after* the recursive Levels and wave profile have already
been completed.  This module cannot create musical events, alter score time,
change Levels, or synthesize PDF coordinates.

Definition 8 treats a pivot as a single moment of structural turning: the
rightmost point of a crest immediately before the wave begins to descend.  The
subsequent trough and first renewed ascent are consulted only to determine the
depth of that turn and whether it is simple or compound; they are not part of
the pivot itself.
"""

from collections import defaultdict
from fractions import Fraction


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _point_key(point: dict) -> tuple[int, Fraction] | None:
    off = _frac(point.get("offset_in_measure_quarter"))
    if off is None:
        return None
    try:
        mi = int(point.get("measure_index", -1))
    except Exception:
        return None
    return mi, off


def _height(point: dict) -> int:
    try:
        return int(point.get("height", 0))
    except Exception:
        return 0


def build_pivot_profile(wave_profile: list[dict]) -> list[dict]:
    """Derive simple/compound pivot moments from the fixed wave profile.

    One pivot is emitted for each rightmost crest point whose immediately
    following wave point is lower and whose ensuing descent is eventually
    followed by an ascent within the same meter segment.  The pivot itself is
    the crest point ``t_c``: a single score-time moment of structural turning.

    Classification looks forward without extending the pivot in time:

    * simple   -> total descent depth before the next ascent is exactly 1 level
    * compound -> total descent depth before the next ascent is 2+ levels

    The trough and recovery remain diagnostic points used only to calculate the
    depth.  A terminal descent with no subsequent ascent is not called a pivot,
    because the later ascent is what establishes that a structural turn has
    taken place rather than an unfinished falling tail.
    """
    by_segment: dict[int, list[dict]] = defaultdict(list)
    for point in wave_profile:
        try:
            segment_index = int(point.get("segment_index", 0))
        except Exception:
            segment_index = 0
        by_segment[segment_index].append(point)

    out: list[dict] = []
    pivot_index = 0
    for segment_index in sorted(by_segment):
        points = sorted(
            by_segment[segment_index],
            key=lambda p: (
                _frac(p.get("onset_quarter"), Fraction(0)),
                int(p.get("measure_index", -1)),
                _frac(p.get("offset_in_measure_quarter"), Fraction(0)),
            ),
        )
        i = 0
        while i < len(points) - 1:
            pivot = points[i]
            after = points[i + 1]
            pivot_h = _height(pivot)
            after_h = _height(after)
            if pivot_h <= 0 or after_h <= 0 or after_h >= pivot_h:
                i += 1
                continue

            # The pivot is the rightmost crest point immediately before this
            # first descent.  Look ahead only to classify its depth.
            minimum_h = after_h
            trough_index = i + 1
            recovery_index = None
            j = i + 1
            while j < len(points) - 1:
                current_h = _height(points[j])
                next_h = _height(points[j + 1])
                if next_h < current_h:
                    if next_h < minimum_h:
                        minimum_h = next_h
                    trough_index = j + 1
                    j += 1
                    continue
                if next_h == current_h:
                    if current_h <= minimum_h:
                        trough_index = j + 1
                    j += 1
                    continue
                recovery_index = j + 1
                break

            if recovery_index is None:
                i += 1
                continue

            total_depth = pivot_h - minimum_h
            if total_depth <= 0:
                i += 1
                continue

            kind = "simple" if total_depth == 1 else "compound"
            trough = points[trough_index]
            recovery = points[recovery_index]
            pivot_key = _point_key(pivot)
            after_key = _point_key(after)
            trough_key = _point_key(trough)
            recovery_key = _point_key(recovery)
            if None in (pivot_key, after_key, trough_key, recovery_key):
                i = max(i + 1, trough_index)
                continue

            out.append({
                "pivot_index": pivot_index,
                "segment_index": segment_index,
                "kind": kind,
                "simple": kind == "simple",
                "compound": kind == "compound",
                "drop_depth": int(total_depth),
                "immediate_drop_depth": int(pivot_h - after_h),
                "crest_height": int(pivot_h),
                "after_drop_height": int(after_h),
                "trough_height": int(minimum_h),
                "recovery_height": int(_height(recovery)),
                # Definition-8 pivot moment: the rightmost crest point t_c.
                "pivot_measure_index": int(pivot["measure_index"]),
                "pivot_measure_number": pivot.get("measure_number"),
                "pivot_offset_in_measure_quarter": pivot.get("offset_in_measure_quarter"),
                "pivot_onset_quarter": pivot.get("onset_quarter"),
                "pivot_attack": bool(pivot.get("attack")),
                # Compatibility aliases identify the same single moment; there
                # is deliberately no distinct temporal endpoint for a pivot.
                "start_measure_index": int(pivot["measure_index"]),
                "start_measure_number": pivot.get("measure_number"),
                "start_offset_in_measure_quarter": pivot.get("offset_in_measure_quarter"),
                "start_onset_quarter": pivot.get("onset_quarter"),
                "start_attack": bool(pivot.get("attack")),
                # Look-ahead diagnostics used only for classification.
                "after_drop_measure_index": int(after["measure_index"]),
                "after_drop_measure_number": after.get("measure_number"),
                "after_drop_offset_in_measure_quarter": after.get("offset_in_measure_quarter"),
                "after_drop_onset_quarter": after.get("onset_quarter"),
                "after_drop_attack": bool(after.get("attack")),
                "trough_measure_index": int(trough["measure_index"]),
                "trough_measure_number": trough.get("measure_number"),
                "trough_offset_in_measure_quarter": trough.get("offset_in_measure_quarter"),
                "trough_onset_quarter": trough.get("onset_quarter"),
                "recovery_measure_index": int(recovery["measure_index"]),
                "recovery_measure_number": recovery.get("measure_number"),
                "recovery_offset_in_measure_quarter": recovery.get("offset_in_measure_quarter"),
                "recovery_onset_quarter": recovery.get("onset_quarter"),
                "pivot_source": "definition-8-structural-turning-point",
            })
            pivot_index += 1
            # Suppress duplicate pivots inside the same compound descent.  Resume
            # at the trough; the next transition is the ascent that validated it.
            i = max(i + 1, trough_index)

    return out


def register_pivot_profile(
    pivot_profile: list[dict],
    wave_anchors_by_page: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict, list[str]]:
    """Register each pivot to its single existing wave anchor.

    A pivot has no temporal span and therefore never becomes a cross-system
    segment.  Its marker is placed only at the already-registered wave point for
    the structural turning moment.  No x-coordinate is estimated or synthesized.
    """
    lookup = {}
    duplicates = []
    for page, rows in wave_anchors_by_page.items():
        for row in rows:
            key = _point_key(row)
            if key is None:
                continue
            if key in lookup:
                duplicates.append(key)
                continue
            lookup[key] = (int(page), row)

    by_page: dict[int, list[dict]] = defaultdict(list)
    missing = []
    mapped_pivots = 0
    simple_count = 0
    compound_count = 0

    for pivot in pivot_profile:
        pivot_key = (
            int(pivot["pivot_measure_index"]),
            _frac(pivot["pivot_offset_in_measure_quarter"]),
        )
        found = lookup.get(pivot_key)
        if found is None:
            missing.append({
                **pivot,
                "reason": "existing-wave-anchor-not-found",
            })
            continue

        page, anchor = found
        system = int(anchor.get("system_index", 0))
        x_norm = float(anchor["cx_norm"])
        x_abs = float(anchor["recovered_x_abs"])
        by_page[page].append({
            **pivot,
            "registration_source": "existing-wave-anchor-only",
            "pivot_coordinate_synthesis": False,
            "page_index": page,
            "system_index": system,
            "span_role": "point",
            "cx_norm": x_norm,
            "pivot_cx_norm": x_norm,
            "pivot_x_abs": x_abs,
            # Compatibility start-coordinate aliases refer to the same point.
            "start_page_index": page,
            "start_system_index": system,
            "start_cx_norm": x_norm,
            "start_x_abs": x_abs,
        })
        mapped_pivots += 1
        if pivot.get("kind") == "simple":
            simple_count += 1
        else:
            compound_count += 1

    for rows in by_page.values():
        rows.sort(key=lambda r: (
            int(r.get("system_index", 0)),
            int(r.get("pivot_index", -1)),
        ))

    warnings = []
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate existing wave-anchor key(s) were found while registering pivots.")
    if missing:
        warnings.append(f"{len(missing)} pivot(s) could not be matched to their existing wave anchor.")

    stats = {
        "pivot_profile_expected": len(pivot_profile),
        "pivot_profile_mapped": mapped_pivots,
        "pivot_profile_missing": len(missing),
        "simple_pivots": simple_count,
        "compound_pivots": compound_count,
        "cross_system_pivots": 0,
        "pivot_visual_pieces": sum(len(v) for v in by_page.values()),
        "pivot_registration": "score-time-pivot-point->existing-wave-anchor-only",
        "pivot_coordinate_synthesis": False,
        "pivot_missing": missing,
        "pivot_duplicate_wave_anchor_keys": [f"{mi}:{off}" for mi, off in duplicates],
    }
    return dict(by_page), stats, warnings
