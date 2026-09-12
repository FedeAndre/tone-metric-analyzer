from __future__ import annotations

"""Dissertation-faithful pivot derivation from the validated wave profile.

A pivot is derived *after* the recursive Levels and validated wave profile have
already been completed.  This module cannot create musical events, alter score
time, change Levels, or synthesize PDF coordinates.

The dissertation defines a pivot as the region spanning the moments immediately
before and after a wave descent.  A simple pivot descends one level before the
wave rises again; a compound pivot descends at least two levels before the next
rise.  Equal-height points between the drop and the recovery are permitted and
do not create extra pivots.
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
    """Derive simple/compound pivots from the fixed score-time wave profile.

    One pivot is emitted for each descent that is eventually followed by an
    ascent within the same meter segment.  Its visible region is the first drop:
    the adjacent wave positions immediately before and after that drop.  We then
    look ahead, without changing either endpoint, to classify the descent:

    * simple   -> total descent depth before the next ascent is exactly 1 level
    * compound -> total descent depth before the next ascent is 2+ levels

    A terminal descent with no subsequent ascent is not called a pivot because
    the dissertation defines a pivot as a descent followed by a new ascent.
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
            start = points[i]
            after = points[i + 1]
            start_h = _height(start)
            after_h = _height(after)
            if start_h <= 0 or after_h <= 0 or after_h >= start_h:
                i += 1
                continue

            # The pivot region itself is the adjacent pair spanning the first
            # descent.  Classification may look farther ahead to see whether the
            # descent continues before the eventual recovery.
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
                # First ascent after this descent run.
                recovery_index = j + 1
                break

            if recovery_index is None:
                # A falling tail at the end of a segment is not a pivot because
                # there is no following ascent that establishes a new wave.
                i += 1
                continue

            total_depth = start_h - minimum_h
            if total_depth <= 0:
                i += 1
                continue
            kind = "simple" if total_depth == 1 else "compound"
            trough = points[trough_index]
            recovery = points[recovery_index]
            start_key = _point_key(start)
            after_key = _point_key(after)
            trough_key = _point_key(trough)
            recovery_key = _point_key(recovery)
            if None in (start_key, after_key, trough_key, recovery_key):
                i = max(i + 1, trough_index)
                continue

            out.append({
                "pivot_index": pivot_index,
                "segment_index": segment_index,
                "kind": kind,
                "simple": kind == "simple",
                "compound": kind == "compound",
                "drop_depth": int(total_depth),
                "immediate_drop_depth": int(start_h - after_h),
                "crest_height": int(start_h),
                "after_drop_height": int(after_h),
                "trough_height": int(minimum_h),
                "recovery_height": int(_height(recovery)),
                # Visible dissertation-style region: two adjacent positions
                # immediately before and after the wave drop.
                "start_measure_index": int(start["measure_index"]),
                "start_measure_number": start.get("measure_number"),
                "start_offset_in_measure_quarter": start.get("offset_in_measure_quarter"),
                "start_onset_quarter": start.get("onset_quarter"),
                "start_attack": bool(start.get("attack")),
                "end_measure_index": int(after["measure_index"]),
                "end_measure_number": after.get("measure_number"),
                "end_offset_in_measure_quarter": after.get("offset_in_measure_quarter"),
                "end_onset_quarter": after.get("onset_quarter"),
                "end_attack": bool(after.get("attack")),
                # Look-ahead diagnostics used only to classify the pivot.
                "trough_measure_index": int(trough["measure_index"]),
                "trough_measure_number": trough.get("measure_number"),
                "trough_offset_in_measure_quarter": trough.get("offset_in_measure_quarter"),
                "trough_onset_quarter": trough.get("onset_quarter"),
                "recovery_measure_index": int(recovery["measure_index"]),
                "recovery_measure_number": recovery.get("measure_number"),
                "recovery_offset_in_measure_quarter": recovery.get("offset_in_measure_quarter"),
                "recovery_onset_quarter": recovery.get("onset_quarter"),
                "pivot_source": "derived-wave-descent-before-next-ascent",
            })
            pivot_index += 1
            # Suppress duplicate pivots inside a compound descent.  Resume at the
            # trough; the next transition is the ascent that validated this pivot.
            i = max(i + 1, trough_index)

    return out


def register_pivot_profile(
    pivot_profile: list[dict],
    wave_anchors_by_page: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict, list[str]]:
    """Register pivots only to already-registered wave anchors.

    Same-system pivots become one visual span.  If the two adjacent score-time
    positions straddle a system/page break, two endpoint markers are emitted so
    the pivot is not discarded.  No x-coordinate is estimated or synthesized.
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
    cross_system = 0
    simple_count = 0
    compound_count = 0

    for pivot in pivot_profile:
        start_key = (
            int(pivot["start_measure_index"]),
            _frac(pivot["start_offset_in_measure_quarter"]),
        )
        end_key = (
            int(pivot["end_measure_index"]),
            _frac(pivot["end_offset_in_measure_quarter"]),
        )
        start_found = lookup.get(start_key)
        end_found = lookup.get(end_key)
        if start_found is None or end_found is None:
            missing.append({
                **pivot,
                "reason": "existing-wave-anchor-not-found",
                "missing_start": start_found is None,
                "missing_end": end_found is None,
            })
            continue

        start_page, start = start_found
        end_page, end = end_found
        start_system = int(start.get("system_index", 0))
        end_system = int(end.get("system_index", 0))
        common = {
            **pivot,
            "registration_source": "existing-wave-anchors-only",
            "pivot_coordinate_synthesis": False,
            "start_page_index": start_page,
            "start_system_index": start_system,
            "start_cx_norm": float(start["cx_norm"]),
            "start_x_abs": float(start["recovered_x_abs"]),
            "end_page_index": end_page,
            "end_system_index": end_system,
            "end_cx_norm": float(end["cx_norm"]),
            "end_x_abs": float(end["recovered_x_abs"]),
        }
        if start_page == end_page and start_system == end_system:
            row = {
                **common,
                "page_index": start_page,
                "system_index": start_system,
                "span_role": "complete",
                "cx_norm": (float(start["cx_norm"]) + float(end["cx_norm"])) / 2.0,
            }
            by_page[start_page].append(row)
        else:
            cross_system += 1
            by_page[start_page].append({
                **common,
                "page_index": start_page,
                "system_index": start_system,
                "span_role": "start-endpoint",
                "cx_norm": float(start["cx_norm"]),
            })
            by_page[end_page].append({
                **common,
                "page_index": end_page,
                "system_index": end_system,
                "span_role": "end-endpoint",
                "cx_norm": float(end["cx_norm"]),
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
            str(r.get("span_role", "")),
        ))

    warnings = []
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate existing wave-anchor key(s) were found while registering pivots.")
    if missing:
        warnings.append(f"{len(missing)} pivot(s) could not be matched to their two existing wave anchors.")

    stats = {
        "pivot_profile_expected": len(pivot_profile),
        "pivot_profile_mapped": mapped_pivots,
        "pivot_profile_missing": len(missing),
        "simple_pivots": simple_count,
        "compound_pivots": compound_count,
        "cross_system_pivots": cross_system,
        "pivot_visual_pieces": sum(len(v) for v in by_page.values()),
        "pivot_registration": "score-time-pivot->existing-wave-anchors-only",
        "pivot_coordinate_synthesis": False,
        "pivot_missing": missing,
        "pivot_duplicate_wave_anchor_keys": [f"{mi}:{off}" for mi, off in duplicates],
    }
    return dict(by_page), stats, warnings
