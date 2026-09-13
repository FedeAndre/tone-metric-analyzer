from __future__ import annotations

"""Full dissertation pipeline: Levels -> Waves -> Pivots -> Trees.

The validated recursive Levels engine remains authoritative for ordinary notation.
This module adds one deliberately narrow extension requested for the online edition:
explicit MusicXML duplets (2:n) and triplets (3:n) are treated as local binary or
ternary subdivision stages.  The dissertation defines binary and ternary local
organizations, but it does not state a general tuplet-notation rule; therefore the
extension is recorded in the output rather than silently presented as dissertation text.
"""

from collections import defaultdict
from fractions import Fraction
from typing import Iterable

from .engine import analyze as analyze_levels, geometric_sequence
from .models import Hit, MeasureInfo, frac_to_str
from .waves import build_wave_profile
from .pivots import build_pivot_profile
from .trees import build_tree_profile


def _frac(value, default=None):
    if isinstance(value, Fraction):
        return value
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _height(levels: set[int] | list[int] | tuple[int, ...]) -> int:
    return max((int(x) for x in levels), default=0)


def _recursive_template(start: int, end: int, arity: int, level: int, points: dict[int, set[int]]) -> None:
    """Finite local copy of the dissertation's recursive geometric sieve."""
    if end <= start:
        return
    length = end - start + 1
    anchors = [start + n - 1 for n in geometric_sequence(arity, length) if n <= length]
    if not anchors:
        anchors = [start]
    if anchors[-1] != end:
        anchors.append(end)
    for pos in anchors:
        points.setdefault(pos, set()).add(level)
    for a, b in zip(anchors, anchors[1:]):
        if b - a > 1:
            _recursive_template(a, b, arity, level + 1, points)


def _local_template(arity: int) -> dict[int, set[int]]:
    if arity not in (2, 3):
        raise ValueError("Only explicit duplet (2) and triplet (3) stages are supported.")
    points: dict[int, set[int]] = {}
    _recursive_template(1, arity + 1, arity, 0, points)
    return points


def _source_tuplet_metadata(source) -> tuple[int | None, int | None, str | None, Fraction | None, Fraction | None]:
    actual = getattr(source, "tuplet_actual_notes", None)
    normal = getattr(source, "tuplet_normal_notes", None)
    group = getattr(source, "tuplet_group", None)
    start = getattr(source, "tuplet_span_start", None)
    end = getattr(source, "tuplet_span_end", None)
    return (
        int(actual) if actual else None,
        int(normal) if normal else None,
        str(group) if group else None,
        _frac(start),
        _frac(end),
    )


def _collect_tuplet_groups(hits: Iterable[Hit]) -> tuple[list[dict], list[str]]:
    """Collect explicit 2- and 3-note tuplet spans without inferring them from spacing."""
    raw: dict[tuple, dict] = {}
    warnings: list[str] = []
    unsupported_seen: set[tuple] = set()

    for hit in hits:
        source_rows = list(getattr(hit, "sources", None) or [])
        found_source_metadata = False
        for source in source_rows:
            actual, normal, group, start, end = _source_tuplet_metadata(source)
            if actual is None:
                continue
            found_source_metadata = True
            if actual not in (2, 3):
                marker = (actual, normal, group)
                if marker not in unsupported_seen:
                    warnings.append(
                        f"Tuplet group {group or '(unlabeled)'} uses actual-notes={actual}; only explicit duplets and triplets are supported, so it was not guessed."
                    )
                    unsupported_seen.add(marker)
                continue
            if start is None or end is None or end <= start:
                marker = (actual, normal, group, hit.onset)
                if marker not in unsupported_seen:
                    warnings.append(
                        f"Tuplet attack at {frac_to_str(hit.onset)} has no reliable start/end span; it was left unextended rather than inferred from spacing."
                    )
                    unsupported_seen.add(marker)
                continue
            key = (start, end, actual)
            row = raw.setdefault(key, {
                "start": start,
                "end": end,
                "actual": actual,
                "normal_values": set(),
                "group_ids": set(),
                "attack_times": set(),
                "source_count": 0,
            })
            if normal:
                row["normal_values"].add(normal)
            if group:
                row["group_ids"].add(group)
            row["attack_times"].add(hit.onset)
            row["source_count"] += 1

        # Canonical recovery can occasionally retain only hit-level metadata.
        if not found_source_metadata:
            actual = getattr(hit, "tuplet_arity", None)
            start = _frac(getattr(hit, "tuplet_span_start", None))
            end = _frac(getattr(hit, "tuplet_span_end", None))
            group = getattr(hit, "tuplet_group", None)
            if actual in (2, 3) and start is not None and end is not None and end > start:
                key = (start, end, int(actual))
                row = raw.setdefault(key, {
                    "start": start,
                    "end": end,
                    "actual": int(actual),
                    "normal_values": set(),
                    "group_ids": set(),
                    "attack_times": set(),
                    "source_count": 0,
                })
                if group:
                    row["group_ids"].add(str(group))
                row["attack_times"].add(hit.onset)

    groups: list[dict] = []
    for row in raw.values():
        start, end, actual = row["start"], row["end"], row["actual"]
        span = end - start
        bad_times = []
        for t in sorted(row["attack_times"]):
            rel = (t - start) / span
            if rel < 0 or rel >= 1:
                bad_times.append(t)
                continue
            scaled = rel * actual
            if scaled.denominator != 1:
                bad_times.append(t)
        if bad_times:
            warnings.append(
                f"Tuplet span {frac_to_str(start)}–{frac_to_str(end)} contains attack(s) not aligned to its explicit {actual}-way subdivision; the span was not guessed."
            )
            continue
        groups.append({
            **row,
            "normal_values": sorted(row["normal_values"]),
            "group_ids": sorted(row["group_ids"]),
            "attack_times": sorted(row["attack_times"]),
            "span": span,
        })

    # Crossing spans (a<c<b<d) cannot be interpreted as clean local stages. Nested
    # spans are allowed and are applied from the larger span to the smaller span.
    crossed: set[int] = set()
    for i, a in enumerate(groups):
        for j in range(i + 1, len(groups)):
            b = groups[j]
            crosses = (a["start"] < b["start"] < a["end"] < b["end"]) or (
                b["start"] < a["start"] < b["end"] < a["end"]
            )
            if crosses:
                crossed.update((i, j))
    if crossed:
        warnings.append(
            f"{len(crossed)} explicit tuplet span(s) cross another span; crossing tuplets were skipped because the dissertation extension only accepts disjoint or nested local stages."
        )
    groups = [g for i, g in enumerate(groups) if i not in crossed]
    groups.sort(key=lambda g: (-g["span"], g["start"], g["actual"]))
    return groups, warnings


def _apply_tuplet_extension(levels_result: dict, hits: list[Hit]) -> dict:
    groups, global_warnings = _collect_tuplet_groups(hits)

    for seg in levels_result.get("segments", []) or []:
        seg_start = _frac(seg.get("segment_start_quarter"))
        seg_end = _frac(seg.get("segment_end_quarter"))
        if seg_start is None or seg_end is None:
            continue
        local_groups = [g for g in groups if seg_start <= g["start"] < g["end"] <= seg_end]

        structural: dict[Fraction, set[int]] = {}
        for p in seg.get("structural_points", []) or []:
            t = _frac(p.get("time_quarter"))
            if t is None:
                continue
            structural.setdefault(t, set()).update(int(x) for x in p.get("levels", []) if int(x) > 0)

        extension_audit = []
        local_warnings = []

        # Equal-span groups are one denomination stage: boundary heights are
        # snapshotted before any peer span in the batch is written.
        by_span: dict[Fraction, list[dict]] = defaultdict(list)
        for g in local_groups:
            by_span[g["span"]].append(g)

        for span_len in sorted(by_span, reverse=True):
            batch = sorted(by_span[span_len], key=lambda g: (g["start"], g["actual"]))
            boundary_snapshot = {
                t: _height(structural.get(t, set()))
                for g in batch
                for t in (g["start"], g["end"])
            }
            for g in batch:
                a, b, actual = g["start"], g["end"], int(g["actual"])
                left_h = boundary_snapshot.get(a, 0)
                right_h = boundary_snapshot.get(b, 0)
                if left_h <= 0 or right_h <= 0:
                    local_warnings.append(
                        f"Tuplet span {frac_to_str(a)}–{frac_to_str(b)} could not be attached because one or both parent boundaries are absent from the completed ordinary Levels grid."
                    )
                    continue
                start_level = min(left_h, right_h) + 1
                template = _local_template(actual)
                placed_times = []
                for pos, local_levels in sorted(template.items()):
                    t = a + Fraction(pos - 1, actual) * (b - a)
                    shifted = {start_level + int(x) for x in local_levels}
                    structural.setdefault(t, set()).update(shifted)
                    placed_times.append(t)
                extension_audit.append({
                    "kind": "triplet" if actual == 3 else "duplet",
                    "actual_notes": actual,
                    "normal_notes": g["normal_values"],
                    "span_start_quarter": frac_to_str(a),
                    "span_end_quarter": frac_to_str(b),
                    "span_duration_quarter": frac_to_str(b - a),
                    "start_level": start_level,
                    "boundary_heights_snapshot": [left_h, right_h],
                    "group_ids": g["group_ids"],
                    "placed_points_quarter": [frac_to_str(t) for t in placed_times],
                    "source_count": g["source_count"],
                    "source": "explicit-musicxml-tuplet-metadata",
                    "spacing_inference": False,
                })

        event_times: set[Fraction] = set()
        for event in seg.get("events", []) or []:
            t = _frac(event.get("onset_quarter"))
            if t is None:
                continue
            event_times.add(t)
            levels = sorted(structural.get(t, set()))
            event["tone_metric_levels"] = levels
            event["tone_metric_height"] = max(levels, default=0)
            event["tone_metric_density"] = len(levels)
            event["lowest_tone_metric_level"] = min(levels, default=0)
            event["structural_level_point"] = bool(levels)
            event["attack_level_point"] = True
            event["parenthetical"] = False

        rebuilt_points = []
        for t, level_set in sorted(structural.items()):
            if not (seg_start <= t <= seg_end):
                continue
            levels = sorted(level_set)
            is_attack = t in event_times
            rebuilt_points.append({
                "time_quarter": frac_to_str(t),
                "levels": levels,
                "height": max(levels, default=0),
                "density": len(levels),
                "lowest_level": min(levels, default=0),
                "attack": is_attack,
                "structural_level_point": True,
                "attack_level_point": is_attack,
                "parenthetical": not is_attack,
            })
        seg["structural_points"] = rebuilt_points

        level_positions: dict[str, list[str]] = defaultdict(list)
        for t, level_set in sorted(structural.items()):
            if not (seg_start <= t < seg_end):
                continue
            for level in sorted(level_set):
                level_positions[str(level)].append(frac_to_str(t))
        seg["level_positions_quarter"] = dict(level_positions)
        seg["level1_positions_quarter"] = list(level_positions.get("1", []))
        seg["max_level"] = max((int(k) for k in level_positions), default=0)
        seg["levels_enabled"] = list(range(1, int(seg["max_level"]) + 1))
        seg["tuplet_extensions"] = extension_audit
        seg["tuplet_extension_count"] = len(extension_audit)

        # Recompute unresolved attacks after extension instead of preserving an
        # ordinary-grid warning that has just been resolved by explicit tuplets.
        unresolved = sorted(t for t in event_times if not structural.get(t))
        seg["unreachable_attack_times_quarter"] = [frac_to_str(t) for t in unresolved]
        prior_warnings = [
            w for w in (seg.get("warnings", []) or [])
            if "unmapped" not in str(w).lower()
            and "could not be reached" not in str(w).lower()
            and "strict dissertation grid did not map" not in str(w).lower()
            and "reachable attack time" not in str(w).lower()
        ]
        if unresolved:
            prior_warnings.append(
                "Attack time(s) remain outside the ordinary grid and the explicit duplet/triplet extension: "
                + ", ".join(frac_to_str(t) for t in unresolved[:12])
            )
        seg["warnings"] = prior_warnings + local_warnings
        policy = dict(seg.get("arity_policy", {}) or {})
        policy.update({
            "explicit_tuplet_extension_enabled": True,
            "supported_tuplet_actual_notes": [2, 3],
            "duplet_local_arity": 2,
            "triplet_local_arity": 3,
            "tuplet_source": "explicit MusicXML time-modification/tuplet span metadata only",
            "tuplet_spacing_inference_enabled": False,
        })
        seg["arity_policy"] = policy

    levels_result["engine_contract"] = "dissertation-levels-v0.16-plus-explicit-duplet-triplet-extension-v1"
    levels_result["tuplet_extension_policy"] = {
        "status": "documented extension beyond explicit dissertation notation rules",
        "duplet": "actual-notes=2 -> local binary y=2x-1 recursion",
        "triplet": "actual-notes=3 -> local ternary y=3x-2 recursion",
        "metadata_required": True,
        "spacing_inference": False,
        "warnings": global_warnings,
    }
    return levels_result


def analyze_full(hits: list[Hit], measures: list[MeasureInfo]) -> dict:
    """Return the complete Levels/Waves/Pivots/Trees analysis."""
    levels = analyze_levels(hits, measures)
    levels = _apply_tuplet_extension(levels, hits)
    waves = build_wave_profile(levels)
    pivots = build_pivot_profile(waves)
    trees = build_tree_profile(waves)
    return {
        "analysis_contract": "tone-metric-dissertation-full-v1",
        "levels": levels,
        "waves": waves,
        "pivots": pivots,
        "trees": trees,
        "summary": {
            "measures": int(levels.get("measure_count", 0)),
            "attacks": int(levels.get("hit_count", len(hits))),
            "segments": len(levels.get("segments", []) or []),
            "max_level": max((int(s.get("max_level", 0)) for s in levels.get("segments", []) or []), default=0),
            "wave_points": len(waves),
            "pivots": len(pivots),
            "tree_nodes": len(trees.get("nodes", []) or []),
            "tree_branches": len(trees.get("branches", []) or []),
            "duplet_spans": sum(
                1 for s in levels.get("segments", []) or [] for g in s.get("tuplet_extensions", []) or [] if g.get("kind") == "duplet"
            ),
            "triplet_spans": sum(
                1 for s in levels.get("segments", []) or [] for g in s.get("tuplet_extensions", []) or [] if g.get("kind") == "triplet"
            ),
        },
    }
