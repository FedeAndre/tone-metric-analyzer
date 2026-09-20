"""Generalized recursive Tone-Metric Levels engine.

The recursive metric grid is an internal device used only to determine which Levels
are occupied by actual sonic attacks.  Empty metric positions, rests, sustained-note
continuations, dots, ties, and barlines are never emitted as analytical events.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil
from typing import Dict, List, Tuple

from .models import Hit, MeasureInfo, MeterSegment, frac_to_str


def _is_power(n: int, base: int) -> bool:
    if n < 1:
        return False
    while n % base == 0 and n > 1:
        n //= base
    return n == 1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _warn_once(warnings: List[str] | None, text: str) -> None:
    if warnings is not None and text not in warnings:
        warnings.append(text)


def _explicit_tuplet_factor(
    a: Fraction,
    b: Fraction,
    hits: List[Hit],
    warnings: List[str] | None = None,
) -> int | None:
    """Return an explicit local tuplet arity for exactly this parent span."""
    span = b - a
    if span <= 0:
        return None

    strong: Dict[int, set[Fraction]] = {}
    aligned: Dict[int, set[Fraction]] = {}
    composite: set[tuple[int, int]] = set()
    for h in hits:
        if not (a <= h.onset < b):
            continue
        rel = (h.onset - a) / span
        for src in h.sources:
            actual = getattr(src, "tuplet_actual", None)
            normal = getattr(src, "tuplet_normal", None)
            if not actual or not normal or int(actual) == int(normal):
                continue
            actual = int(actual)
            normal = int(normal)
            if not _is_prime(actual):
                composite.add((actual, normal))
                continue
            if (rel * actual).denominator != 1:
                continue
            aligned.setdefault(actual, set()).add(h.onset)
            if src.duration > 0 and src.duration * actual == span:
                strong.setdefault(actual, set()).add(h.onset)

    for actual, normal in sorted(composite):
        _warn_once(
            warnings,
            f"Explicit {actual}:{normal} composite tuplet attacks were retained, but no unique local Tone-Metric arity was inferred automatically.",
        )

    strong_candidates = sorted(p for p, times in strong.items() if times)
    if len(strong_candidates) == 1:
        return strong_candidates[0]
    if len(strong_candidates) > 1:
        _warn_once(
            warnings,
            "Conflicting simultaneous explicit tuplet arities occur in the same parent span; attacks were retained without inventing a single hierarchy.",
        )
        return None

    aligned_candidates = sorted(p for p, times in aligned.items() if len(times) >= 2)
    if len(aligned_candidates) == 1:
        return aligned_candidates[0]
    if len(aligned_candidates) > 1:
        _warn_once(
            warnings,
            "Conflicting simultaneous explicit tuplet arities occur in the same parent span; attacks were retained without inventing a single hierarchy.",
        )
    return None


def meter_properties(num: int, den: int) -> Tuple[Fraction, int, int, bool, List[str]]:
    warnings: List[str] = []
    compound = num > 3 and num % 3 == 0
    if compound:
        beat_count = num // 3
        beat_unit = Fraction(12, den)
    else:
        beat_count = num
        beat_unit = Fraction(4, den)
    if _is_power(beat_count, 2):
        top_base = 2
    elif _is_power(beat_count, 3):
        top_base = 3
    else:
        top_base = 2
        warnings.append(
            f"Meter {num}/{den} has {beat_count} tactus units, which is not a pure power of 2 or 3. "
            "Version 0.3 falls back to binary top-level sequencing; this meter needs theoretical validation."
        )
    return beat_unit, beat_count, top_base, compound, warnings


def beat_unit_name(value: Fraction) -> str:
    names = {
        Fraction(4): "whole note",
        Fraction(2): "half note",
        Fraction(1): "quarter note",
        Fraction(1, 2): "eighth note",
        Fraction(1, 4): "sixteenth note",
        Fraction(3, 2): "dotted quarter note",
        Fraction(3): "dotted half note",
    }
    return names.get(value, f"{frac_to_str(value)} quarter-note units")


def build_segments(measures: List[MeasureInfo], hits: List[Hit]) -> List[MeterSegment]:
    if not measures:
        return []
    segments: List[MeterSegment] = []
    start_i = 0
    cur = (measures[0].numerator, measures[0].denominator)
    for i in range(1, len(measures) + 1):
        changed = i == len(measures) or (measures[i].numerator, measures[i].denominator) != cur
        if changed:
            start = measures[start_i].start
            end = measures[i - 1].end
            beat_unit, beat_count, top_base, compound, warns = meter_properties(*cur)
            seg_hits = [h for h in hits if start <= h.onset < end]
            segments.append(
                MeterSegment(
                    start=start,
                    end=end,
                    start_measure_index=start_i,
                    end_measure_index=i - 1,
                    numerator=cur[0],
                    denominator=cur[1],
                    beat_unit=beat_unit,
                    beat_count=beat_count,
                    top_base=top_base,
                    compound=compound,
                    opening_anacrusis=bool(start_i == 0 and getattr(measures[start_i], "opening_anacrusis", False)),
                    hits=seg_hits,
                    warnings=warns,
                )
            )
            if i < len(measures):
                start_i = i
                cur = (measures[i].numerator, measures[i].denominator)
    return segments


def geometric_sequence(base: int, limit: int) -> List[int]:
    if limit < 1:
        return []
    seq = [1]
    x = 1
    if limit >= 2:
        x = 2
        seq.append(2)
    while seq[-1] < limit:
        x = base * x - (base - 1)
        if x == seq[-1]:
            break
        seq.append(x)
    return seq


def _next_seq_anchor(base: int, minimum: int) -> int:
    seq = geometric_sequence(base, max(2, minimum))
    if seq[-1] >= minimum:
        return seq[-1]
    x = seq[-1]
    while x < minimum:
        x = base * x - (base - 1)
    return x


def recursive_integer_structure(
    start: int,
    end: int,
    base: int,
    level: int,
    point_out: Dict[int, set],
    interval_out: Dict[int, int],
    max_level: int | None = None,
):
    """Build the internal recursive point grid and atomic-span parent Levels."""
    if (max_level is not None and level > max_level) or end <= start:
        return
    length = end - start + 1
    seq = geometric_sequence(base, length)
    anchors = [start + n - 1 for n in seq if n <= length]
    if not anchors:
        anchors = [start]
    if anchors[-1] != end:
        anchors.append(end)

    for pos in anchors:
        point_out.setdefault(pos, set()).add(level)

    for a, b in zip(anchors, anchors[1:]):
        if b - a == 1:
            interval_out[a] = level
        elif b - a > 1:
            recursive_integer_structure(a, b, base, level + 1, point_out, interval_out, max_level)


def _add_local_structure(
    a: Fraction,
    b: Fraction,
    parent_level: int,
    factor: int,
    stacks: Dict[Fraction, set],
) -> List[Tuple[Fraction, Fraction, int]]:
    """Refine the internal grid; these positions are not themselves sonic events."""
    local_points: Dict[int, set] = {}
    local_intervals: Dict[int, int] = {}
    recursive_integer_structure(1, factor + 1, factor, 1, local_points, local_intervals)

    for pos, levels in local_points.items():
        t = a + Fraction(pos - 1, factor) * (b - a)
        shifted = {parent_level + local_level for local_level in levels}
        stacks.setdefault(t, set()).update(shifted)

    children: List[Tuple[Fraction, Fraction, int]] = []
    for local_start, local_level in sorted(local_intervals.items()):
        x = a + Fraction(local_start - 1, factor) * (b - a)
        y = a + Fraction(local_start, factor) * (b - a)
        children.append((x, y, parent_level + local_level))
    return children


def _contains_interior_hit(hits: List[Fraction], a: Fraction, b: Fraction) -> bool:
    return any(a < t < b for t in hits)


def _refine_span(
    a: Fraction,
    b: Fraction,
    hits: List[Hit],
    hit_times: List[Fraction],
    stacks: Dict[Fraction, set],
    parent_level: int,
    default_factor: int,
    depth: int = 0,
    max_depth: int | None = None,
    warnings: List[str] | None = None,
):
    if not _contains_interior_hit(hit_times, a, b):
        return
    if max_depth is None:
        local = sorted(t for t in hit_times if a < t < b)
        boundaries = [a, *local, b]
        gaps = [y - x for x, y in zip(boundaries, boundaries[1:]) if y > x]
        min_gap = min(gaps) if gaps else (b - a)
        ratio = (b - a) / min_gap if min_gap > 0 else Fraction(1)
        needed = 0
        scale = Fraction(1)
        while scale < ratio:
            scale *= 2
            needed += 1
        max_depth = needed + 6
    if depth >= max_depth:
        if warnings is not None:
            warnings.append(
                f"Recursive layer refinement reached the score-derived depth allowance in span {frac_to_str(a)}–{frac_to_str(b)}."
            )
        return

    explicit_factor = _explicit_tuplet_factor(a, b, hits, warnings)
    factor = explicit_factor or default_factor
    child_spans = _add_local_structure(a, b, parent_level, factor, stacks)
    for x, y, child_parent_level in child_spans:
        if _contains_interior_hit(hit_times, x, y):
            _refine_span(
                x,
                y,
                hits,
                hit_times,
                stacks,
                child_parent_level,
                2,
                depth + 1,
                max_depth,
                warnings,
            )


def analyze_segment(segment: MeterSegment) -> dict:
    duration = segment.end - segment.start
    beat = segment.beat_unit
    n_intervals = int(ceil(float(duration / beat)))
    max_pos = n_intervals + 1
    top_end = _next_seq_anchor(segment.top_base, max_pos)
    int_layers: Dict[int, set] = {}
    interval_levels: Dict[int, int] = {}
    recursive_integer_structure(1, top_end, segment.top_base, 1, int_layers, interval_levels)
    int_layers = {pos: levels for pos, levels in int_layers.items() if pos <= max_pos}
    interval_levels = {pos: level for pos, level in interval_levels.items() if pos < max_pos}

    # Internal grid only.  Empty positions are never promoted to analytical events.
    stacks: Dict[Fraction, set] = {}
    for pos, levels in int_layers.items():
        t = segment.start + (pos - 1) * beat
        if t <= segment.end:
            stacks.setdefault(t, set()).update(levels)

    hit_times = [h.onset for h in segment.hits]
    first_factor = 3 if segment.compound else 2
    for i in range(n_intervals):
        a = segment.start + i * beat
        b = min(segment.start + (i + 1) * beat, segment.end)
        if b <= a:
            continue
        parent_level = int(interval_levels.get(i + 1, 1))
        _refine_span(
            a,
            b,
            segment.hits,
            hit_times,
            stacks,
            parent_level,
            first_factor,
            warnings=segment.warnings,
        )

    event_rows = []
    unmapped = []
    for h in segment.hits:
        levels = sorted(int(x) for x in stacks.get(h.onset, []) if int(x) > 0)
        if not levels:
            unmapped.append(h.onset)
            level = 0
        else:
            level = max(levels)
        row = h.to_dict()
        row.update(
            {
                "tone_metric_levels": levels,
                "tone_metric_height": level,
                "tone_metric_density": len(levels),
                "lowest_tone_metric_level": min(levels) if levels else 0,
            }
        )
        event_rows.append(row)

    if unmapped:
        segment.warnings.append(
            "Some attacks could not be reached by the current binary/ternary subdivision model: "
            + ", ".join(frac_to_str(x) for x in unmapped[:8])
        )

    # Compatibility field retained for downstream consumers, but it now contains
    # actual attacks only.  No sustained/rest/dot/tie/barline position is emitted.
    structural_points = [
        {
            "time_quarter": event["onset_quarter"],
            "levels": list(event["tone_metric_levels"]),
            "height": int(event["tone_metric_height"]),
            "density": int(event["tone_metric_density"]),
            "lowest_level": int(event["lowest_tone_metric_level"]),
            "attack": True,
            "opening_anacrusis_pre_entry": False,
        }
        for event in event_rows
        if event.get("tone_metric_levels")
    ]

    level_positions: Dict[str, List[str]] = {}
    for event in event_rows:
        t = event.get("onset_quarter")
        for level in sorted(int(x) for x in event.get("tone_metric_levels", []) if int(x) > 0):
            level_positions.setdefault(str(level), []).append(str(t))
    level1_positions = list(level_positions.get("1", []))
    max_level = max((int(k) for k in level_positions), default=0)

    return {
        "meter": f"{segment.numerator}/{segment.denominator}",
        "segment_start_quarter": frac_to_str(segment.start),
        "segment_end_quarter": frac_to_str(segment.end),
        "beat_unit_quarter": frac_to_str(segment.beat_unit),
        "beat_unit_name": beat_unit_name(segment.beat_unit),
        "level1_positions_quarter": level1_positions,
        "level_positions_quarter": level_positions,
        "max_level": max_level,
        "levels_enabled": list(range(1, max_level + 1)),
        "beat_count": segment.beat_count,
        "top_sequence_base": segment.top_base,
        "compound": segment.compound,
        "opening_anacrusis": bool(segment.opening_anacrusis),
        "opening_anacrusis_pre_entry_quarter": None,
        "arity_policy": {
            "top_recursive_arity": segment.top_base,
            "within_tactus_initial_arity": first_factor,
            "finer_continuation_arity": 2,
            "explicit_tuplet_rule": "prime actual-notes arity replaces the default only in the recursively matched parent span",
            "explicit_tuplet_ratios": sorted(
                {
                    f"{int(src.tuplet_actual)}:{int(src.tuplet_normal)}"
                    for h in segment.hits
                    for src in h.sources
                    if getattr(src, "tuplet_actual", None) and getattr(src, "tuplet_normal", None)
                }
            ),
        },
        "warnings": segment.warnings,
        "events": event_rows,
        "structural_points": structural_points,
        "analysis_position_policy": "actual-attacks-only",
    }


def analyze(hits: List[Hit], measures: List[MeasureInfo]) -> dict:
    segs = build_segments(measures, hits)
    analyzed_segments = []
    for s in segs:
        row = analyze_segment(s)
        row["start_measure_index"] = s.start_measure_index
        row["end_measure_index"] = s.end_measure_index
        analyzed_segments.append(row)
    return {
        "segments": analyzed_segments,
        "measures": [
            {
                "index": m.index,
                "number": m.number,
                "start_quarter": frac_to_str(m.start),
                "end_quarter": frac_to_str(m.end),
                "full_duration_quarter": frac_to_str(m.full_duration),
                "actual_duration_quarter": frac_to_str(m.actual_duration),
                "pickup_shift_quarter": frac_to_str(m.pickup_shift),
                "opening_anacrusis": bool(getattr(m, "opening_anacrusis", False)),
                "meter": f"{m.numerator}/{m.denominator}",
            }
            for m in measures
        ],
        "measure_count": len(measures),
        "hit_count": len(hits),
        "analysis_position_policy": "actual-attacks-only",
    }
