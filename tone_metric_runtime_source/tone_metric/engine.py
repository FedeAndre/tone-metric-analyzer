"""Generalized recursive Tone-Metric Layers engine.

Only the functions reachable from ``analyze`` are retained in this validation build.
Legacy wave, pivot, tree, and superseded recursive-entry implementations were removed
so they cannot alter the recursive Levels result.
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
            segments.append(MeterSegment(
                start=start, end=end, start_measure_index=start_i, end_measure_index=i - 1,
                numerator=cur[0], denominator=cur[1], beat_unit=beat_unit,
                beat_count=beat_count, top_base=top_base, compound=compound,
                hits=seg_hits, warnings=warns,
            ))
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
    """Build the dissertation-style recursive point layers *and* atomic span levels.

    ``point_out[p]`` stores every tone-metric layer that articulates grid point ``p``.
    ``interval_out[p]`` stores the level that directly resolves the atomic interval
    ``p -> p+1``.  Keeping the interval level is essential when the next rhythmic
    subdivision is inserted: the new subdivision is built one level above the
    *parent span*, rather than one level above the numerically highest label already
    present at an endpoint.  The latter was the v0.5.3 bug that made levels climb
    1,2,3,... across a page.
    """
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
            # This level directly articulates the adjacent grid interval.
            interval_out[a] = level
        elif b - a > 1:
            recursive_integer_structure(a, b, base, level + 1, point_out, interval_out, max_level)


def _add_local_structure(
    a: Fraction,
    b: Fraction,
    parent_level: int,
    factor: int,
    stacks: Dict[Fraction, set],
    structural: Dict[Fraction, set],
) -> List[Tuple[Fraction, Fraction, int]]:
    """Insert one rhythmic subdivision above a known parent tone-metric span.

    This implements the relationship described in the dissertation examples: once
    a span has been assigned to level L, its finer rhythmic articulation begins at
    level L+1.  Adjacent spans at the same parent level therefore remain on the same
    horizontal row; processing one span never raises the next span merely because
    the shared endpoint has already received a newly-added label.
    """
    local_points: Dict[int, set] = {}
    local_intervals: Dict[int, int] = {}
    recursive_integer_structure(1, factor + 1, factor, 1, local_points, local_intervals)

    for pos, levels in local_points.items():
        t = a + Fraction(pos - 1, factor) * (b - a)
        shifted = {parent_level + local_level for local_level in levels}
        structural.setdefault(t, set()).update(shifted)
        stacks.setdefault(t, set()).update(shifted)

    children: List[Tuple[Fraction, Fraction, int]] = []
    for local_start, local_level in sorted(local_intervals.items()):
        x = a + Fraction(local_start - 1, factor) * (b - a)
        y = a + Fraction(local_start, factor) * (b - a)
        children.append((x, y, parent_level + local_level))
    return children


def _contains_interior_hit(hits: List[Fraction], a: Fraction, b: Fraction) -> bool:
    return any(a < t < b for t in hits)


def _ternary_spans_from_hits(hits: List[Hit]) -> set[tuple[Fraction, Fraction]]:
    """Recover explicit 3:2 tuplet spans carried by canonical/symbolic hits.

    A span is accepted only when three attack positions from the same tuplet group
    form an exact arithmetic progression.  For MusicXML-only input, where a group
    id may be unavailable, consecutive arity-3 hits are scanned conservatively for
    the same three-position pattern.
    """
    spans: set[tuple[Fraction, Fraction]] = set()
    grouped: Dict[str, List[Fraction]] = {}
    ungrouped: List[Fraction] = []
    for h in hits:
        if getattr(h, "tuplet_arity", None) != 3:
            continue
        span_start = getattr(h, "tuplet_span_start", None)
        span_end = getattr(h, "tuplet_span_end", None)
        if span_start is not None and span_end is not None and span_end > span_start:
            spans.add((span_start, span_end))
        group = getattr(h, "tuplet_group", None)
        if group:
            grouped.setdefault(str(group), []).append(h.onset)
        else:
            ungrouped.append(h.onset)

    def add_windows(times: List[Fraction]):
        ordered = sorted(set(times))
        for i in range(len(ordered) - 2):
            a, b, c = ordered[i:i + 3]
            d1, d2 = b - a, c - b
            if d1 > 0 and d1 == d2:
                spans.add((a, a + 3 * d1))

    for times in grouped.values():
        add_windows(times)
    add_windows(ungrouped)
    return spans


def _lowest_free_level(levels) -> int:
    occupied = {int(x) for x in (levels or []) if int(x) > 0}
    level = 1
    while level in occupied:
        level += 1
    return level


def _contains_complete_ternary_subdivision(hits: List[Fraction], a: Fraction, b: Fraction) -> bool:
    """Fallback detection for a complete unlabelled 3-way attack subdivision."""
    if b <= a:
        return False
    step = (b - a) / 3
    return (a + step) in hits and (a + 2 * step) in hits


def _refine_span(
    a: Fraction,
    b: Fraction,
    hit_times: List[Fraction],
    stacks: Dict[Fraction, set],
    structural: Dict[Fraction, set],
    parent_level: int,
    factor: int,
    next_factor: int,
    depth: int = 0,
    max_depth: int | None = None,
    warnings: List[str] | None = None,
    ternary_spans: set[tuple[Fraction, Fraction]] | None = None,
):
    if not _contains_interior_hit(hit_times, a, b):
        return
    if max_depth is None:
        # Derive the recursion allowance from the finest rhythmic spacing that is
        # actually present in this parent span.  There is no fixed layer-count
        # ceiling: denser notation receives more recursive depth automatically.
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
        max_depth = needed + 6  # margin for recursive sequence articulation
    if depth >= max_depth:
        if warnings is not None:
            warnings.append(
                f'Recursive layer refinement reached the score-derived depth allowance in span {frac_to_str(a)}–{frac_to_str(b)}.'
            )
        return

    # In a binary meter, a complete triplet inside this particular parent span is
    # a local ternary subdivision rather than something to discard.  The tactus
    # itself is unchanged; only this refinement step switches to arity 3.
    explicit_ternary = ternary_spans is not None and (a, b) in ternary_spans
    local_factor = 3 if explicit_ternary or _contains_complete_ternary_subdivision(hit_times, a, b) else factor
    child_spans = _add_local_structure(a, b, parent_level, local_factor, stacks, structural)
    for x, y, child_parent_level in child_spans:
        if _contains_interior_hit(hit_times, x, y):
            _refine_span(
                x, y, hit_times, stacks, structural,
                child_parent_level, next_factor, 2,
                depth + 1, max_depth, warnings, ternary_spans,
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

    stacks: Dict[Fraction, set] = {}
    structural: Dict[Fraction, set] = {}
    for pos, levels in int_layers.items():
        t = segment.start + (pos - 1) * beat
        if t <= segment.end:
            stacks.setdefault(t, set()).update(levels)
            structural.setdefault(t, set()).update(levels)

    hit_times = [h.onset for h in segment.hits]
    ternary_spans = _ternary_spans_from_hits(segment.hits)
    first_factor = 3 if segment.compound else 2
    for i in range(n_intervals):
        a = segment.start + i * beat
        b = min(segment.start + (i + 1) * beat, segment.end)
        if b <= a:
            continue
        # Atomic beat interval i corresponds to integer-grid interval (i+1)->(i+2).
        # Its recursive level is the parent from which any finer rhythmic activity
        # must grow.  Crucially, this is independent of labels added while refining
        # neighboring intervals.
        parent_level = int(interval_levels.get(i + 1, 1))
        _refine_span(
            a, b, hit_times, stacks, structural,
            parent_level, first_factor, 2, warnings=segment.warnings, ternary_spans=ternary_spans,
        )

    # Some tuplets begin on an off-beat and therefore cross the fixed tactus
    # intervals used by the ordinary recursive pass.  The dissertation's layering
    # rule starts a new local unit at the lowest free Level at its beginning.  For
    # an explicit, semantically validated 3:2 tuplet span that still contains
    # unmapped attacks, add exactly one local ternary articulation at that lowest
    # free Level.  This does not move or relabel the underlying binary tactus.
    for a, b in sorted(ternary_spans):
        if a < segment.start or b > segment.end or b <= a:
            continue
        tuple_hits = [t for t in hit_times if a <= t < b]
        if not tuple_hits or all(stacks.get(t) for t in tuple_hits):
            continue
        parent_level = max(0, _lowest_free_level(stacks.get(a, set())) - 1)
        _add_local_structure(a, b, parent_level, 3, stacks, structural)

    event_rows = []
    unmapped = []
    for h in segment.hits:
        levels = sorted(stacks.get(h.onset, []))
        if not levels:
            unmapped.append(h.onset)
            level = 0
        else:
            level = max(levels)
        row = h.to_dict()
        row.update({
            'tone_metric_levels': levels,
            # Definition 7 in the mathematical paper. H(t) and D(t) are kept
            # separate because hierarchical reach and Level convergence answer
            # different analytical questions.
            'tone_metric_height': level,
            'tone_metric_density': len(levels),
            'lowest_tone_metric_level': min(levels) if levels else 0,
        })
        event_rows.append(row)

    if unmapped:
        segment.warnings.append(
            'Some attacks could not be reached by the current binary/ternary subdivision model: ' + ', '.join(frac_to_str(x) for x in unmapped[:8])
        )

    structural_points = [
        {
            'time_quarter': frac_to_str(t),
            'levels': sorted(v),
            'height': max(v),
            'density': len(v),
            'lowest_level': min(v),
        }
        for t, v in sorted(structural.items()) if segment.start <= t <= segment.end
    ]
    level_positions: Dict[str, List[str]] = {}
    for t, levels in sorted(structural.items()):
        if not (segment.start <= t < segment.end):
            continue
        for level in sorted(int(x) for x in levels):
            level_positions.setdefault(str(level), []).append(frac_to_str(t))
    level1_positions = list(level_positions.get('1', []))
    max_level = max((int(k) for k in level_positions), default=0)
    # v0.10.0 validation scope is recursive Levels only.  Pivot/tree analysis is
    # intentionally not executed here, so legacy downstream analyses cannot alter
    # or fail the generalized layer result.
    return {
        'meter': f"{segment.numerator}/{segment.denominator}",
        'segment_start_quarter': frac_to_str(segment.start),
        'segment_end_quarter': frac_to_str(segment.end),
        'beat_unit_quarter': frac_to_str(segment.beat_unit),
        'beat_unit_name': beat_unit_name(segment.beat_unit),
        'level1_positions_quarter': level1_positions,
        'level_positions_quarter': level_positions,
        'max_level': max_level,
        'levels_enabled': list(range(1, max_level + 1)),
        'beat_count': segment.beat_count,
        'top_sequence_base': segment.top_base,
        'compound': segment.compound,
        # This records the arities actually used by the current score-analysis
        # route. The full ordered mixed-arity formalism and cumulative scale
        # vector are implemented independently in tone_metric.theory so they can
        # be tested without changing validated score behavior.
        'arity_policy': {
            'top_recursive_arity': segment.top_base,
            'within_tactus_initial_arity': first_factor,
            'finer_continuation_arity': 2,
            'explicit_local_ternary_span_count': len(ternary_spans),
        },
        'warnings': segment.warnings,
        'events': event_rows,
        'structural_points': structural_points,
    }


def analyze(hits: List[Hit], measures: List[MeasureInfo]) -> dict:
    segs = build_segments(measures, hits)
    analyzed_segments = []
    for s in segs:
        row = analyze_segment(s)
        row['start_measure_index'] = s.start_measure_index
        row['end_measure_index'] = s.end_measure_index
        analyzed_segments.append(row)
    return {
        'segments': analyzed_segments,
        'measures': [
            {
                'index': m.index,
                'number': m.number,
                'start_quarter': frac_to_str(m.start),
                'end_quarter': frac_to_str(m.end),
                'full_duration_quarter': frac_to_str(m.full_duration),
                'actual_duration_quarter': frac_to_str(m.actual_duration),
                'pickup_shift_quarter': frac_to_str(m.pickup_shift),
                'meter': f"{m.numerator}/{m.denominator}",
            }
            for m in measures
        ],
        'measure_count': len(measures),
        'hit_count': len(hits),
    }
