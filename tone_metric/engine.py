"""Dissertation-faithful Tone-Metric Levels engine (v0.16).

This module contains only the mathematical Levels procedure.  It deliberately does
*not* interpret notation, recover OMR events, infer tuplets, register PDF geometry,
or calculate waves/pivots/trees.  Those are separate stages.

Operational principles implemented here:
1. Level 1 is a tactus-derived structural layer and exists independently of attacks.
2. A rhythmic denomination is completed everywhere it is needed before the next
   finer denomination is considered.
3. New rows follow the three Chapter-4 boundary cases.  Using the worked figures to
   resolve the dissertation's directional wording contradiction, the new horizontal
   row starts one level above the *lower* of the two boundary heights from the
   preceding completed stage.  If the limits are equal, this is simply one level
   above both.  Crucially, boundary heights are snapshotted before a stage begins;
   labels added during the stage can never raise a neighbouring span.
4. The binary and ternary sequences are explicit: p=2 -> 1,2,3,5,9,17,33,...;
   p=3 -> 1,2,4,10,28,... .  Arity comes from an explicit meter profile, never from
   detected attack spacing.

The output distinguishes structural points from attack points.  Parentheses are a
rendering of structural points with ``attack=False``; they are not manufactured by
post-processing the drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Tuple

from .models import Hit, MeasureInfo, MeterSegment, frac_to_str


class UnsupportedMeterError(ValueError):
    """Raised when the dissertation-faithful core has no explicit meter profile."""


@dataclass(frozen=True)
class MeterProfile:
    numerator: int
    denominator: int
    beat_unit: Fraction
    beat_count: int
    top_sequence_arity: int
    first_subdivision_arity: int
    label: str


# Explicitly modeled dissertation families.  There is intentionally no generic
# fallback: unsupported meters must be added as a documented extension/profile.
_METER_PROFILES: dict[tuple[int, int], MeterProfile] = {
    (2, 2): MeterProfile(2, 2, Fraction(2), 2, 2, 2, "pure-binary"),
    (4, 4): MeterProfile(4, 4, Fraction(1), 4, 2, 2, "pure-binary"),
    (3, 4): MeterProfile(3, 4, Fraction(1), 3, 3, 2, "mixed-ternary-binary"),
    (6, 8): MeterProfile(6, 8, Fraction(3, 2), 2, 2, 3, "mixed-binary-ternary"),
    (9, 8): MeterProfile(9, 8, Fraction(3, 2), 3, 3, 3, "mixed-ternary"),
    (12, 8): MeterProfile(12, 8, Fraction(3, 2), 4, 2, 3, "mixed-binary-ternary"),
}


def meter_profile(num: int, den: int) -> MeterProfile:
    try:
        return _METER_PROFILES[(int(num), int(den))]
    except KeyError as exc:
        raise UnsupportedMeterError(
            f"Meter {num}/{den} has no dissertation-validated Tone-Metric arity profile. "
            "The v0.16 core will not guess an arity from note spacing or beat count."
        ) from exc


def meter_properties(num: int, den: int) -> Tuple[Fraction, int, int, bool, List[str]]:
    """Compatibility wrapper used by the existing data model/UI.

    Unlike older builds, this is a strict lookup and never falls back to binary.
    """
    p = meter_profile(num, den)
    compound = p.first_subdivision_arity == 3
    return p.beat_unit, p.beat_count, p.top_sequence_arity, compound, []


def beat_unit_name(value: Fraction) -> str:
    names = {
        Fraction(4): "whole note",
        Fraction(3): "dotted half note",
        Fraction(2): "half note",
        Fraction(3, 2): "dotted quarter note",
        Fraction(1): "quarter note",
        Fraction(1, 2): "eighth note",
        Fraction(1, 4): "sixteenth note",
        Fraction(1, 8): "thirty-second note",
        Fraction(1, 16): "sixty-fourth note",
    }
    return names.get(value, f"{frac_to_str(value)} quarter-note units")


def build_segments(measures: List[MeasureInfo], hits: List[Hit]) -> List[MeterSegment]:
    """Split at every written meter change; each segment restarts Level 1."""
    if not measures:
        return []
    segments: List[MeterSegment] = []
    start_i = 0
    cur = (measures[0].numerator, measures[0].denominator)
    for i in range(1, len(measures) + 1):
        changed = i == len(measures) or (measures[i].numerator, measures[i].denominator) != cur
        if not changed:
            continue
        start = measures[start_i].start
        end = measures[i - 1].end
        profile = meter_profile(*cur)
        seg_hits = [h for h in hits if start <= h.onset < end]
        segments.append(MeterSegment(
            start=start,
            end=end,
            start_measure_index=start_i,
            end_measure_index=i - 1,
            numerator=cur[0],
            denominator=cur[1],
            beat_unit=profile.beat_unit,
            beat_count=profile.beat_count,
            top_base=profile.top_sequence_arity,
            compound=profile.first_subdivision_arity == 3,
            hits=seg_hits,
            warnings=[],
        ))
        if i < len(measures):
            start_i = i
            cur = (measures[i].numerator, measures[i].denominator)
    return segments


def geometric_sequence(base: int, limit: int) -> List[int]:
    """Return sequence coordinates through the first term >= ``limit``.

    Binary:  1, 2, 3, 5, 9, 17, 33, ...  (y = 2x - 1)
    Ternary: 1, 2, 4, 10, 28, ...          (y = 3x - 2)
    """
    if base not in (2, 3):
        raise ValueError("The dissertation core defines binary (2) and ternary (3) sequences only.")
    if limit < 1:
        return []
    out = [1]
    if limit == 1:
        return out
    out.append(2)
    while out[-1] < limit:
        nxt = base * out[-1] - (base - 1)
        if nxt <= out[-1]:
            raise RuntimeError("Non-increasing Tone-Metric sequence")
        out.append(nxt)
    return out


def _next_sequence_anchor(base: int, minimum: int) -> int:
    return geometric_sequence(base, max(1, minimum))[-1]


def _recursive_grid(
    start: int,
    end: int,
    base: int,
    level: int,
    point_levels: Dict[int, set[int]],
    interval_levels: Dict[int, int] | None = None,
) -> None:
    """Resolve one complete denomination using the chosen geometric sequence.

    ``start`` and ``end`` are inclusive integer grid points.  The same sequence is
    recursively applied to every gap until all adjacent positions of this
    denomination are resolved.  ``interval_levels`` records the level that directly
    resolves each adjacent interval and is used only for audit/testing.
    """
    if end <= start:
        return
    length = end - start + 1
    anchors = [start + n - 1 for n in geometric_sequence(base, length) if n <= length]
    if not anchors:
        anchors = [start]
    if anchors[-1] != end:
        # For local finite units (especially ternary 1..4), the endpoint belongs to
        # the enclosing span even when it is reached by a recursive gap.
        anchors.append(end)

    for pos in anchors:
        point_levels.setdefault(pos, set()).add(level)

    for a, b in zip(anchors, anchors[1:]):
        if b - a == 1:
            if interval_levels is not None:
                interval_levels[a] = level
        elif b - a > 1:
            _recursive_grid(a, b, base, level + 1, point_levels, interval_levels)


def chapter4_placement_case(left_height: int, right_height: int) -> tuple[str, int]:
    """Resolve the three Chapter-4 boundary cases from the worked figures.

    The prose contains an above/below contradiction.  The figures show the invariant
    that preserves horizontal rows without gaps: start the new row one level above
    the *lower* boundary height from the previous stage.  Thus:
      - equal limits -> one above both;
      - left lower than right -> one above left;
      - right lower than left -> one above right.

    This is intentionally *not* ``max(stack)+1``.
    """
    left = int(left_height)
    right = int(right_height)
    if left < 0 or right < 0:
        raise ValueError("Boundary heights must be non-negative")
    if left == right:
        return "equal-limits", left + 1
    if left < right:
        return "left-lower-than-right", left + 1
    return "right-lower-than-left", right + 1


def _height(stacks: Dict[Fraction, set[int]], t: Fraction) -> int:
    levels = stacks.get(t) or set()
    return max((int(x) for x in levels), default=0)


def _local_stage_template(factor: int) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Template levels for one subdivision stage, normalized to start at Level 0."""
    if factor not in (2, 3):
        raise ValueError("Only binary and ternary local stages are dissertation-defined.")
    point_levels: dict[int, set[int]] = {}
    interval_levels: dict[int, int] = {}
    _recursive_grid(1, factor + 1, factor, 0, point_levels, interval_levels)
    return point_levels, interval_levels


def _contains_interior(times: Iterable[Fraction], a: Fraction, b: Fraction) -> bool:
    return any(a < t < b for t in times)


def _reachable_relative(position: Fraction, first_factor: int) -> tuple[bool, int]:
    """Return (reachable, exact subdivision-stage depth) for an interior beat offset.

    After the explicitly modeled first local factor, all finer ordinary denominations
    are binary.  This is a mathematical reachability check, not a spacing heuristic.
    """
    p = Fraction(position)
    if not (Fraction(0) < p < Fraction(1)):
        return True, 0
    den = int(p.denominator)
    twos = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    if first_factor == 2:
        return den == 1, twos
    if first_factor == 3:
        if den not in (1, 3):
            return False, 0
        # A ternary stage is always performed before any binary continuation.
        return True, 1 + twos
    return False, 0


def _attack_reachability(
    hit_times: list[Fraction], start: Fraction, beat: Fraction, first_factor: int
) -> tuple[set[Fraction], int]:
    unreachable: set[Fraction] = set()
    max_depth = 0
    for t in hit_times:
        rel = (t - start) / beat
        # Beat-boundary attacks are already represented by the top denomination.
        if rel.denominator == 1:
            continue
        within = rel - (rel.numerator // rel.denominator)
        ok, depth = _reachable_relative(within, first_factor)
        if not ok:
            unreachable.add(t)
        else:
            max_depth = max(max_depth, depth)
    return unreachable, max_depth


def _apply_complete_stage(
    *,
    spans: list[tuple[Fraction, Fraction]],
    factor: int,
    hit_times: list[Fraction],
    unreachable: set[Fraction],
    stacks: Dict[Fraction, set[int]],
    structural: Dict[Fraction, set[int]],
    stage_index: int,
    denomination: Fraction,
) -> tuple[list[tuple[Fraction, Fraction]], dict]:
    """Complete one rhythmic denomination across all active spans before returning."""
    # Snapshot is the critical v0.16 invariant: new labels in this stage cannot
    # influence any other span's boundary case.
    previous_heights = {t: _height(stacks, t) for span in spans for t in span}
    template_points, _template_intervals = _local_stage_template(factor)
    next_spans: list[tuple[Fraction, Fraction]] = []
    placement_counts: dict[str, int] = {
        "equal-limits": 0,
        "left-lower-than-right": 0,
        "right-lower-than-left": 0,
    }
    processed = 0

    eligible_times = [t for t in hit_times if t not in unreachable]
    for a, b in spans:
        if b <= a or not _contains_interior(eligible_times, a, b):
            continue
        left_h = previous_heights.get(a, _height(stacks, a))
        right_h = previous_heights.get(b, _height(stacks, b))
        case, start_level = chapter4_placement_case(left_h, right_h)
        placement_counts[case] += 1
        processed += 1

        # A local template's Level 0 becomes ``start_level``.  For ternary units the
        # recursive Level 1 resolves the remaining gap before any finer stage begins.
        for pos, normalized_levels in template_points.items():
            t = a + Fraction(pos - 1, factor) * (b - a)
            shifted = {start_level + int(local) for local in normalized_levels}
            stacks.setdefault(t, set()).update(shifted)
            structural.setdefault(t, set()).update(shifted)

        for j in range(factor):
            x = a + Fraction(j, factor) * (b - a)
            y = a + Fraction(j + 1, factor) * (b - a)
            if _contains_interior(eligible_times, x, y):
                next_spans.append((x, y))

    return next_spans, {
        "stage_index": stage_index,
        "subdivision_arity": factor,
        "denomination_quarter": frac_to_str(denomination),
        "denomination_name": beat_unit_name(denomination),
        "input_span_count": len(spans),
        "processed_span_count": processed,
        "next_active_span_count": len(next_spans),
        "placement_cases": placement_counts,
        "boundary_snapshot_only": True,
    }


def analyze_segment(segment: MeterSegment) -> dict:
    profile = meter_profile(segment.numerator, segment.denominator)
    duration = segment.end - segment.start
    beat = profile.beat_unit
    if duration <= 0 or beat <= 0 or duration % beat != 0:
        raise ValueError(
            f"Segment {frac_to_str(segment.start)}–{frac_to_str(segment.end)} is not an exact multiple "
            f"of the explicit {beat_unit_name(beat)} tactus unit for {segment.numerator}/{segment.denominator}."
        )

    n_intervals = int(duration / beat)
    max_pos = n_intervals + 1

    # --- Stage 0: tactus-derived sequential/recursive denomination. ---
    # Extend to the next true sequence anchor before trimming the finite excerpt.
    # This avoids turning the excerpt's arbitrary end into a false Level-1 anchor.
    top_end = _next_sequence_anchor(profile.top_sequence_arity, max_pos)
    top_points: Dict[int, set[int]] = {}
    top_intervals: Dict[int, int] = {}
    _recursive_grid(1, top_end, profile.top_sequence_arity, 1, top_points, top_intervals)

    stacks: Dict[Fraction, set[int]] = {}
    structural: Dict[Fraction, set[int]] = {}
    for pos, levels in top_points.items():
        if pos > max_pos:
            continue
        t = segment.start + (pos - 1) * beat
        if t <= segment.end:
            stacks.setdefault(t, set()).update(levels)
            structural.setdefault(t, set()).update(levels)

    hit_times = sorted({h.onset for h in segment.hits})
    unreachable, required_depth = _attack_reachability(
        hit_times, segment.start, beat, profile.first_subdivision_arity
    )

    # All beat intervals are possible parents at the first finer denomination.
    active_spans = [
        (segment.start + i * beat, segment.start + (i + 1) * beat)
        for i in range(n_intervals)
    ]
    stage_audit = [{
        "stage_index": 0,
        "subdivision_arity": profile.top_sequence_arity,
        "denomination_quarter": frac_to_str(beat),
        "denomination_name": beat_unit_name(beat),
        "sequence": geometric_sequence(profile.top_sequence_arity, max_pos),
        "structural_point_count": sum(1 for t in structural if segment.start <= t <= segment.end),
        "level1_is_attack_independent": True,
    }]

    # --- Finer denominations, breadth-first. ---
    denomination = beat
    for depth in range(1, required_depth + 1):
        factor = profile.first_subdivision_arity if depth == 1 else 2
        denomination /= factor
        active_spans, audit = _apply_complete_stage(
            spans=active_spans,
            factor=factor,
            hit_times=hit_times,
            unreachable=unreachable,
            stacks=stacks,
            structural=structural,
            stage_index=depth,
            denomination=denomination,
        )
        stage_audit.append(audit)
        if not active_spans:
            # No finer ordinary attack remains.  Do not generate unused grid levels.
            break

    event_rows = []
    unmapped: list[Fraction] = []
    attack_times = set(hit_times)
    for event_index, h in enumerate(segment.hits):
        levels = sorted(int(x) for x in stacks.get(h.onset, set()))
        if not levels:
            unmapped.append(h.onset)
        row = h.to_dict()
        row.update({
            "event_index": event_index,
            "tone_metric_levels": levels,
            "tone_metric_height": max(levels, default=0),
            "tone_metric_density": len(levels),
            "lowest_tone_metric_level": min(levels, default=0),
            "structural_level_point": bool(levels),
            "attack_level_point": True,
            "parenthetical": False,
        })
        event_rows.append(row)

    warnings = list(segment.warnings)
    if unreachable:
        warnings.append(
            "Strict dissertation grid did not map non-binary/non-profile attack time(s): "
            + ", ".join(frac_to_str(x) for x in sorted(unreachable)[:12])
            + ". No arity was inferred from spacing."
        )
    if unmapped:
        extra = [x for x in unmapped if x not in unreachable]
        if extra:
            warnings.append(
                "Reachable attack time(s) remained unmapped, indicating a Levels-engine error: "
                + ", ".join(frac_to_str(x) for x in extra[:12])
            )

    structural_points = []
    for t, levels_set in sorted(structural.items()):
        if not (segment.start <= t <= segment.end):
            continue
        levels = sorted(int(x) for x in levels_set)
        is_attack = t in attack_times
        structural_points.append({
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

    level_positions: Dict[str, List[str]] = {}
    for point in structural_points:
        t = Fraction(point["time_quarter"])
        if not (segment.start <= t < segment.end):
            continue
        for level in point["levels"]:
            level_positions.setdefault(str(level), []).append(point["time_quarter"])

    max_level = max((int(k) for k in level_positions), default=0)
    return {
        "meter": f"{segment.numerator}/{segment.denominator}",
        "segment_start_quarter": frac_to_str(segment.start),
        "segment_end_quarter": frac_to_str(segment.end),
        "beat_unit_quarter": frac_to_str(beat),
        "beat_unit_name": beat_unit_name(beat),
        "beat_count": profile.beat_count,
        "meter_profile": profile.label,
        "top_sequence_base": profile.top_sequence_arity,
        "compound": profile.first_subdivision_arity == 3,
        "level1_positions_quarter": list(level_positions.get("1", [])),
        "level_positions_quarter": level_positions,
        "max_level": max_level,
        "levels_enabled": list(range(1, max_level + 1)),
        "arity_policy": {
            "source": "explicit-meter-profile",
            "top_recursive_arity": profile.top_sequence_arity,
            "within_tactus_initial_arity": profile.first_subdivision_arity,
            "finer_continuation_arity": 2,
            "spacing_inference_enabled": False,
            "generic_4_4_triplet_extension_enabled": False,
        },
        "denomination_stages": stage_audit,
        "chapter4_boundary_rule": {
            "equal": "one above both limits",
            "left_lower": "one above the left (lower) limit",
            "right_lower": "one above the right (lower) limit",
            "implemented_as": "min(previous_stage_left_height, previous_stage_right_height) + 1",
            "uses_stage_snapshot": True,
        },
        "warnings": warnings,
        "events": event_rows,
        "structural_points": structural_points,
        "unreachable_attack_times_quarter": [frac_to_str(x) for x in sorted(unreachable)],
    }


def analyze(hits: List[Hit], measures: List[MeasureInfo]) -> dict:
    """Run the pure Levels engine on already-preprocessed sonic attacks."""
    segments = build_segments(measures, hits)
    analyzed_segments = []
    for s in segments:
        row = analyze_segment(s)
        row["start_measure_index"] = s.start_measure_index
        row["end_measure_index"] = s.end_measure_index
        analyzed_segments.append(row)
    return {
        "engine_contract": "dissertation-levels-v0.16",
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
                "meter": f"{m.numerator}/{m.denominator}",
            }
            for m in measures
        ],
        "measure_count": len(measures),
        "hit_count": len(hits),
    }
