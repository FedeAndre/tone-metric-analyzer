from __future__ import annotations

"""Constraint-based reconstruction of exact rhythmic attacks from the clean
optical notation graph.

Horizontal engraving position is used only to establish left-to-right order and
near-vertical simultaneity.  It is never converted proportionally into musical
time.  Exact onsets arise from notated durations, rests, ties, and measure
capacity.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import ceil
from statistics import median
from typing import Iterable


def frac_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def parse_meter(value: str) -> tuple[int, int]:
    try:
        num_s, den_s = value.strip().split("/", 1)
        num = int(num_s)
        den = int(den_s)
    except Exception as exc:
        raise ValueError("Meter must be written like 4/4, 3/4, or 2/2.") from exc
    if num <= 0 or den <= 0:
        raise ValueError("Meter values must be positive.")
    return num, den


def meter_duration_quarter(meter: tuple[int, int]) -> Fraction:
    num, den = meter
    return Fraction(num * 4, den)


REST_DURATIONS = {
    "rest_whole": Fraction(4),
    "rest_half": Fraction(2),
    "rest_quarter": Fraction(1),
    "rest_8th": Fraction(1, 2),
    "rest_16th": Fraction(1, 4),
    "rest_32nd": Fraction(1, 8),
    "rest_64th": Fraction(1, 16),
}


@dataclass
class VisualEvent:
    id: str
    page: int
    measure_local: int
    staff_id: int
    x: float
    y: float
    kind: str
    voice: str
    base_duration: Fraction
    duration: Fraction
    attacks: bool
    notehead_ids: list[int] = field(default_factory=list)
    stem_id: int | None = None
    dot_confidence: float | None = None
    dotted_selected: bool = False
    tie_continuation: bool = False
    column: int | None = None
    onset: Fraction | None = None
    exact: bool = False


@dataclass
class VoiceSolution:
    key: tuple[int, str]
    events: list[VisualEvent]
    duration_sum: Fraction
    complete: bool
    status: str


def _note_duration(heads: list[dict], stem: dict | None) -> Fraction | None:
    head_types = {str(head.get("head_type", "")) for head in heads}
    if "hollow" in head_types:
        return Fraction(4) if stem is None else Fraction(2)
    if "filled" not in head_types:
        return None
    if stem is None:
        return None
    beam_level = max(0, int(stem.get("beam_level", 0) or 0))
    return Fraction(1, 2 ** beam_level)


def _staff_center(staff: dict) -> float:
    lines = [float(value) for value in staff.get("lines_y", [])]
    return float(sum(lines) / len(lines)) if lines else 0.0


def _voice_for_rest(
    rest: dict,
    staff: dict,
    active_directions: set[str],
) -> str:
    if len(active_directions) == 1:
        return next(iter(active_directions))
    center = _staff_center(staff)
    spacing = float(staff.get("spacing", 18.0) or 18.0)
    delta = float(rest.get("cy", center)) - center
    if delta <= -0.25 * spacing:
        return "up"
    if delta >= 0.25 * spacing:
        return "down"
    if not active_directions:
        return "neutral"
    return "neutral"


def _select_dots_for_complete_voice(
    events: list[VisualEvent],
    capacity: Fraction,
) -> bool:
    """Choose a metrically necessary set of dot candidates, if unique enough.

    Base values are always preferred.  Dots are accepted only when adding their
    1/2-duration increments makes this voice fill the measure exactly.
    """
    base_sum = sum((event.base_duration for event in events), Fraction(0))
    if base_sum == capacity:
        for event in events:
            event.duration = event.base_duration
            event.dotted_selected = False
        return True
    if base_sum > capacity:
        return False

    candidates = [
        event for event in events
        if event.dot_confidence is not None
        and event.dot_confidence >= 0.55
        and event.kind == "note"
    ]
    target = capacity - base_sum
    solutions: list[tuple[float, tuple[int, ...]]] = []
    for count in range(1, len(candidates) + 1):
        for subset in combinations(range(len(candidates)), count):
            increment = sum(
                (candidates[index].base_duration / 2 for index in subset),
                Fraction(0),
            )
            if increment != target:
                continue
            confidence = sum(
                float(candidates[index].dot_confidence or 0.0)
                for index in subset
            )
            # Fewer accepted dots win first; visual confidence breaks ties.
            score = confidence - 0.10 * len(subset)
            solutions.append((score, subset))
    if not solutions:
        return False

    solutions.sort(reverse=True)
    chosen = set(solutions[0][1])
    for index, event in enumerate(candidates):
        event.dotted_selected = index in chosen
        event.duration = (
            event.base_duration * Fraction(3, 2)
            if event.dotted_selected
            else event.base_duration
        )
    return sum((event.duration for event in events), Fraction(0)) == capacity


def _cluster_columns(
    events: list[VisualEvent],
    tolerance: float,
) -> list[list[VisualEvent]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda event: event.x)
    groups: list[list[VisualEvent]] = []
    for event in ordered:
        if not groups:
            groups.append([event])
            continue
        xs = [item.x for item in groups[-1]]
        if event.x - min(xs) <= tolerance:
            groups[-1].append(event)
        else:
            groups.append([event])
    for column, group in enumerate(groups):
        for event in group:
            event.column = column
    return groups


def _set_time(
    times: dict[int, Fraction],
    column: int,
    value: Fraction,
    conflicts: list[str],
    source: str,
) -> bool:
    if column in times and times[column] != value:
        conflicts.append(
            f"column {column}: {frac_text(times[column])} != "
            f"{frac_text(value)} ({source})"
        )
        return False
    times[column] = value
    return True


def _propagate_edges(
    times: dict[int, Fraction],
    edges: list[tuple[int, int, Fraction, str]],
    conflicts: list[str],
) -> None:
    changed = True
    while changed:
        changed = False
        for left, right, duration, source in edges:
            if left in times and right not in times:
                if _set_time(
                    times,
                    right,
                    times[left] + duration,
                    conflicts,
                    source,
                ):
                    changed = True
            elif right in times and left not in times:
                if _set_time(
                    times,
                    left,
                    times[right] - duration,
                    conflicts,
                    source,
                ):
                    changed = True
            elif left in times and right in times:
                if times[right] - times[left] != duration:
                    conflicts.append(
                        f"{source}: column interval "
                        f"{frac_text(times[right] - times[left])} "
                        f"!= duration {frac_text(duration)}"
                    )


def _build_measure_events(
    page: dict,
    measure: dict,
) -> list[VisualEvent]:
    measure_id = int(measure["id"])
    staves = {int(staff["id"]): staff for staff in page.get("staves", [])}
    stems = {int(stem["id"]): stem for stem in page.get("stems", [])}
    dots = {int(dot["id"]): dot for dot in page.get("dots", [])}

    tie_right = {
        int(tie["right_notehead_id"])
        for tie in page.get("tie_candidates", [])
    }

    local_heads = [
        head for head in page.get("noteheads", [])
        if head.get("measure_local") == measure_id
    ]
    by_stem: dict[int, list[dict]] = {}
    unstemmed: list[dict] = []
    for head in local_heads:
        stem_id = head.get("stem_id")
        if stem_id is None:
            unstemmed.append(head)
        else:
            by_stem.setdefault(int(stem_id), []).append(head)

    events: list[VisualEvent] = []
    active_by_staff: dict[int, set[str]] = {}

    for stem_id, heads in by_stem.items():
        stem = stems.get(stem_id)
        if stem is None:
            continue
        duration = _note_duration(heads, stem)
        if duration is None:
            continue
        staff_id = int(heads[0].get("staff_id"))
        direction = str(stem.get("direction") or "neutral")
        active_by_staff.setdefault(staff_id, set()).add(direction)
        dot_rows = [
            dots[int(head["dot_id"])]
            for head in heads
            if head.get("dot_id") is not None
            and int(head["dot_id"]) in dots
        ]
        dot_conf = (
            max(float(row.get("visual_confidence", 0.0)) for row in dot_rows)
            if dot_rows else None
        )
        head_ids = [int(head["id"]) for head in heads]
        tied = bool(set(head_ids) & tie_right)
        events.append(
            VisualEvent(
                id=f"p{page['page']}:m{measure_id}:s{stem_id}",
                page=int(page["page"]),
                measure_local=measure_id,
                staff_id=staff_id,
                x=float(median([float(head["cx"]) for head in heads])),
                y=float(median([float(head["cy"]) for head in heads])),
                kind="note",
                voice=direction,
                base_duration=duration,
                duration=duration,
                attacks=not tied,
                notehead_ids=head_ids,
                stem_id=stem_id,
                dot_confidence=dot_conf,
                tie_continuation=tied,
            )
        )

    # Unstemmed hollow heads are whole-note events.  Heads at the same x form
    # one chord.  Filled unstemmed heads are intentionally left unresolved.
    remaining = sorted(
        [head for head in unstemmed if head.get("head_type") == "hollow"],
        key=lambda head: float(head["cx"]),
    )
    while remaining:
        first = remaining.pop(0)
        staff_id = int(first.get("staff_id"))
        spacing = float(staves.get(staff_id, {}).get("spacing", 18.0) or 18.0)
        group = [first]
        keep = []
        for head in remaining:
            if (
                int(head.get("staff_id")) == staff_id
                and abs(float(head["cx"]) - float(first["cx"])) <= 0.45 * spacing
            ):
                group.append(head)
            else:
                keep.append(head)
        remaining = keep
        head_ids = [int(head["id"]) for head in group]
        tied = bool(set(head_ids) & tie_right)
        dot_rows = [
            dots[int(head["dot_id"])]
            for head in group
            if head.get("dot_id") is not None
            and int(head["dot_id"]) in dots
        ]
        dot_conf = (
            max(float(row.get("visual_confidence", 0.0)) for row in dot_rows)
            if dot_rows else None
        )
        events.append(
            VisualEvent(
                id=f"p{page['page']}:m{measure_id}:w{head_ids[0]}",
                page=int(page["page"]),
                measure_local=measure_id,
                staff_id=staff_id,
                x=float(median([float(head["cx"]) for head in group])),
                y=float(median([float(head["cy"]) for head in group])),
                kind="note",
                voice="neutral",
                base_duration=Fraction(4),
                duration=Fraction(4),
                attacks=not tied,
                notehead_ids=head_ids,
                stem_id=None,
                dot_confidence=dot_conf,
                tie_continuation=tied,
            )
        )

    for rest in page.get("rests", []):
        if rest.get("measure_local") != measure_id:
            continue
        duration = REST_DURATIONS.get(str(rest.get("rest_type", "")))
        if duration is None:
            continue
        staff_id = int(rest.get("staff_id"))
        staff = staves.get(staff_id, {})
        voice = _voice_for_rest(
            rest,
            staff,
            active_by_staff.get(staff_id, set()),
        )
        events.append(
            VisualEvent(
                id=f"p{page['page']}:m{measure_id}:r{rest['id']}",
                page=int(page["page"]),
                measure_local=measure_id,
                staff_id=staff_id,
                x=float(rest["cx"]),
                y=float(rest["cy"]),
                kind="rest",
                voice=voice,
                base_duration=duration,
                duration=duration,
                attacks=False,
            )
        )
    return events


def reconstruct_attacks(
    graph: dict,
    *,
    meter: tuple[int, int],
) -> dict:
    """Reconstruct exact attack onsets conservatively.

    The result is marked ready only when every retained note attack receives an
    exact rational onset and no duration-equation conflict remains.
    """
    capacity = meter_duration_quarter(meter)
    measures_out: list[dict] = []
    attacks_out: list[dict] = []
    unresolved_out: list[dict] = []
    global_measure_index = 0
    absolute_measure_start = Fraction(0)

    for page in graph.get("pages", []):
        staves = {int(staff["id"]): staff for staff in page.get("staves", [])}
        systems = {
            int(system["id"]): system
            for system in page.get("systems", [])
        }
        measures = sorted(
            page.get("measures", []),
            key=lambda measure: (
                int(measure.get("system_id", 0)),
                int(measure.get("measure_in_system", 0)),
                float(measure.get("left", 0.0)),
            ),
        )

        for measure in measures:
            events = _build_measure_events(page, measure)
            system = systems.get(int(measure.get("system_id", 0)), {})
            spacings = [
                float(staves[staff_id].get("spacing", 18.0) or 18.0)
                for staff_id in system.get("staff_ids", [])
                if staff_id in staves
            ]
            spacing = float(median(spacings)) if spacings else 18.0
            columns = _cluster_columns(events, 0.60 * spacing)

            voices: dict[tuple[int, str], list[VisualEvent]] = {}
            neutral_rests: dict[int, list[VisualEvent]] = {}
            for event in events:
                if event.voice == "neutral" and event.kind == "rest":
                    neutral_rests.setdefault(event.staff_id, []).append(event)
                    continue
                voices.setdefault(
                    (event.staff_id, event.voice),
                    [],
                ).append(event)

            # A centered rest belongs to the sole active voice on that staff.
            # With two active voices it remains intentionally unresolved rather
            # than being guessed.
            for staff_id, rests in neutral_rests.items():
                keys = [key for key in voices if key[0] == staff_id]
                if len(keys) == 1:
                    voices[keys[0]].extend(rests)
                elif not keys:
                    voices[(staff_id, "neutral")] = list(rests)
                else:
                    unresolved_out.extend({
                        "page": int(page["page"]),
                        "measure_index": global_measure_index,
                        "event_id": rest.id,
                        "reason": "centered-rest-voice-ambiguous",
                    } for rest in rests)

            solutions: list[VoiceSolution] = []
            edges: list[tuple[int, int, Fraction, str]] = []
            times: dict[int, Fraction] = {}
            conflicts: list[str] = []

            for key, voice_events in voices.items():
                voice_events.sort(key=lambda event: (event.x, event.y))
                complete = _select_dots_for_complete_voice(
                    voice_events,
                    capacity,
                )
                total = sum(
                    (event.duration for event in voice_events),
                    Fraction(0),
                )
                status = (
                    "complete"
                    if complete
                    else "underfull"
                    if total < capacity
                    else "overfull"
                )
                solutions.append(
                    VoiceSolution(
                        key=key,
                        events=voice_events,
                        duration_sum=total,
                        complete=complete,
                        status=status,
                    )
                )

                for left, right in zip(voice_events, voice_events[1:]):
                    if left.column is None or right.column is None:
                        continue
                    if left.column == right.column:
                        continue
                    edges.append(
                        (
                            int(left.column),
                            int(right.column),
                            left.duration,
                            f"staff {key[0]} voice {key[1]}",
                        )
                    )

                if complete and voice_events:
                    first = voice_events[0]
                    if first.column is not None:
                        _set_time(
                            times,
                            int(first.column),
                            Fraction(0),
                            conflicts,
                            f"complete voice {key}",
                        )
                    cursor = Fraction(0)
                    for event in voice_events:
                        if event.column is not None:
                            _set_time(
                                times,
                                int(event.column),
                                cursor,
                                conflicts,
                                f"complete voice {key}",
                            )
                        cursor += event.duration
                    if cursor != capacity:
                        conflicts.append(
                            f"complete voice {key} ended at "
                            f"{frac_text(cursor)} not {frac_text(capacity)}"
                        )

            _propagate_edges(times, edges, conflicts)

            # Opening incomplete measure: if no complete voice exists and the
            # longest explicitly notated voice has a unique shared duration,
            # position it as a pickup at the end of a full measure.
            pickup_shift = Fraction(0)
            if global_measure_index == 0 and not any(s.complete for s in solutions):
                positive = [s.duration_sum for s in solutions if s.duration_sum > 0]
                if positive:
                    actual = max(positive)
                    support = sum(value == actual for value in positive)
                    if actual < capacity and support >= min(2, len(positive)):
                        pickup_shift = capacity - actual
                        for solution in solutions:
                            if solution.duration_sum != actual or not solution.events:
                                continue
                            cursor = pickup_shift
                            for event in solution.events:
                                if event.column is not None:
                                    _set_time(
                                        times,
                                        int(event.column),
                                        cursor,
                                        conflicts,
                                        "opening pickup",
                                    )
                                cursor += event.duration
                        _propagate_edges(times, edges, conflicts)

            for event in events:
                if event.column is not None and int(event.column) in times:
                    value = times[int(event.column)]
                    if Fraction(0) <= value < capacity:
                        event.onset = value
                        event.exact = True

            retained_attacks = [event for event in events if event.attacks]
            unresolved_attacks = [
                event for event in retained_attacks if not event.exact
            ]
            for event in unresolved_attacks:
                unresolved_out.append({
                    "page": int(page["page"]),
                    "measure_index": global_measure_index,
                    "event_id": event.id,
                    "staff_id": event.staff_id,
                    "voice": event.voice,
                    "reason": "exact-onset-not-proven",
                })

            for event in retained_attacks:
                if not event.exact or event.onset is None:
                    continue
                attacks_out.append({
                    "measure_index": global_measure_index,
                    "measure_number": str(global_measure_index + 1),
                    "page": int(page["page"]),
                    "system_id": int(measure.get("system_id", 0)),
                    "offset_in_measure_quarter": frac_text(event.onset),
                    "onset_quarter": frac_text(
                        absolute_measure_start + event.onset
                    ),
                    "duration_quarter": frac_text(event.duration),
                    "staff_id": event.staff_id,
                    "voice": event.voice,
                    "x": round(event.x, 3),
                    "y": round(event.y, 3),
                    "notehead_ids": event.notehead_ids,
                    "stem_id": event.stem_id,
                    "dotted": bool(event.dotted_selected),
                    "tie_continuation": bool(event.tie_continuation),
                })

            measures_out.append({
                "measure_index": global_measure_index,
                "measure_number": str(global_measure_index + 1),
                "page": int(page["page"]),
                "system_id": int(measure.get("system_id", 0)),
                "measure_in_system": int(measure.get("measure_in_system", 0)),
                "start_quarter": frac_text(absolute_measure_start),
                "full_duration_quarter": frac_text(capacity),
                "pickup_shift_quarter": frac_text(pickup_shift),
                "voice_solutions": [
                    {
                        "staff_id": solution.key[0],
                        "voice": solution.key[1],
                        "duration_sum_quarter": frac_text(
                            solution.duration_sum
                        ),
                        "complete": solution.complete,
                        "status": solution.status,
                        "event_count": len(solution.events),
                    }
                    for solution in solutions
                ],
                "column_count": len(columns),
                "known_column_count": len(times),
                "conflicts": sorted(set(conflicts)),
            })

            global_measure_index += 1
            absolute_measure_start += capacity

    # Simultaneous attacks from different voices/chords collapse into one sonic
    # hit.  Tied continuations were already retained in timing but excluded as
    # attacks.
    hits: dict[tuple[int, str], dict] = {}
    for attack in attacks_out:
        key = (
            int(attack["measure_index"]),
            str(attack["offset_in_measure_quarter"]),
        )
        row = hits.setdefault(key, {
            "measure_index": int(attack["measure_index"]),
            "measure_number": attack["measure_number"],
            "offset_in_measure_quarter": attack["offset_in_measure_quarter"],
            "onset_quarter": attack["onset_quarter"],
            "source_count": 0,
            "sources": [],
        })
        row["source_count"] += 1
        row["sources"].append(attack)

    hit_rows = sorted(
        hits.values(),
        key=lambda row: (
            int(row["measure_index"]),
            Fraction(str(row["offset_in_measure_quarter"])),
        ),
    )
    conflicts_total = sum(
        len(measure["conflicts"]) for measure in measures_out
    )
    ready = not unresolved_out and conflicts_total == 0 and bool(hit_rows)
    return {
        "engine": "tma-optical-rhythm-constraints-v1",
        "semantic_timing_used": False,
        "x_position_used_as_time": False,
        "meter": {
            "numerator": meter[0],
            "denominator": meter[1],
            "measure_duration_quarter": frac_text(capacity),
        },
        "measures": measures_out,
        "attacks": attacks_out,
        "hits": hit_rows,
        "unresolved": unresolved_out,
        "stats": {
            "measures": len(measures_out),
            "exact_attacks": len(attacks_out),
            "sonic_hits": len(hit_rows),
            "unresolved_items": len(unresolved_out),
            "constraint_conflicts": conflicts_total,
        },
        "rhythmic_attacks_ready": ready,
    }
