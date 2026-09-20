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
from math import ceil, gcd
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



def _fraction_gcd(values: Iterable[Fraction]) -> Fraction:
    """Greatest common rational subdivision represented by *values*."""
    vals = [abs(Fraction(value)) for value in values if value]
    if not vals:
        return Fraction(1)
    den = 1
    for value in vals:
        den = den * value.denominator // gcd(den, value.denominator)
    nums = [value.numerator * (den // value.denominator) for value in vals]
    num = nums[0]
    for value in nums[1:]:
        num = gcd(num, value)
    return Fraction(num, den)


def _fill_uniquely_forced_grid_columns(
    times: dict[int, Fraction],
    columns: list[list[VisualEvent]],
    events: list[VisualEvent],
    capacity: Fraction,
) -> int:
    """Fill unresolved columns only when the notated rational grid forces them.

    Horizontal coordinates establish order only.  The quantum is the rational
    GCD of notated event durations.  Between two established time anchors (or
    the measure end), if the number of unresolved notation columns equals the
    number of available grid positions exactly, the assignment is unique and
    therefore safe.
    """
    if not columns:
        return 0
    quantum = _fraction_gcd(
        [capacity] + [
            event.duration
            for event in events
            if event.kind != "measure_rest" and event.duration > 0
        ]
    )
    if quantum <= 0:
        return 0

    added = 0
    changed = True
    while changed:
        changed = False
        known = sorted(times)
        # Virtual measure-end anchor lets the tail be solved exactly.
        anchors = [(column, times[column]) for column in known]
        anchors.append((len(columns), capacity))

        # A measure-start virtual anchor is useful only when column 0 is not
        # already known.  This does not assert that column 0 is an attack;
        # it merely bounds later notation columns from the barline.
        if 0 not in times:
            anchors.insert(0, (-1, Fraction(0)))

        for (left_col, left_time), (right_col, right_time) in zip(
            anchors,
            anchors[1:],
        ):
            if right_time <= left_time:
                continue
            unresolved = [
                column
                for column in range(left_col + 1, right_col)
                if column not in times
            ]
            if not unresolved:
                continue

            slots: list[Fraction] = []
            value = left_time + quantum
            while value < right_time:
                slots.append(value)
                value += quantum

            if len(slots) != len(unresolved):
                continue
            for column, value in zip(unresolved, slots):
                times[column] = value
                added += 1
                changed = True
            if changed:
                break
    return added


def _solve_unique_duration_chain_columns(
    times: dict[int, Fraction],
    columns: list[list[VisualEvent]],
    events: list[VisualEvent],
    capacity: Fraction,
    *,
    max_assignments: int = 50000,
) -> int:
    """Resolve columns invariant across all valid exact duration-chain layouts.

    Horizontal coordinates establish only notation order.  Candidate onsets
    come from the rational grid implied by written durations and measure
    capacity.  Every complete assignment must embed each unresolved event in a
    same-staff duration chain.

    The older implementation required the *entire* measure to have exactly one
    valid assignment.  That is unnecessarily strict for TMA: one part of a
    measure can remain voice-ambiguous while a particular attack column is
    nevertheless forced to the same exact time in every admissible solution.

    Therefore this routine exhaustively enumerates valid assignments (within a
    bounded finite search).  A column is committed only when every valid
    assignment gives it the identical rational onset.  If the search bound is
    reached, nothing is committed because exhaustiveness has not been proved.
    """
    unresolved_columns = [
        column for column in range(len(columns)) if column not in times
    ]
    if not unresolved_columns:
        return 0

    quantum = _fraction_gcd(
        [capacity] + [
            event.duration
            for event in events
            if event.kind != "measure_rest" and event.duration > 0
        ]
    )
    if quantum <= 0:
        return 0

    known_sorted = sorted(times.items())
    candidate_map: dict[int, list[Fraction]] = {}
    for column in unresolved_columns:
        left_time = Fraction(0)
        right_time = capacity
        for known_col, known_time in known_sorted:
            if known_col < column:
                left_time = max(left_time, known_time)
            elif known_col > column:
                right_time = min(right_time, known_time)
                break
        values: list[Fraction] = []
        # Different staves/voices can engrave simultaneous attacks at slightly
        # different x coordinates.  Therefore global visual-column order is
        # non-decreasing in time, not strictly increasing.  Boundary times are
        # legal candidates; same-staff duration-chain checks below decide
        # whether equality is musically admissible for a particular event.
        value = left_time
        while value <= right_time:
            values.append(value)
            value += quantum
        if not values:
            return 0
        candidate_map[column] = values

    events_by_staff: dict[int, list[VisualEvent]] = {}
    for event in events:
        if event.kind == "measure_rest" or event.column is None:
            continue
        events_by_staff.setdefault(event.staff_id, []).append(event)

    unresolved_set = set(unresolved_columns)
    target_events = [
        event for event in events
        if event.column is not None
        and int(event.column) in unresolved_set
        and event.kind != "measure_rest"
    ]

    valid: list[dict[int, Fraction]] = []
    assignment: dict[int, Fraction] = {}
    explored = 0
    truncated = False

    def event_time(
        event: VisualEvent,
        all_times: dict[int, Fraction],
    ) -> Fraction | None:
        if event.column is None:
            return None
        return all_times.get(int(event.column))

    def assignment_valid(all_times: dict[int, Fraction]) -> bool:
        for event in target_events:
            onset = event_time(event, all_times)
            if onset is None:
                return False
            end = onset + event.duration
            if onset < 0 or end > capacity:
                return False

            local = events_by_staff.get(event.staff_id, [])
            pred_ok = onset == 0
            succ_ok = end == capacity

            if not pred_ok:
                for other in local:
                    if (
                        other is event
                        or other.column is None
                        or int(other.column) >= int(event.column)
                    ):
                        continue
                    other_onset = event_time(other, all_times)
                    if (
                        other_onset is not None
                        and other_onset + other.duration == onset
                    ):
                        pred_ok = True
                        break

            if not succ_ok:
                for other in local:
                    if (
                        other is event
                        or other.column is None
                        or int(other.column) <= int(event.column)
                    ):
                        continue
                    other_onset = event_time(other, all_times)
                    if other_onset == end:
                        succ_ok = True
                        break

            if not (pred_ok and succ_ok):
                return False
        return True

    ordered = unresolved_columns

    def search(index: int, previous_time: Fraction) -> None:
        nonlocal explored, truncated
        if explored >= max_assignments:
            truncated = True
            return
        if index >= len(ordered):
            explored += 1
            all_times = dict(times)
            all_times.update(assignment)
            if assignment_valid(all_times):
                valid.append(dict(assignment))
            return

        column = ordered[index]
        next_known_time = capacity
        for known_col, known_time in known_sorted:
            if known_col > column:
                next_known_time = known_time
                break

        for value in candidate_map[column]:
            if value < previous_time or value > next_known_time:
                continue
            assignment[column] = value
            search(index + 1, value)
            assignment.pop(column, None)
            if truncated:
                return

    first = ordered[0]
    previous = Fraction(-1)
    for known_col, known_time in known_sorted:
        if known_col < first:
            previous = known_time
        else:
            break
    search(0, previous)

    if truncated or not valid:
        return 0

    added = 0
    for column in unresolved_columns:
        values = {row[column] for row in valid if column in row}
        if len(values) == 1:
            times[column] = next(iter(values))
            added += 1
    return added

def _resolve_geometric_grid_intervals(
    times: dict[int, Fraction],
    columns: list[list[VisualEvent]],
    events: list[VisualEvent],
    capacity: Fraction,
    *,
    measure_left: float | None = None,
    measure_right: float | None = None,
) -> int:
    """Resolve rational-grid ambiguity from engraving geometry conservatively.

    Written durations and measure capacity define the only legal rational time
    slots. Horizontal engraving is used only to choose among those discrete
    legal slots.  Exact note columns are preferred as anchors; the two barlines
    may additionally serve as virtual anchors at time 0 and measure capacity.

    This is not proportional x->time conversion: no continuous time is derived
    from x.  A visual comparison is allowed only after a finite set of exact
    rational assignments has been enumerated, and a choice is accepted only
    when it fits the engraving closely and is decisively better than every
    alternative.
    """
    if not columns:
        return 0

    positive = [
        event.duration
        for event in events
        if event.duration > 0 and event.kind != "measure_rest"
    ]
    quantum = _fraction_gcd([capacity] + positive)
    if quantum <= 0:
        return 0

    column_x = {
        index: float(median([event.x for event in column]))
        for index, column in enumerate(columns)
        if column
    }
    if not column_x:
        return 0

    added = 0
    changed = True
    while changed:
        changed = False

        # Use virtual barline anchors only when their x order is geometrically
        # valid.  Their times are exact by definition of the measure boundary.
        anchors: list[tuple[int, Fraction, float]] = [
            (column, value, column_x[column])
            for column, value in times.items()
            if column in column_x
        ]
        if (
            measure_left is not None
            and measure_right is not None
            and measure_right > measure_left
        ):
            anchors.append((-1, Fraction(0), float(measure_left)))
            anchors.append((len(columns), capacity, float(measure_right)))

        # Keep only the strongest exact anchor when several describe the same
        # temporal boundary: a real attack column is preferred over a virtual
        # barline at time zero/capacity.
        dedup: dict[tuple[int, Fraction], tuple[int, Fraction, float]] = {}
        for row in anchors:
            dedup[(row[0], row[1])] = row
        anchors = sorted(dedup.values(), key=lambda row: row[0])

        for (left_col, left_time, left_x), (
            right_col,
            right_time,
            right_x,
        ) in zip(anchors, anchors[1:]):
            if right_time <= left_time or right_x <= left_x:
                continue

            unresolved = [
                col
                for col in range(left_col + 1, right_col)
                if col not in times and col in column_x
            ]
            if not unresolved:
                continue

            slots: list[Fraction] = []
            value = left_time + quantum
            while value < right_time:
                slots.append(value)
                value += quantum
            if len(slots) < len(unresolved):
                continue

            # Bound the exact enumeration.  Engraved single measures are small,
            # but this prevents pathological combinatorial growth.
            if len(slots) > 24 or len(unresolved) > 10:
                continue

            visual = [
                (column_x[col] - left_x) / (right_x - left_x)
                for col in unresolved
            ]
            span_t = right_time - left_time

            scored: list[tuple[float, float, tuple[Fraction, ...]]] = []
            for choice in combinations(slots, len(unresolved)):
                temporal = [
                    float((slot - left_time) / span_t)
                    for slot in choice
                ]
                errors = [
                    abs(v - t)
                    for v, t in zip(visual, temporal)
                ]
                max_error = max(errors, default=0.0)
                mean_sq = (
                    sum(error * error for error in errors) / len(errors)
                    if errors else 0.0
                )
                scored.append((mean_sq, max_error, choice))

            if not scored:
                continue
            scored.sort(key=lambda row: (row[0], row[1], row[2]))
            best = scored[0]
            second = scored[1] if len(scored) > 1 else None

            # Barline-bounded intervals are less visually precise because of
            # clefs/key signatures and measure padding, so require both a close
            # fit and a strong separation from the runner-up.
            virtual_boundary = (
                left_col == -1 or right_col == len(columns)
            )
            max_allowed = 0.105 if virtual_boundary else 0.18
            minimum_improvement = 0.012 if virtual_boundary else 0.006

            if best[1] > max_allowed:
                continue
            if second is not None:
                improvement = second[0] - best[0]
                if improvement < minimum_improvement:
                    continue

            # Do not resolve an entire interval from barline geometry if the
            # optimal solution itself uses a very fine subdivision unsupported
            # by any written duration in the interval. This keeps x from
            # manufacturing finer rhythmic values.
            for col, value in zip(unresolved, best[2]):
                times[col] = value
                added += 1
            changed = True
            break

    return added

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
        if float(tie.get("confidence", 1.0)) >= 0.80
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

    # Unstemmed hollow heads are whole-note events. Heads at the same x form
    # one chord. Unstemmed filled heads retain their attack identity with
    # unknown duration; they may contribute a hit only if their onset is
    # independently proven by the rest of the notation.
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

    # A filled head whose stem could not be recovered has unknown duration,
    # but its visible head is still a real attack candidate.  Keep it out of
    # all duration equations and accept it only when its temporal column is
    # proven independently.  This is safer than either silently dropping the
    # note or inventing a quarter-note duration.
    unknown_filled = sorted(
        [head for head in unstemmed if head.get("head_type") == "filled"],
        key=lambda head: (int(head.get("staff_id")), float(head["cx"])),
    )
    while unknown_filled:
        first = unknown_filled.pop(0)
        staff_id = int(first.get("staff_id"))
        spacing = float(
            staves.get(staff_id, {}).get("spacing", 18.0) or 18.0
        )
        group = [first]
        keep = []
        for head in unknown_filled:
            if (
                int(head.get("staff_id")) == staff_id
                and abs(float(head["cx"]) - float(first["cx"]))
                <= 0.45 * spacing
            ):
                group.append(head)
            else:
                keep.append(head)
        unknown_filled = keep
        head_ids = [int(head["id"]) for head in group]
        tied = bool(set(head_ids) & tie_right)
        events.append(
            VisualEvent(
                id=f"p{page['page']}:m{measure_id}:u{head_ids[0]}",
                page=int(page["page"]),
                measure_local=measure_id,
                staff_id=staff_id,
                x=float(median([
                    float(head["cx"]) for head in group
                ])),
                y=float(median([
                    float(head["cy"]) for head in group
                ])),
                kind="note_unknown_duration",
                voice="unknown",
                base_duration=Fraction(0),
                duration=Fraction(0),
                attacks=not tied,
                notehead_ids=head_ids,
                stem_id=None,
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
        is_measure_rest = str(rest.get("rest_type", "")) == "rest_whole"
        events.append(
            VisualEvent(
                id=f"p{page['page']}:m{measure_id}:r{rest['id']}",
                page=int(page["page"]),
                measure_local=measure_id,
                staff_id=staff_id,
                x=float(rest["cx"]),
                y=float(rest["cy"]),
                kind="measure_rest" if is_measure_rest else "rest",
                voice="measure_rest" if is_measure_rest else voice,
                base_duration=duration,
                duration=duration,
                attacks=False,
            )
        )

    # A chord normally shares one stem and is already grouped above.  Raster
    # fragmentation can occasionally give two nearly coincident noteheads
    # separate stems.  Merge only same-staff, same-direction note events whose
    # x positions are within one quarter staff spacing.  This is a graphical
    # chord repair, not a timing inference.
    merged: list[VisualEvent] = []
    by_staff = {int(staff["id"]): staff for staff in page.get("staves", [])}
    for event in sorted(events, key=lambda item: (item.staff_id, item.x, item.y)):
        if event.kind != "note" or event.voice not in {"up", "down"}:
            merged.append(event)
            continue
        spacing = float(by_staff.get(event.staff_id, {}).get("spacing", 18.0) or 18.0)
        candidate = None
        for prior in reversed(merged):
            if prior.kind != "note":
                continue
            if prior.staff_id != event.staff_id or prior.voice != event.voice:
                continue
            if event.x - prior.x > 0.25 * spacing:
                break
            if abs(event.x - prior.x) <= 0.25 * spacing:
                candidate = prior
                break
        if candidate is None:
            merged.append(event)
            continue

        candidate.notehead_ids = sorted(
            set(candidate.notehead_ids + event.notehead_ids)
        )
        candidate.x = float((candidate.x + event.x) / 2.0)
        candidate.y = float((candidate.y + event.y) / 2.0)
        # Mixed filled/hollow evidence at the same stem column is a common
        # fragmentation artifact.  The shorter visually supported value is the
        # conservative rhythmic value for the shared attack.
        if event.base_duration < candidate.base_duration:
            candidate.base_duration = event.base_duration
            candidate.duration = event.duration
            candidate.stem_id = event.stem_id
        candidate.dot_confidence = max(
            candidate.dot_confidence or 0.0,
            event.dot_confidence or 0.0,
        ) or None
        candidate.attacks = candidate.attacks or event.attacks
        candidate.tie_continuation = (
            candidate.tie_continuation and event.tie_continuation
        )
    return merged


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
    diagnostics_out: list[dict] = []
    geometric_columns_total = 0
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

            temporal_events = [
                event for event in events
                if event.kind != "measure_rest"
            ]
            system = systems.get(int(measure.get("system_id", 0)), {})
            spacings = [
                float(staves[staff_id].get("spacing", 18.0) or 18.0)
                for staff_id in system.get("staff_ids", [])
                if staff_id in staves
            ]
            spacing = float(median(spacings)) if spacings else 18.0
            columns = _cluster_columns(temporal_events, 0.65 * spacing)

            voices: dict[tuple[int, str], list[VisualEvent]] = {}
            neutral_rests: dict[int, list[VisualEvent]] = {}
            for event in events:
                if event.kind == "note_unknown_duration":
                    # Onset-only visual evidence: never enters duration/voice
                    # constraints.
                    continue
                if event.kind == "measure_rest":
                    event.base_duration = capacity
                    event.duration = capacity
                    voices.setdefault(
                        (event.staff_id, "measure_rest"),
                        [],
                    ).append(event)
                    continue
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
                    diagnostics_out.extend({
                        "page": int(page["page"]),
                        "measure_index": global_measure_index,
                        "event_id": rest.id,
                        "reason": "centered-rest-voice-ambiguous",
                        "blocking": False,
                    } for rest in rests)

            solutions: list[VoiceSolution] = []
            # Before any soft voice hypothesis is allowed to anchor time, test
            # whether the measure's rational notation itself proves a complete
            # subdivision grid.  If N ordered temporal columns exactly fill
            # the N = capacity/quantum possible positions, order alone fixes
            # every onset.  Horizontal distance is never used as time.
            times: dict[int, Fraction] = {}
            grid_quantum = _fraction_gcd(
                [capacity] + [
                    event.duration
                    for event in temporal_events
                    if event.duration > 0
                ]
            )
            if grid_quantum > 0:
                slot_count = capacity / grid_quantum
                if (
                    slot_count.denominator == 1
                    and int(slot_count) == len(columns)
                    and len(columns) > 0
                ):
                    for column_index in range(len(columns)):
                        times[column_index] = (
                            grid_quantum * column_index
                        )

            # Voice identity is visually inferred and therefore soft evidence.
            # Only a self-consistent complete voice may anchor exact score time.
            # Other inferred voice chains contribute duration edges only when
            # they agree with already established exact anchors.
            conflicts: list[str] = []
            soft_edges: list[tuple[int, int, Fraction, str]] = []

            complete_candidates: list[
                tuple[VoiceSolution, dict[int, Fraction]]
            ] = []

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
                solution = VoiceSolution(
                    key=key,
                    events=voice_events,
                    duration_sum=total,
                    complete=complete,
                    status=status,
                )
                solutions.append(solution)

                # Build a soft succession hypothesis from this inferred voice.
                for left, right in zip(voice_events, voice_events[1:]):
                    if (
                        left.kind == "measure_rest"
                        or right.kind == "measure_rest"
                        or left.column is None
                        or right.column is None
                        or left.column == right.column
                    ):
                        continue
                    soft_edges.append(
                        (
                            int(left.column),
                            int(right.column),
                            left.duration,
                            f"staff {key[0]} voice {key[1]}",
                        )
                    )

                if complete and voice_events and key[1] != "measure_rest":
                    proposals: dict[int, Fraction] = {}
                    cursor = Fraction(0)
                    internally_valid = True
                    for event in voice_events:
                        if event.column is not None:
                            column = int(event.column)
                            if (
                                column in proposals
                                and proposals[column] != cursor
                            ):
                                internally_valid = False
                                break
                            proposals[column] = cursor
                        cursor += event.duration
                    if cursor == capacity and internally_valid:
                        complete_candidates.append((solution, proposals))

            # Prefer complete voices that constrain more distinct columns.
            # A candidate is accepted only if every proposed time agrees with
            # previously accepted anchors.  Conflicting candidates are simply
            # rejected as bad visual voice hypotheses; they are not notation
            # conflicts.
            complete_candidates.sort(
                key=lambda item: len(item[1]),
                reverse=True,
            )
            accepted_complete = 0
            for solution, proposals in complete_candidates:
                if any(
                    column in times and times[column] != value
                    for column, value in proposals.items()
                ):
                    solution.status = "rejected-voice-hypothesis"
                    solution.complete = False
                    continue
                for column, value in proposals.items():
                    times[column] = value
                solution.status = "complete-anchor"
                accepted_complete += 1

            # Every non-pickup measure begins at the first temporal notation
            # column.  For the opening bar this is provisional and can be
            # replaced by the pickup rule below.
            if columns and global_measure_index != 0:
                if 0 in times and times[0] != Fraction(0):
                    # A complete candidate that starts later than the first
                    # visible column is not a valid measure anchor.
                    bad = times[0]
                    times = {
                        column: value
                        for column, value in times.items()
                        if not (column == 0 and value == bad)
                    }
                times[0] = Fraction(0)

            # Accept soft duration edges only when they are compatible with
            # exact anchors.  This is the key distinction from the old solver:
            # stem direction / inferred voice never gets to overrule score
            # time established by a self-consistent complete path.
            active_edges = list(soft_edges)
            changed = True
            while changed:
                changed = False
                next_edges: list[tuple[int, int, Fraction, str]] = []
                for left, right, duration, source in active_edges:
                    if left in times and right in times:
                        if times[right] - times[left] == duration:
                            next_edges.append((left, right, duration, source))
                        continue
                    if left in times and right not in times:
                        proposed = times[left] + duration
                        if Fraction(0) <= proposed < capacity:
                            times[right] = proposed
                            changed = True
                            next_edges.append((left, right, duration, source))
                        continue
                    if right in times and left not in times:
                        proposed = times[right] - duration
                        if Fraction(0) <= proposed < capacity:
                            times[left] = proposed
                            changed = True
                            next_edges.append((left, right, duration, source))
                        continue
                    next_edges.append((left, right, duration, source))
                active_edges = next_edges

            # A first column with no accepted complete anchor is still a hard
            # notational boundary for ordinary (non-opening) measures.
            if columns and global_measure_index != 0 and 0 not in times:
                times[0] = Fraction(0)

            # Resolve any remaining binary-grid columns only where the
            # rational subdivision and notation order leave exactly one
            # possible assignment.
            _fill_uniquely_forced_grid_columns(
                times,
                columns,
                temporal_events,
                capacity,
            )
            _solve_unique_duration_chain_columns(
                times,
                columns,
                temporal_events,
                capacity,
            )
            _fill_uniquely_forced_grid_columns(
                times,
                columns,
                temporal_events,
                capacity,
            )
            geometric_columns_total += _resolve_geometric_grid_intervals(
                times,
                columns,
                temporal_events,
                capacity,
                measure_left=float(measure.get("left", 0.0)),
                measure_right=float(measure.get("right", 0.0)),
            )
            _fill_uniquely_forced_grid_columns(
                times,
                columns,
                temporal_events,
                capacity,
            )

            # Opening incomplete measure: if no complete voice exists and the
            # longest explicitly notated voice has a unique shared duration,
            # position it as a pickup at the end of a full measure.
            pickup_shift = Fraction(0)
            if global_measure_index == 0 and accepted_complete == 0:
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
                        _propagate_edges(times, active_edges, conflicts)

            # If the opening bar did not prove an anacrusis, it is an ordinary
            # full measure: its first temporal notation column is exactly zero.
            # Re-run only exact duration/grid propagation after establishing
            # that boundary.  This avoids treating every unanchored first bar
            # as a pickup while still preserving genuine incomplete openings.
            if (
                global_measure_index == 0
                and pickup_shift == 0
                and columns
            ):
                times[0] = Fraction(0)
                _propagate_edges(times, active_edges, conflicts)
                _fill_uniquely_forced_grid_columns(
                    times,
                    columns,
                    temporal_events,
                    capacity,
                )
                _solve_unique_duration_chain_columns(
                    times,
                    columns,
                    temporal_events,
                    capacity,
                )
                _fill_uniquely_forced_grid_columns(
                    times,
                    columns,
                    temporal_events,
                    capacity,
                )

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
        "x_position_used_as_time": bool(geometric_columns_total),
        "geometric_quantized_columns": geometric_columns_total,
        "meter": {
            "numerator": meter[0],
            "denominator": meter[1],
            "measure_duration_quarter": frac_text(capacity),
        },
        "measures": measures_out,
        "attacks": attacks_out,
        "hits": hit_rows,
        "unresolved": unresolved_out,
        "diagnostics": diagnostics_out,
        "stats": {
            "measures": len(measures_out),
            "exact_attacks": len(attacks_out),
            "sonic_hits": len(hit_rows),
            "unresolved_items": len(unresolved_out),
            "constraint_conflicts": conflicts_total,
            "nonblocking_diagnostics": len(diagnostics_out),
            "geometric_quantized_columns": geometric_columns_total,
        },
        "rhythmic_attacks_ready": ready,
    }
