from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import re

from lxml import etree

CROSS_STAFF_INTERLINE_FACTOR = 1.50
CROSS_STAFF_SPACING_FACTOR = 0.65
MIN_CROSS_STAFF_TOLERANCE_INTERLINES = 0.55
AMBIGUITY_MARGIN = 0.15
DISPLAY_MIN_GAP_INTERLINES = 0.20


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def x_overlaps(self, other: "Box") -> bool:
        return min(self.right, other.right) > max(self.left, other.left)


@dataclass
class Chord:
    id: str
    staff: str
    box: Box
    attack_head_xs: list[float]
    all_head_xs: list[float]
    measure_index: int
    timed: bool = False

    @property
    def anchor_x(self) -> float:
        values = self.attack_head_xs or self.all_head_xs
        return float(median(values)) if values else self.box.cx


@dataclass
class Event:
    measure_index: int
    time: Fraction | None
    slot_x: float | None
    chords: list[Chord] = field(default_factory=list)
    recovered: bool = False

    @property
    def observed_x(self) -> float:
        if self.chords:
            return float(median([chord.anchor_x for chord in self.chords]))
        if self.slot_x is not None:
            return float(self.slot_x)
        raise ValueError("Event has neither chord geometry nor slot geometry")


@dataclass(frozen=True)
class Strike:
    page_index: int
    system_index: int
    measure_index: int
    x: float
    system_top: float
    system_bottom: float
    omr_width: float
    omr_height: float
    recovered: bool = False


@dataclass(frozen=True)
class Diagnostics:
    base_events: int
    recovered_events: int
    reconciled_chords: int
    sounding_chords: int
    assigned_chords: int


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _sheet_number(name: str) -> int:
    match = re.search(r"sheet#(\d+)", name, re.I)
    return int(match.group(1)) if match else 10**9


def _box(node) -> Box | None:
    bounds = next((child for child in node if _local(child.tag) == "bounds"), None)
    if bounds is None:
        return None
    try:
        return Box(
            float(bounds.get("x")),
            float(bounds.get("y")),
            float(bounds.get("w")),
            float(bounds.get("h")),
        )
    except (TypeError, ValueError):
        return None


def _system_vertical_bounds(system, page_height: float) -> tuple[float, float]:
    ys: list[float] = []
    for staff in (node for node in system.iter() if _local(node.tag) == "staff"):
        for line in (node for node in staff.iter() if _local(node.tag) == "line"):
            for point in line:
                if _local(point.tag) == "point" and point.get("y") is not None:
                    ys.append(float(point.get("y")))
    return (min(ys), max(ys)) if ys else (0.0, page_height)


def _interline(root) -> float:
    for scale in (node for node in root.iter() if _local(node.tag) == "scale"):
        interline = next(
            (child for child in scale if _local(child.tag) == "interline"),
            None,
        )
        if interline is not None and interline.get("main"):
            value = float(interline.get("main"))
            if value > 0:
                return value
    return 20.0


def _assign_measure_index(
    chord_box: Box,
    stack_bounds: list[tuple[float, float]],
) -> int:
    containing: list[tuple[float, int]] = []
    for index, (left, right) in enumerate(stack_bounds):
        if left <= chord_box.cx <= right:
            containing.append((abs(chord_box.cx - (left + right) / 2.0), index))
    if containing:
        return min(containing)[1]

    distances: list[tuple[float, int]] = []
    for index, (left, right) in enumerate(stack_bounds):
        if chord_box.cx < left:
            distance = left - chord_box.cx
        elif chord_box.cx > right:
            distance = chord_box.cx - right
        else:
            distance = 0.0
        distances.append((distance, index))
    distance, index = min(distances)
    if distance <= max(2.0, chord_box.w * 0.20):
        return index
    raise ValueError(
        f"Sounding chord at x={chord_box.cx:.1f} does not belong to any measure"
    )


def _nearest_spacing_for_staff(
    events: list[Event],
    staff: str,
    target_time: Fraction | None,
) -> float | None:
    points: list[tuple[Fraction, float]] = []
    for event in events:
        if event.time is None:
            continue
        staff_x = [chord.anchor_x for chord in event.chords if chord.staff == staff]
        if staff_x:
            points.append((event.time, float(median(staff_x))))
    points.sort(key=lambda item: item[0])
    if len(points) < 2:
        return None

    spacings: list[tuple[float, float]] = []
    for (time_a, x_a), (time_b, x_b) in zip(points, points[1:]):
        if time_a == time_b:
            continue
        dx = abs(x_b - x_a)
        if dx <= 0:
            continue
        if target_time is None:
            time_distance = 0.0
        else:
            midpoint = (time_a + time_b) / 2
            time_distance = abs(float(midpoint - target_time))
        spacings.append((time_distance, dx))
    if not spacings:
        return None
    spacings.sort(key=lambda item: item[0])
    return float(median([dx for _, dx in spacings[: min(3, len(spacings))]]))


def _candidate_merge_score(
    chord: Chord,
    event: Event,
    all_events: list[Event],
    interline: float,
) -> float | None:
    scores: list[float] = []

    for member in event.chords:
        if member.staff == chord.staff and chord.box.x_overlaps(member.box):
            scores.append(0.0)

    for member in event.chords:
        if member.staff == chord.staff:
            continue
        local_spacing = _nearest_spacing_for_staff(
            all_events, member.staff, event.time
        )
        tolerance = CROSS_STAFF_INTERLINE_FACTOR * interline
        if local_spacing is not None:
            tolerance = min(
                tolerance,
                CROSS_STAFF_SPACING_FACTOR * local_spacing,
            )
        tolerance = max(
            tolerance,
            MIN_CROSS_STAFF_TOLERANCE_INTERLINES * interline,
        )
        difference = abs(chord.anchor_x - member.anchor_x)
        if difference <= tolerance:
            scores.append(difference / max(tolerance, 1e-9))

    return min(scores) if scores else None


def _fit_monotone_positions(
    events: list[Event],
    left: float,
    right: float,
    interline: float,
) -> list[float]:
    """Fit one left-to-right display position per exact symbolic onset.

    Event identity and order are fixed before this function runs. Geometry can
    choose only where to draw the strike. Isotonic regression repairs displaced
    polyphonic engraving without creating, deleting, merging, or retiming hits.
    """
    if not events:
        return []

    min_gap = max(2.0, interline * DISPLAY_MIN_GAP_INTERLINES)
    observations = [event.observed_x for event in events]
    transformed = [
        observation - index * min_gap
        for index, observation in enumerate(observations)
    ]

    blocks: list[list[float]] = []
    for index, value in enumerate(transformed):
        blocks.append([float(index), float(index), value, 1.0])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right_block = blocks.pop()
            left_block = blocks.pop()
            weight = left_block[3] + right_block[3]
            mean = (
                left_block[2] * left_block[3]
                + right_block[2] * right_block[3]
            ) / weight
            blocks.append([left_block[0], right_block[1], mean, weight])

    fitted = [0.0] * len(events)
    for start, end, mean, _weight in blocks:
        for index in range(int(start), int(end) + 1):
            fitted[index] = mean + index * min_gap

    floor = left + 1.0
    ceiling = right - 1.0
    if fitted[0] < floor:
        shift = floor - fitted[0]
        fitted = [x + shift for x in fitted]
    if fitted[-1] > ceiling:
        shift = fitted[-1] - ceiling
        fitted = [x - shift for x in fitted]

    if any(b <= a for a, b in zip(fitted, fitted[1:])):
        raise ValueError("Display registration is not strictly left-to-right")
    return fitted


def extract_hit_strikes(
    omr_path: str | Path,
) -> tuple[int, list[Strike], Diagnostics]:
    """Build the global hit set from a saved Audiveris project.

    Exact symbolic BEGIN onsets establish timed hits. Semantic note/chord
    geometry independently audits those hits so recognized attacks cannot
    silently disappear. Omitted semantic chords are reconciled to existing
    events only with independent simultaneity evidence; otherwise they become
    recovered hits. Ambiguous reconciliation fails closed instead of guessing.
    """
    omr_path = Path(omr_path)
    strikes: list[Strike] = []
    base_event_count = 0
    recovered_event_count = 0
    reconciled_chord_count = 0
    sounding_chord_count = 0
    assigned_chord_ids: set[tuple[str, str]] = set()
    global_measure = 0

    with ZipFile(omr_path) as archive:
        members = sorted(
            [
                name
                for name in archive.namelist()
                if re.search(r"sheet#\d+/sheet#\d+\.xml$", name, re.I)
            ],
            key=lambda name: (_sheet_number(name), name.lower()),
        )
        if not members:
            raise ValueError("Audiveris project contains no sheet XML")

        page_index = 0
        for member_name in members:
            root = etree.fromstring(archive.read(member_name))
            picture = next(
                (node for node in root.iter() if _local(node.tag) == "picture"),
                None,
            )
            if picture is None:
                raise ValueError(f"{member_name}: missing picture geometry")
            width = float(picture.get("width") or 0)
            height = float(picture.get("height") or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f"{member_name}: invalid picture geometry")

            interline = _interline(root)
            by_id = {
                node.get("id"): node
                for node in root.iter()
                if node.get("id")
            }
            head_positions: dict[str, tuple[float, float]] = {}
            chord_head_ids: dict[str, list[str]] = defaultdict(list)
            tied_right_heads: set[str] = set()

            for relation in (
                node for node in root.iter() if _local(node.tag) == "relation"
            ):
                child = next(iter(relation), None)
                if child is None:
                    continue
                kind = _local(child.tag)
                source = relation.get("source")
                target = relation.get("target")

                if (
                    kind == "containment"
                    and source in by_id
                    and target in by_id
                    and _local(by_id[source].tag) == "head-chord"
                    and _local(by_id[target].tag) == "head"
                ):
                    head_box = _box(by_id[target])
                    if head_box is not None:
                        head_positions[target] = (head_box.cx, head_box.cy)
                        chord_head_ids[source].append(target)
                elif (
                    kind == "slur-head"
                    and (child.get("side") or "").upper() == "RIGHT"
                ):
                    slur = by_id.get(source)
                    if (
                        slur is not None
                        and _local(slur.tag) == "slur"
                        and (slur.get("tie") or "").lower() == "true"
                        and target
                    ):
                        tied_right_heads.add(target)

            pages = [node for node in root.iter() if _local(node.tag) == "page"] or [root]
            for page in pages:
                systems = [node for node in page if _local(node.tag) == "system"]
                if not systems:
                    systems = [
                        node for node in page.iter() if _local(node.tag) == "system"
                    ]

                for system_index, system in enumerate(systems):
                    system_top, system_bottom = _system_vertical_bounds(system, height)
                    stacks = [node for node in system if _local(node.tag) == "stack"]
                    parts = [node for node in system if _local(node.tag) == "part"]
                    if not stacks:
                        continue

                    stack_bounds = [
                        (
                            float(stack.get("left") or 0),
                            float(stack.get("right") or 0),
                        )
                        for stack in stacks
                    ]
                    staff_ids = {
                        staff.get("id")
                        for staff in system.iter()
                        if _local(staff.tag) == "staff" and staff.get("id")
                    }

                    system_chords_by_measure: dict[int, dict[str, Chord]] = defaultdict(dict)
                    for chord_id, chord_node in by_id.items():
                        if (
                            _local(chord_node.tag) != "head-chord"
                            or chord_node.get("staff") not in staff_ids
                        ):
                            continue
                        chord_box = _box(chord_node)
                        if chord_box is None:
                            continue
                        heads = chord_head_ids.get(chord_id, [])
                        all_head_xs = [
                            head_positions[head_id][0]
                            for head_id in heads
                            if head_id in head_positions
                        ]
                        attack_head_xs = [
                            head_positions[head_id][0]
                            for head_id in heads
                            if head_id in head_positions and head_id not in tied_right_heads
                        ]
                        if not attack_head_xs:
                            continue

                        measure_offset = _assign_measure_index(chord_box, stack_bounds)
                        measure_index = global_measure + measure_offset + 1
                        chord = Chord(
                            id=chord_id,
                            staff=chord_node.get("staff") or "",
                            box=chord_box,
                            attack_head_xs=attack_head_xs,
                            all_head_xs=all_head_xs,
                            measure_index=measure_index,
                            timed=False,
                        )
                        system_chords_by_measure[measure_offset][chord_id] = chord
                        sounding_chord_count += 1

                    for stack_index, stack in enumerate(stacks):
                        measure_index = global_measure + stack_index + 1
                        left, right = stack_bounds[stack_index]
                        slot_nodes = {
                            slot.get("id"): slot
                            for slot in stack
                            if _local(slot.tag) == "slot" and slot.get("id")
                        }

                        events_by_time: dict[Fraction, Event] = {}
                        represented_ids: set[str] = set()

                        for part in parts:
                            measures = [
                                node for node in part if _local(node.tag) == "measure"
                            ]
                            if stack_index >= len(measures):
                                continue
                            measure = measures[stack_index]
                            for entry in (
                                node for node in measure.iter() if _local(node.tag) == "entry"
                            ):
                                key_node = next(
                                    (child for child in entry if _local(child.tag) == "key"),
                                    None,
                                )
                                value_node = next(
                                    (child for child in entry if _local(child.tag) == "value"),
                                    None,
                                )
                                if (
                                    key_node is None
                                    or value_node is None
                                    or (value_node.get("status") or "").upper() != "BEGIN"
                                ):
                                    continue

                                chord_id = value_node.get("chord")
                                chord = system_chords_by_measure[stack_index].get(chord_id)
                                if chord is None:
                                    continue

                                slot_id = (key_node.text or "").strip()
                                slot = slot_nodes.get(slot_id)
                                if slot is None:
                                    raise ValueError(
                                        f"{member_name}: BEGIN chord {chord_id} has no slot geometry"
                                    )
                                try:
                                    time_offset = Fraction(slot.get("time-offset") or "0")
                                except Exception as exc:
                                    raise ValueError(f"{member_name}: invalid slot time") from exc

                                event = events_by_time.setdefault(
                                    time_offset,
                                    Event(
                                        measure_index=measure_index,
                                        time=time_offset,
                                        slot_x=left + float(slot.get("x-offset") or 0),
                                    ),
                                )
                                chord.timed = True
                                event.chords.append(chord)
                                represented_ids.add(chord_id)
                                assigned_chord_ids.add((member_name, chord_id))

                        base_events = [events_by_time[time] for time in sorted(events_by_time)]
                        base_event_count += len(base_events)

                        unvoiced = [
                            chord
                            for chord_id, chord in system_chords_by_measure[stack_index].items()
                            if chord_id not in represented_ids
                        ]

                        unmatched: list[Chord] = []
                        for chord in sorted(
                            unvoiced,
                            key=lambda item: (item.anchor_x, item.staff, item.id),
                        ):
                            candidates: list[tuple[float, Event]] = []
                            for event in base_events:
                                score = _candidate_merge_score(
                                    chord,
                                    event,
                                    base_events,
                                    interline,
                                )
                                if score is not None:
                                    candidates.append((score, event))
                            candidates.sort(key=lambda item: item[0])

                            if not candidates:
                                unmatched.append(chord)
                                continue

                            if (
                                len(candidates) > 1
                                and candidates[1][0] - candidates[0][0] < AMBIGUITY_MARGIN
                            ):
                                raise ValueError(
                                    f"Measure {measure_index}: ambiguous onset assignment for chord {chord.id}"
                                )

                            candidates[0][1].chords.append(chord)
                            reconciled_chord_count += 1
                            assigned_chord_ids.add((member_name, chord.id))

                        recovered_events: list[Event] = []
                        remaining = sorted(
                            unmatched,
                            key=lambda item: (item.anchor_x, item.staff, item.id),
                        )
                        while remaining:
                            seed = remaining.pop(0)
                            group = [seed]
                            changed = True
                            while changed:
                                changed = False
                                for chord in list(remaining):
                                    should_merge = False
                                    for member in group:
                                        if chord.staff == member.staff:
                                            continue
                                        observations = sorted(
                                            event.observed_x for event in base_events
                                        )
                                        spacings = [
                                            b - a
                                            for a, b in zip(observations, observations[1:])
                                            if b > a
                                        ]
                                        local_spacing = (
                                            float(median(spacings))
                                            if spacings
                                            else 2.0 * interline
                                        )
                                        tolerance = max(
                                            MIN_CROSS_STAFF_TOLERANCE_INTERLINES * interline,
                                            min(
                                                CROSS_STAFF_INTERLINE_FACTOR * interline,
                                                CROSS_STAFF_SPACING_FACTOR * local_spacing,
                                            ),
                                        )
                                        if abs(chord.anchor_x - member.anchor_x) <= tolerance:
                                            should_merge = True
                                            break
                                    if should_merge:
                                        group.append(chord)
                                        remaining.remove(chord)
                                        changed = True

                            event = Event(
                                measure_index=measure_index,
                                time=None,
                                slot_x=None,
                                chords=group,
                                recovered=True,
                            )
                            recovered_events.append(event)
                            recovered_event_count += 1
                            for chord in group:
                                assigned_chord_ids.add((member_name, chord.id))

                        positions = _fit_monotone_positions(
                            base_events,
                            left,
                            right,
                            interline,
                        )
                        for event, x in zip(base_events, positions):
                            strikes.append(
                                Strike(
                                    page_index=page_index,
                                    system_index=system_index,
                                    measure_index=measure_index,
                                    x=x,
                                    system_top=system_top,
                                    system_bottom=system_bottom,
                                    omr_width=width,
                                    omr_height=height,
                                    recovered=False,
                                )
                            )
                        for event in recovered_events:
                            strikes.append(
                                Strike(
                                    page_index=page_index,
                                    system_index=system_index,
                                    measure_index=measure_index,
                                    x=event.observed_x,
                                    system_top=system_top,
                                    system_bottom=system_bottom,
                                    omr_width=width,
                                    omr_height=height,
                                    recovered=True,
                                )
                            )

                    global_measure += len(stacks)
                page_index += 1

    if not strikes:
        raise ValueError("No sounding note attacks were found")

    if len(assigned_chord_ids) != sounding_chord_count:
        raise ValueError(
            "Semantic-chord coverage invariant failed: "
            f"{len(assigned_chord_ids)} of {sounding_chord_count} sounding chords assigned"
        )

    strikes.sort(
        key=lambda strike: (
            strike.page_index,
            strike.system_index,
            strike.measure_index,
            strike.x,
        )
    )
    diagnostics = Diagnostics(
        base_events=base_event_count,
        recovered_events=recovered_event_count,
        reconciled_chords=reconciled_chord_count,
        sounding_chords=sounding_chord_count,
        assigned_chords=len(assigned_chord_ids),
    )
    return len(strikes), strikes, diagnostics
