from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile
import re

from lxml import etree

RECOVERY_MERGE_TOLERANCE_PX = 8.0
MIN_EVENT_SEPARATION_PX = 12.0
SYNTHETIC_TARGET_COST = 45.0


@dataclass(frozen=True)
class Strike:
    page_index: int
    system_index: int
    x: float
    system_top: float
    system_bottom: float
    omr_width: float
    omr_height: float


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _sheet_number(name: str) -> int:
    match = re.search(r'sheet#(\d+)', name, re.I)
    return int(match.group(1)) if match else 10**9


def _bounds_center(node) -> tuple[float, float] | None:
    bounds = next((c for c in node if _local(c.tag).lower() == 'bounds'), None)
    if bounds is None:
        return None
    try:
        return (
            float(bounds.get('x')) + float(bounds.get('w')) / 2.0,
            float(bounds.get('y')) + float(bounds.get('h')) / 2.0,
        )
    except Exception:
        return None


def _system_vertical_bounds(system, page_height: float) -> tuple[float, float]:
    ys: list[float] = []
    for staff in (e for e in system.iter() if _local(e.tag).lower() == 'staff'):
        for line in (e for e in staff.iter() if _local(e.tag).lower() == 'line'):
            for point in line:
                if _local(point.tag).lower() != 'point':
                    continue
                try:
                    ys.append(float(point.get('y')))
                except Exception:
                    pass
    return (min(ys), max(ys)) if ys else (0.0, page_height)


def _spread_targets(
    xs: list[float],
    left: float,
    right: float,
    minsep: float = MIN_EVENT_SEPARATION_PX,
) -> list[float]:
    if not xs:
        return []
    out = sorted(float(x) for x in xs)
    for i in range(1, len(out)):
        if out[i] < out[i - 1] + minsep:
            out[i] = out[i - 1] + minsep
    ceiling = float(right) - 2.0
    if out[-1] > ceiling:
        shift = out[-1] - ceiling
        out = [x - shift for x in out]
        for i in range(len(out) - 2, -1, -1):
            if out[i] > out[i + 1] - minsep:
                out[i] = out[i + 1] - minsep
    floor = float(left) + 2.0
    if out[0] < floor:
        shift = floor - out[0]
        out = [x + shift for x in out]
    return out


def _assign_positions(
    targets: list[float],
    candidate_lists: list[list[float]],
    minsep: float = MIN_EVENT_SEPARATION_PX,
) -> list[float]:
    """Choose one display x for each distinct musical hit.

    A real attacking notehead x is preferred. When old engraving places two
    different score-times on nearly the same visual x, or when Audiveris gives
    a non-monotone slot layout, the repaired rhythmic target is used instead.
    This preserves one visible strike per distinct hit without inventing or
    deleting events.
    """
    if len(targets) != len(candidate_lists):
        raise ValueError('target/candidate mismatch')

    options: list[list[tuple[float, float, bool]]] = []
    for target, candidates in zip(targets, candidate_lists):
        values: list[tuple[float, float, bool]] = []
        for x in sorted(set(round(float(x), 6) for x in candidates)):
            values.append((x, abs(x - target), False))
        values.append((float(target), SYNTHETIC_TARGET_COST, True))

        best_by_x: dict[float, tuple[float, bool]] = {}
        for x, cost, synthetic in values:
            if x not in best_by_x or cost < best_by_x[x][0]:
                best_by_x[x] = (cost, synthetic)
        options.append([(x, *best_by_x[x]) for x in sorted(best_by_x)])

    states: dict[int, tuple[float, list[float]]] = {
        j: (cost, [x]) for j, (x, cost, _synthetic) in enumerate(options[0])
    }

    for i in range(1, len(options)):
        new_states: dict[int, tuple[float, list[float]]] = {}
        for j, (x, cost, _synthetic) in enumerate(options[i]):
            best: tuple[float, list[float]] | None = None
            for _previous_index, (previous_cost, path) in states.items():
                if x - path[-1] < minsep - 1e-9:
                    continue
                candidate = (previous_cost + cost, path + [x])
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is not None:
                new_states[j] = best
        if not new_states:
            return list(targets)
        states = new_states

    return min(states.values(), key=lambda item: item[0])[1]


def extract_hit_strikes(omr_path: str | Path) -> tuple[int, list[Strike]]:
    """Return exactly one strike for every recognized new sounding hit.

    Timed hit identity comes from semantic BEGIN head-chords grouped by exact
    Audiveris rhythmic slot. Simultaneous attacks across voices/staves therefore
    merge to one hit. Rests do not qualify. A chord whose heads are all RIGHT
    endpoints of semantic ties is a continuation and does not qualify.

    Audiveris can recognize a valid sounding head-chord but omit it from all
    voice BEGIN records. Those semantic attacks are recovered from notehead
    geometry. If ANY head in an unvoiced chord coincides with a head already
    assigned to a timed event, the whole chord is treated as simultaneous with
    that event rather than counted again.

    Display x is deliberately separate from hit identity. Old engraving can
    place different musical times on nearly the same x, or simultaneous voices
    at different x values. The display layer therefore repairs slot ordering,
    keeps distinct hits visibly separated, and prefers real attacking notehead
    positions whenever that does not collapse/reverse musical time.
    """
    path = Path(omr_path)
    strikes: list[Strike] = []

    with ZipFile(path) as zf:
        members = sorted(
            [n for n in zf.namelist() if re.search(r'sheet#\d+/sheet#\d+\.xml$', n, re.I)],
            key=lambda n: (_sheet_number(n), n.lower()),
        )
        if not members:
            raise ValueError('Audiveris project contains no sheet XML')

        page_index = 0
        for member in members:
            root = etree.fromstring(zf.read(member))
            by_id = {e.get('id'): e for e in root.iter() if e.get('id')}

            picture = next((e for e in root.iter() if _local(e.tag).lower() == 'picture'), None)
            if picture is None:
                raise ValueError(f'{member}: missing picture geometry')
            width = float(picture.get('width') or 0)
            height = float(picture.get('height') or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f'{member}: invalid picture geometry')

            chord_heads: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
            tied_right_heads: set[str] = set()

            for relation in (e for e in root.iter() if _local(e.tag).lower() == 'relation'):
                child = next(iter(relation), None)
                if child is None:
                    continue
                kind = _local(child.tag).lower()
                source = relation.get('source')
                target = relation.get('target')

                if kind == 'containment' and source in by_id and target in by_id:
                    source_node = by_id[source]
                    target_node = by_id[target]
                    if (
                        _local(source_node.tag).lower() == 'head-chord'
                        and _local(target_node.tag).lower() == 'head'
                    ):
                        center = _bounds_center(target_node)
                        if center is not None:
                            chord_heads[source].append((target, center[0], center[1]))

                elif kind == 'slur-head' and (child.get('side') or '').upper() == 'RIGHT':
                    slur = by_id.get(source)
                    if (
                        slur is not None
                        and _local(slur.tag).lower() == 'slur'
                        and (slur.get('tie') or '').lower() == 'true'
                        and target
                    ):
                        tied_right_heads.add(target)

            def new_heads(chord_id: str) -> list[tuple[str, float, float]]:
                return [h for h in chord_heads.get(chord_id, []) if h[0] not in tied_right_heads]

            pages = [e for e in root.iter() if _local(e.tag).lower() == 'page'] or [root]
            for page in pages:
                systems = [e for e in page if _local(e.tag).lower() == 'system']
                if not systems:
                    systems = [e for e in page.iter() if _local(e.tag).lower() == 'system']

                staff_to_system = {
                    staff.get('id'): system_index
                    for system_index, system in enumerate(systems)
                    for staff in system.iter()
                    if _local(staff.tag).lower() == 'staff' and staff.get('id')
                }

                represented_chords: set[str] = set()
                system_strike_x: dict[int, list[float]] = defaultdict(list)
                system_event_head_x: dict[int, list[float]] = defaultdict(list)

                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    stacks = [e for e in system if _local(e.tag).lower() == 'stack']
                    parts = [e for e in system if _local(e.tag).lower() == 'part']

                    for stack_index, stack in enumerate(stacks):
                        slot_nodes = {
                            slot.get('id'): slot
                            for slot in stack
                            if _local(slot.tag).lower() == 'slot' and slot.get('id')
                        }
                        slot_heads: dict[str, list[tuple[str, float, float]]] = defaultdict(list)

                        for part in parts:
                            measures = [e for e in part if _local(e.tag).lower() == 'measure']
                            if stack_index >= len(measures):
                                continue
                            measure = measures[stack_index]

                            for entry in (e for e in measure.iter() if _local(e.tag).lower() == 'entry'):
                                key_node = next((c for c in entry if _local(c.tag).lower() == 'key'), None)
                                value = next((c for c in entry if _local(c.tag).lower() == 'value'), None)
                                if key_node is None or value is None:
                                    continue
                                if (value.get('status') or '').upper() != 'BEGIN':
                                    continue

                                chord_id = value.get('chord')
                                chord = by_id.get(chord_id)
                                if chord is None or _local(chord.tag).lower() != 'head-chord':
                                    continue

                                heads = new_heads(chord_id)
                                if not heads:
                                    continue

                                represented_chords.add(chord_id)
                                slot_id = (key_node.text or '').strip()
                                if slot_id:
                                    slot_heads[slot_id].extend(heads)

                        left = float(stack.get('left') or 0)
                        right = float(stack.get('right') or left)
                        events: list[tuple[Fraction, int, float, list[float]]] = []

                        for slot_id, heads in slot_heads.items():
                            slot = slot_nodes.get(slot_id)
                            if slot is None or slot.get('x-offset') is None:
                                continue
                            slot_x = left + float(slot.get('x-offset'))
                            try:
                                time_offset = Fraction(slot.get('time-offset') or '0')
                            except Exception:
                                time_offset = Fraction(0)
                            try:
                                numeric_slot_id = int(slot_id)
                            except Exception:
                                numeric_slot_id = 10**9
                            events.append(
                                (time_offset, numeric_slot_id, slot_x, [h[1] for h in heads])
                            )

                        events.sort(key=lambda item: (item[0], item[1]))
                        if events:
                            targets = _spread_targets([event[2] for event in events], left, right)
                            positions = _assign_positions(
                                targets,
                                [event[3] for event in events],
                            )
                            for x, event in zip(positions, events):
                                strikes.append(
                                    Strike(
                                        page_index,
                                        system_index,
                                        x,
                                        top,
                                        bottom,
                                        width,
                                        height,
                                    )
                                )
                                system_strike_x[system_index].append(x)
                                system_event_head_x[system_index].extend(event[3])

                # Recover sounding semantic chords omitted from all timed BEGIN records.
                for chord_id, chord in by_id.items():
                    if _local(chord.tag).lower() != 'head-chord' or chord_id in represented_chords:
                        continue
                    heads = new_heads(chord_id)
                    if not heads:
                        continue

                    system_index = staff_to_system.get(chord.get('staff') or '')
                    if system_index is None:
                        continue

                    # If ANY head of this unvoiced chord shares the visual attack
                    # column of a timed event, the whole chord is already represented.
                    if any(
                        abs(head[1] - existing_head_x) <= RECOVERY_MERGE_TOLERANCE_PX
                        for head in heads
                        for existing_head_x in system_event_head_x[system_index]
                    ):
                        continue

                    chord_center = _bounds_center(chord)
                    target_x = (
                        chord_center[0]
                        if chord_center is not None
                        else sum(h[1] for h in heads) / len(heads)
                    )
                    x = min((h[1] for h in heads), key=lambda candidate: abs(candidate - target_x))

                    if any(
                        abs(x - prior_x) <= RECOVERY_MERGE_TOLERANCE_PX
                        for prior_x in system_strike_x[system_index]
                    ):
                        continue

                    top, bottom = _system_vertical_bounds(systems[system_index], height)
                    strikes.append(
                        Strike(
                            page_index,
                            system_index,
                            x,
                            top,
                            bottom,
                            width,
                            height,
                        )
                    )
                    system_strike_x[system_index].append(x)

                page_index += 1

    if not strikes:
        raise ValueError('No sounding note attacks were found')

    return len(strikes), strikes
