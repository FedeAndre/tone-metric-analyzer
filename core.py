from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import re

from lxml import etree


RECOVERY_MERGE_TOLERANCE_PX = 8.0


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


def extract_hit_strikes(omr_path: str | Path) -> tuple[int, list[Strike]]:
    """Return exactly one strike for every recognized new sounding hit.

    Primary identity comes from Audiveris semantic rhythmic slots. A BEGIN
    head-chord with at least one untied head makes its global slot a hit, and
    simultaneous voices/staves therefore merge to one strike.

    Audiveris can recognize a valid head-chord but fail to insert it into a
    voice. Such semantic head-chords are recovered directly from their
    notehead geometry instead of being silently discarded. Recovery is used
    only when a qualifying chord has no BEGIN voice entry; if its notehead is
    already essentially coincident with an existing strike, it is considered
    represented rather than duplicated.

    Rests and non-head-chords never create hits. Chords whose heads are all
    RIGHT endpoints of semantic ties are continuations and never create hits.
    MusicXML is not used.
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
            picture = next((e for e in root.iter() if _local(e.tag).lower() == 'picture'), None)
            if picture is None:
                raise ValueError(f'{member}: missing picture geometry')
            width = float(picture.get('width') or 0)
            height = float(picture.get('height') or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f'{member}: invalid picture geometry')

            by_id = {e.get('id'): e for e in root.iter() if e.get('id')}
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

                staff_to_system: dict[str, int] = {}
                for system_index, system in enumerate(systems):
                    for staff in (e for e in system.iter() if _local(e.tag).lower() == 'staff'):
                        staff_id = staff.get('id')
                        if staff_id:
                            staff_to_system[staff_id] = system_index

                represented_chords: set[str] = set()
                system_strike_x: dict[int, list[float]] = defaultdict(list)

                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    stacks = [e for e in system if _local(e.tag).lower() == 'stack']
                    parts = [e for e in system if _local(e.tag).lower() == 'part']

                    for stack_index, stack in enumerate(stacks):
                        sounding_slot_ids: set[str] = set()

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
                                if not new_heads(chord_id):
                                    continue

                                represented_chords.add(chord_id)
                                slot_id = (key_node.text or '').strip()
                                if slot_id:
                                    sounding_slot_ids.add(slot_id)

                        slot_nodes = {
                            slot.get('id'): slot
                            for slot in stack
                            if _local(slot.tag).lower() == 'slot' and slot.get('id')
                        }
                        left = float(stack.get('left') or 0)

                        for slot_id in sorted(sounding_slot_ids, key=lambda x: int(x)):
                            slot = slot_nodes.get(slot_id)
                            if slot is None:
                                raise ValueError(f'{member}: sounding slot {slot_id} has no stack slot geometry')
                            x_offset = slot.get('x-offset')
                            if x_offset is None:
                                raise ValueError(f'{member}: sounding slot {slot_id} has no x-offset')
                            x = left + float(x_offset)
                            strikes.append(Strike(page_index, system_index, x, top, bottom, width, height))
                            system_strike_x[system_index].append(x)

                # Recover semantic note attacks Audiveris recognized but omitted
                # from all voice-entry BEGIN records.
                for chord_id, chord in by_id.items():
                    if _local(chord.tag).lower() != 'head-chord' or chord_id in represented_chords:
                        continue
                    heads = new_heads(chord_id)
                    if not heads:
                        continue

                    staff_id = chord.get('staff')
                    system_index = staff_to_system.get(staff_id or '')
                    if system_index is None:
                        raise ValueError(
                            f'{member}: unvoiced sounding head-chord {chord_id} has no system staff mapping'
                        )

                    xs = sorted(h[1] for h in heads)
                    x = xs[len(xs) // 2]
                    existing = system_strike_x[system_index]
                    if any(abs(x - prior_x) <= RECOVERY_MERGE_TOLERANCE_PX for prior_x in existing):
                        continue

                    top, bottom = _system_vertical_bounds(systems[system_index], height)
                    strikes.append(Strike(page_index, system_index, x, top, bottom, width, height))
                    existing.append(x)

                page_index += 1

    if not strikes:
        raise ValueError('No sounding note attacks were found')

    return len(strikes), strikes
