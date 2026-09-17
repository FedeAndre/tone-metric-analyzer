from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import re

from lxml import etree


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
    """Return exactly one vertical strike for every global new sounding hit.

    Hit identity comes only from Audiveris's semantic rhythmic slots:
      * a voice entry with status=BEGIN starts a chord at a global slot;
      * only head-chords can create hits;
      * a chord whose heads are all RIGHT endpoints of semantic ties is a
        continuation and creates no hit;
      * if any head in the chord is untied, that slot contains a new attack;
      * simultaneous attacks in any number of voices/staves merge into the
        same global slot and therefore exactly one strike.

    The strike x-coordinate is the global rhythmic slot position from the OMR
    stack. Individual notehead x-coordinates never create additional strikes.
    MusicXML is not used for hit identity or strike placement.
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
            chord_heads: dict[str, set[str]] = defaultdict(set)
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
                        chord_heads[source].add(target)

                elif kind == 'slur-head' and (child.get('side') or '').upper() == 'RIGHT':
                    slur = by_id.get(source)
                    if (
                        slur is not None
                        and _local(slur.tag).lower() == 'slur'
                        and (slur.get('tie') or '').lower() == 'true'
                        and target
                    ):
                        tied_right_heads.add(target)

            pages = [e for e in root.iter() if _local(e.tag).lower() == 'page'] or [root]
            for page in pages:
                systems = [e for e in page if _local(e.tag).lower() == 'system']
                if not systems:
                    systems = [e for e in page.iter() if _local(e.tag).lower() == 'system']

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

                                heads = chord_heads.get(chord_id, set())
                                if not heads:
                                    continue
                                if not any(head_id not in tied_right_heads for head_id in heads):
                                    continue

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
                                raise ValueError(
                                    f'{member}: sounding slot {slot_id} has no stack slot geometry'
                                )
                            x_offset = slot.get('x-offset')
                            if x_offset is None:
                                raise ValueError(
                                    f'{member}: sounding slot {slot_id} has no x-offset'
                                )
                            strikes.append(
                                Strike(
                                    page_index=page_index,
                                    system_index=system_index,
                                    x=left + float(x_offset),
                                    system_top=top,
                                    system_bottom=bottom,
                                    omr_width=width,
                                    omr_height=height,
                                )
                            )

                page_index += 1

    if not strikes:
        raise ValueError('No sounding note attacks were found')

    logical_hit_count = len(strikes)
    return logical_hit_count, strikes
