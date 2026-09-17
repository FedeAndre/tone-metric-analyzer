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


def _bounds_center(node) -> tuple[float, float] | None:
    bounds = next((c for c in node if _local(c.tag).lower() == 'bounds'), None)
    if bounds is None:
        return None
    try:
        x = float(bounds.get('x')) + float(bounds.get('w')) / 2.0
        y = float(bounds.get('y')) + float(bounds.get('h')) / 2.0
    except Exception:
        return None
    return x, y


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


def _cluster_x(values: list[float], tolerance_px: float = 4.0) -> list[float]:
    """Collapse visually identical attack columns only.

    Four OMR pixels are about one rendered display pixel in the current PDF path;
    this removes duplicate chord/voice anchors that are effectively the same
    vertical column without using x-distance to invent musical time.
    """
    if not values:
        return []
    values = sorted(values)
    groups: list[list[float]] = []
    for x in values:
        if groups:
            center = sum(groups[-1]) / len(groups[-1])
            if abs(x - center) <= tolerance_px:
                groups[-1].append(x)
                continue
        groups.append([x])
    return [sum(g) / len(g) for g in groups]


def extract_hit_strikes(omr_path: str | Path) -> tuple[int, list[Strike]]:
    """Extract only new sounding attacks and their exact visible strike columns.

    Source of truth:
      * Audiveris voice entries with status=BEGIN identify sounding chord starts.
      * Only semantic head-chords qualify; rests and non-note symbols are excluded.
      * A notehead that is the RIGHT endpoint of a semantic tie is a continuation.
      * A chord with at least one untied head is a new sounding attack.
      * Drawing x comes only from the contained attacking notehead geometry.

    Slot x-coordinates and MusicXML engraving coordinates are never used to place
    strikes. If one logical hit contains simultaneous attacks engraved at visibly
    different x positions, each visible attack column is marked so no real attack
    disappears from the diagnostic overlay. The logical hit count remains merged by
    rhythmic slot.
    """
    path = Path(omr_path)
    logical_hit_count = 0
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

            pages = [e for e in root.iter() if _local(e.tag).lower() == 'page'] or [root]
            for page in pages:
                systems = [e for e in page if _local(e.tag).lower() == 'system']
                if not systems:
                    systems = [e for e in page.iter() if _local(e.tag).lower() == 'system']

                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    stacks = [e for e in system if _local(e.tag).lower() == 'stack']
                    parts = [e for e in system if _local(e.tag).lower() == 'part']

                    for stack_index, _stack in enumerate(stacks):
                        slot_chords: dict[str, set[str]] = defaultdict(set)

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
                                slot_id = (key_node.text or '').strip()
                                if slot_id:
                                    slot_chords[slot_id].add(chord_id)

                        stack_attack_x: list[float] = []
                        for chord_ids in slot_chords.values():
                            slot_sounds = False
                            for chord_id in chord_ids:
                                heads = chord_heads.get(chord_id, [])
                                if not heads:
                                    continue
                                new_heads = [h for h in heads if h[0] not in tied_right_heads]
                                if not new_heads:
                                    continue
                                slot_sounds = True
                                xs = sorted(x for _hid, x, _y in new_heads)
                                stack_attack_x.append(xs[len(xs) // 2])
                            if slot_sounds:
                                logical_hit_count += 1

                        for x in _cluster_x(stack_attack_x):
                            strikes.append(
                                Strike(
                                    page_index=page_index,
                                    system_index=system_index,
                                    x=x,
                                    system_top=top,
                                    system_bottom=bottom,
                                    omr_width=width,
                                    omr_height=height,
                                )
                            )

                page_index += 1

    if logical_hit_count <= 0 or not strikes:
        raise ValueError('No sounding note attacks were found')
    return logical_hit_count, strikes
