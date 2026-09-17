from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import re

from lxml import etree


ONSET_CLUSTER_INTERLINE_FRACTION = 0.70
MIN_CLUSTER_TOLERANCE_PX = 6.0
MAX_CLUSTER_TOLERANCE_PX = 22.0


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
    return tag.rsplit('}', 1)[-1].lower()


def _sheet_number(name: str) -> int:
    match = re.search(r'sheet#(\d+)', name, re.I)
    return int(match.group(1)) if match else 10**9


def _bounds_center(node) -> tuple[float, float] | None:
    bounds = next((c for c in node if _local(c.tag) == 'bounds'), None)
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
    for staff in (e for e in system.iter() if _local(e.tag) == 'staff'):
        for line in (e for e in staff.iter() if _local(e.tag) == 'line'):
            for point in line:
                if _local(point.tag) != 'point':
                    continue
                try:
                    ys.append(float(point.get('y')))
                except Exception:
                    pass
    return (min(ys), max(ys)) if ys else (0.0, page_height)


def _interline(root) -> float:
    for scale in (e for e in root.iter() if _local(e.tag) == 'scale'):
        inter = next((c for c in scale if _local(c.tag) == 'interline'), None)
        if inter is not None:
            try:
                value = float(inter.get('main'))
                if value > 0:
                    return value
            except Exception:
                pass
    return 20.0


def _cluster_tolerance(root) -> float:
    value = _interline(root) * ONSET_CLUSTER_INTERLINE_FRACTION
    return max(MIN_CLUSTER_TOLERANCE_PX, min(MAX_CLUSTER_TOLERANCE_PX, value))


def _cluster_chords_by_x(
    chords: list[tuple[str, float]], tolerance: float
) -> list[list[tuple[str, float]]]:
    """Merge only visibly aligned sounding chords into one global onset.

    Clustering is measure-local and never uses Audiveris rhythmic slot IDs or
    slot times. The span rule prevents a chain of close x values from swallowing
    several successive attacks.
    """
    if not chords:
        return []
    ordered = sorted(chords, key=lambda item: item[1])
    groups: list[list[tuple[str, float]]] = []
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        group_x = [q[1] for q in groups[-1]]
        if item[1] - min(group_x) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def extract_hit_strikes(omr_path: str | Path) -> tuple[int, list[Strike]]:
    """Return one strike for every visible new note/chord onset.

    Every semantic head-chord recognized by Audiveris is considered directly.
    A head that is the RIGHT endpoint of a semantic tie is a continuation. A
    chord with at least one untied head is a sounding attack. Chords are assigned
    to their engraved measure by geometry, then visually aligned attacks within
    that measure are merged into one global hit.

    Audiveris voice entries, rhythmic slots, slot IDs, slot time offsets and
    MusicXML are intentionally NOT used to create, delete, merge or retime hits.
    """
    path = Path(omr_path)
    strikes: list[Strike] = []

    with ZipFile(path) as zf:
        members = sorted(
            [
                n
                for n in zf.namelist()
                if re.search(r'sheet#\d+/sheet#\d+\.xml$', n, re.I)
            ],
            key=lambda n: (_sheet_number(n), n.lower()),
        )
        if not members:
            raise ValueError('Audiveris project contains no sheet XML')

        page_index = 0
        for member in members:
            root = etree.fromstring(zf.read(member))
            picture = next(
                (e for e in root.iter() if _local(e.tag) == 'picture'), None
            )
            if picture is None:
                raise ValueError(f'{member}: missing picture geometry')
            width = float(picture.get('width') or 0)
            height = float(picture.get('height') or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f'{member}: invalid picture geometry')

            tolerance = _cluster_tolerance(root)
            by_id = {e.get('id'): e for e in root.iter() if e.get('id')}
            chord_heads: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
            tied_right_heads: set[str] = set()

            for relation in (
                e for e in root.iter() if _local(e.tag) == 'relation'
            ):
                child = next(iter(relation), None)
                if child is None:
                    continue
                kind = _local(child.tag)
                source = relation.get('source')
                target = relation.get('target')

                if kind == 'containment' and source in by_id and target in by_id:
                    source_node = by_id[source]
                    target_node = by_id[target]
                    if (
                        _local(source_node.tag) == 'head-chord'
                        and _local(target_node.tag) == 'head'
                    ):
                        pos = _bounds_center(target_node)
                        if pos is not None:
                            chord_heads[source].append((target, pos[0], pos[1]))

                elif (
                    kind == 'slur-head'
                    and (child.get('side') or '').upper() == 'RIGHT'
                ):
                    slur = by_id.get(source)
                    if (
                        slur is not None
                        and _local(slur.tag) == 'slur'
                        and (slur.get('tie') or '').lower() == 'true'
                        and target
                    ):
                        tied_right_heads.add(target)

            pages = [e for e in root.iter() if _local(e.tag) == 'page'] or [root]
            for page in pages:
                systems = [e for e in page if _local(e.tag) == 'system']
                if not systems:
                    systems = [
                        e for e in page.iter() if _local(e.tag) == 'system'
                    ]

                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    staff_ids = {
                        staff.get('id')
                        for staff in system.iter()
                        if _local(staff.tag) == 'staff' and staff.get('id')
                    }
                    stacks = [e for e in system if _local(e.tag) == 'stack']
                    if not stacks:
                        continue

                    system_chords: list[tuple[str, float]] = []
                    for chord_id, chord in by_id.items():
                        if (
                            _local(chord.tag) != 'head-chord'
                            or chord.get('staff') not in staff_ids
                        ):
                            continue
                        new_heads = [
                            head
                            for head in chord_heads.get(chord_id, [])
                            if head[0] not in tied_right_heads
                        ]
                        if not new_heads:
                            continue
                        anchor_x = float(median(head[1] for head in new_heads))
                        system_chords.append((chord_id, anchor_x))

                    per_measure: list[list[tuple[str, float]]] = [
                        [] for _ in stacks
                    ]
                    for chord_id, x in system_chords:
                        candidates: list[tuple[float, int]] = []
                        for i, stack in enumerate(stacks):
                            left = float(stack.get('left') or 0)
                            right = float(stack.get('right') or left)
                            if left - 3.0 <= x <= right + 3.0:
                                center_x = (left + right) / 2.0
                                candidates.append((abs(x - center_x), i))
                        if not candidates:
                            raise ValueError(
                                f'{member}: sounding head-chord {chord_id} at '
                                f'x={x:.1f} does not fall in any measure stack'
                            )
                        _, measure_index = min(candidates)
                        per_measure[measure_index].append((chord_id, x))

                    for chords in per_measure:
                        for group in _cluster_chords_by_x(chords, tolerance):
                            x = float(median(item[1] for item in group))
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

    if not strikes:
        raise ValueError('No sounding note attacks were found')
    return len(strikes), strikes
