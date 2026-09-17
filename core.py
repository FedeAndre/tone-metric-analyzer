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

# In dense two-voice engraving, a simultaneous lower voice (stem down) may be
# displaced left while the upper voice (stem up) is displaced right by roughly
# one interline. This is NOT ordinary x clustering. It is a narrow collision-
# engraving rule applied only when semantic timing does not contradict it.
DISPLACED_MIN_DX_INTERLINE = 1.05
DISPLACED_MAX_DX_INTERLINE = 1.50
DISPLACED_MIN_DY_INTERLINE = 1.20
DISPLACED_MAX_DY_INTERLINE = 1.90


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


def _bounds(node) -> tuple[float, float, float, float] | None:
    bounds = next((c for c in node if _local(c.tag) == 'bounds'), None)
    if bounds is None:
        return None
    try:
        x = float(bounds.get('x'))
        y = float(bounds.get('y'))
        return x, y, float(bounds.get('w')), float(bounds.get('h'))
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


def _cluster_chords_by_x(chords: list[dict], tolerance: float) -> list[list[dict]]:
    """Merge only plainly aligned attacks into a single onset."""
    if not chords:
        return []
    ordered = sorted(chords, key=lambda item: item['x'])
    groups: list[list[dict]] = []
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        group_x = [q['x'] for q in groups[-1]]
        if item['x'] - min(group_x) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _merge_displaced_simultaneous_voices(groups: list[list[dict]], interline: float) -> list[list[dict]]:
    """Merge a very specific opposite-stem collision engraving pattern.

    The rule is intentionally conservative:
      * both adjacent onset groups must be single chords;
      * left chord must be stem-down and right chord stem-up;
      * x/y displacement must match the characteristic voice-collision geometry;
      * the two single-note pitches must be three staff steps apart when pitch is known;
      * if both chords have explicit BEGIN slots, those slots must be identical.
        Different explicit slots always win and remain separate.
      * if one/both chords are unvoiced, the geometry may establish simultaneity.

    Thus rhythmic-slot data can veto a merge or corroborate one, but it never
    creates a hit and never suppresses a geometrically distinct sequential pair
    with different explicit times.
    """
    if len(groups) < 2:
        return groups
    out: list[list[dict]] = []
    i = 0
    while i < len(groups):
        if i + 1 < len(groups) and len(groups[i]) == 1 and len(groups[i + 1]) == 1:
            a = groups[i][0]
            b = groups[i + 1][0]
            dx = b['x'] - a['x']
            dy = abs(b['y'] - a['y'])
            pitch_ok = (
                a.get('pitch') is None
                or b.get('pitch') is None
                or abs(a['pitch'] - b['pitch']) == 3
            )
            slots_compatible = not (
                a.get('slot') is not None
                and b.get('slot') is not None
                and a.get('slot') != b.get('slot')
            )
            if (
                a.get('stem') == 'down'
                and b.get('stem') == 'up'
                and DISPLACED_MIN_DX_INTERLINE * interline <= dx <= DISPLACED_MAX_DX_INTERLINE * interline
                and DISPLACED_MIN_DY_INTERLINE * interline <= dy <= DISPLACED_MAX_DY_INTERLINE * interline
                and pitch_ok
                and slots_compatible
            ):
                out.append(groups[i] + groups[i + 1])
                i += 2
                continue
        out.append(groups[i])
        i += 1
    return out


def extract_hit_strikes(omr_path: str | Path) -> tuple[int, list[Strike]]:
    """Return one strike for every visible new note/chord onset.

    Primary hit candidates are semantic head-chords, not old analyzer events.
    Tied-right heads are continuations and are excluded. Geometry assigns chords
    to measures and merges plainly aligned simultaneous attacks. A narrow,
    integrated opposite-stem collision rule then merges displaced simultaneous
    voices without reviving any legacy timing pipeline.

    Audiveris voice/slot metadata is read only as a local simultaneity veto/hint:
    different explicit BEGIN slots prevent a displaced-voice merge; the metadata
    cannot create hits, delete note candidates, or provide strike positions.
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
            picture = next((e for e in root.iter() if _local(e.tag) == 'picture'), None)
            if picture is None:
                raise ValueError(f'{member}: missing picture geometry')
            width = float(picture.get('width') or 0)
            height = float(picture.get('height') or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f'{member}: invalid picture geometry')

            interline = _interline(root)
            tolerance = _cluster_tolerance(root)
            by_id = {e.get('id'): e for e in root.iter() if e.get('id')}
            chord_heads: dict[str, list[tuple[str, float, float, int | None]]] = defaultdict(list)
            tied_right_heads: set[str] = set()
            chord_stem: dict[str, str] = {}

            for relation in (e for e in root.iter() if _local(e.tag) == 'relation'):
                child = next(iter(relation), None)
                if child is None:
                    continue
                kind = _local(child.tag)
                source = relation.get('source')
                target = relation.get('target')

                if kind == 'containment' and source in by_id and target in by_id:
                    source_node = by_id[source]
                    target_node = by_id[target]
                    if _local(source_node.tag) == 'head-chord' and _local(target_node.tag) == 'head':
                        pos = _bounds_center(target_node)
                        if pos is not None:
                            pitch = None
                            try:
                                pitch = int(target_node.get('pitch'))
                            except Exception:
                                pass
                            chord_heads[source].append((target, pos[0], pos[1], pitch))
                elif kind == 'chord-stem' and source and target:
                    chord_stem[source] = target
                elif kind == 'slur-head' and (child.get('side') or '').upper() == 'RIGHT':
                    slur = by_id.get(source)
                    if (
                        slur is not None
                        and _local(slur.tag) == 'slur'
                        and (slur.get('tie') or '').lower() == 'true'
                        and target
                    ):
                        tied_right_heads.add(target)

            def stem_direction(chord_id: str, heads: list[tuple[str, float, float, int | None]]) -> str | None:
                stem_id = chord_stem.get(chord_id)
                stem = by_id.get(stem_id) if stem_id else None
                if stem is None or not heads:
                    return None
                bounds = _bounds(stem)
                if bounds is None:
                    return None
                _, y0, _, h = bounds
                y1 = y0 + h
                head_y = float(median(hh[2] for hh in heads))
                return 'up' if (head_y - y0) > (y1 - head_y) else 'down'

            pages = [e for e in root.iter() if _local(e.tag) == 'page'] or [root]
            for page in pages:
                systems = [e for e in page if _local(e.tag) == 'system']
                if not systems:
                    systems = [e for e in page.iter() if _local(e.tag) == 'system']

                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    staff_ids = {
                        staff.get('id')
                        for staff in system.iter()
                        if _local(staff.tag) == 'staff' and staff.get('id')
                    }
                    stacks = [e for e in system if _local(e.tag) == 'stack']
                    parts = [e for e in system if _local(e.tag) == 'part']
                    if not stacks:
                        continue

                    begin_slot_by_measure: list[dict[str, str]] = [dict() for _ in stacks]
                    for part in parts:
                        measures = [e for e in part if _local(e.tag) == 'measure']
                        for measure_index, measure in enumerate(measures[:len(stacks)]):
                            for entry in (e for e in measure.iter() if _local(e.tag) == 'entry'):
                                key = next((c for c in entry if _local(c.tag) == 'key'), None)
                                value = next((c for c in entry if _local(c.tag) == 'value'), None)
                                if key is None or value is None:
                                    continue
                                if (value.get('status') or '').upper() != 'BEGIN':
                                    continue
                                chord_id = value.get('chord')
                                slot_id = (key.text or '').strip()
                                if chord_id and slot_id:
                                    begin_slot_by_measure[measure_index][chord_id] = slot_id

                    system_chords: list[dict] = []
                    for chord_id, chord in by_id.items():
                        if _local(chord.tag) != 'head-chord' or chord.get('staff') not in staff_ids:
                            continue
                        new_heads = [
                            head for head in chord_heads.get(chord_id, [])
                            if head[0] not in tied_right_heads
                        ]
                        if not new_heads:
                            continue
                        system_chords.append({
                            'id': chord_id,
                            'x': float(median(head[1] for head in new_heads)),
                            'y': float(median(head[2] for head in new_heads)),
                            'pitch': (
                                int(median([head[3] for head in new_heads if head[3] is not None]))
                                if any(head[3] is not None for head in new_heads)
                                else None
                            ),
                            'stem': stem_direction(chord_id, new_heads),
                        })

                    per_measure: list[list[dict]] = [[] for _ in stacks]
                    for record in system_chords:
                        x = record['x']
                        candidates: list[tuple[float, int]] = []
                        for i, stack in enumerate(stacks):
                            left = float(stack.get('left') or 0)
                            right = float(stack.get('right') or left)
                            if left - 3.0 <= x <= right + 3.0:
                                candidates.append((abs(x - (left + right) / 2.0), i))
                        if not candidates:
                            raise ValueError(
                                f"{member}: sounding head-chord {record['id']} at x={x:.1f} does not fall in any measure stack"
                            )
                        _, measure_index = min(candidates)
                        record = dict(record)
                        record['slot'] = begin_slot_by_measure[measure_index].get(record['id'])
                        per_measure[measure_index].append(record)

                    for chords in per_measure:
                        groups = _cluster_chords_by_x(chords, tolerance)
                        groups = _merge_displaced_simultaneous_voices(groups, interline)
                        for group in groups:
                            group_x = [item['x'] for item in group]
                            x = (
                                min(group_x)
                                if max(group_x) - min(group_x) > tolerance
                                else float(median(group_x))
                            )
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
