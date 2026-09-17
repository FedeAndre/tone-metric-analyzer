from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile
import re

from lxml import etree


@dataclass(frozen=True, order=True)
class Hit:
    measure_index: int
    offset: Fraction


@dataclass(frozen=True)
class Strike:
    hit: Hit
    page_index: int
    system_index: int
    x: float
    system_top: float
    system_bottom: float
    omr_width: float
    omr_height: float


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _read_musicxml(path: str | Path):
    path = Path(path)
    if path.suffix.lower() != '.mxl':
        return etree.parse(str(path)).getroot()
    with ZipFile(path) as zf:
        container = etree.fromstring(zf.read('META-INF/container.xml'))
        node = next((e for e in container.iter() if _local(e.tag) == 'rootfile'), None)
        if node is None or not node.get('full-path'):
            raise ValueError('MusicXML archive has no rootfile')
        return etree.fromstring(zf.read(node.get('full-path')))


def _duration(note, divisions: int) -> Fraction:
    text = note.findtext('duration')
    if not text:
        return Fraction(0)
    return Fraction(int(text), divisions)


def _tie_types(note) -> set[str]:
    out: set[str] = set()
    for node in list(note.findall('tie')) + list(note.findall('notations/tied')):
        kind = (node.get('type') or '').strip().lower()
        if kind:
            out.add(kind)
    return out


def extract_hits(musicxml_path: str | Path) -> list[Hit]:
    """Return every and only new sounding onsets, merged globally per measure/time."""
    root = _read_musicxml(musicxml_path)
    if _local(root.tag) != 'score-partwise':
        raise ValueError('Only score-partwise MusicXML is supported')

    hits: set[Hit] = set()
    parts = root.findall('part')
    if not parts:
        raise ValueError('No parts found in MusicXML')

    for part in parts:
        divisions = 1
        for measure_index, measure in enumerate(part.findall('measure')):
            attributes = measure.find('attributes')
            if attributes is not None and attributes.find('divisions') is not None:
                divisions = int(attributes.findtext('divisions'))
                if divisions <= 0:
                    raise ValueError('MusicXML divisions must be positive')

            cursor = Fraction(0)
            last_non_chord_onset = Fraction(0)

            for node in measure:
                tag = _local(node.tag)
                if tag == 'note':
                    duration = _duration(node, divisions)
                    is_chord = node.find('chord') is not None
                    is_grace = node.find('grace') is not None

                    if is_chord:
                        onset = last_non_chord_onset
                    else:
                        onset = cursor
                        last_non_chord_onset = onset
                        if not is_grace:
                            cursor += duration

                    if node.find('rest') is not None:
                        continue
                    if is_grace:
                        continue
                    if 'stop' in _tie_types(node):
                        continue

                    hits.add(Hit(measure_index, onset))

                elif tag == 'backup':
                    cursor -= Fraction(int(node.findtext('duration') or '0'), divisions)
                elif tag == 'forward':
                    cursor += Fraction(int(node.findtext('duration') or '0'), divisions)

    return sorted(hits)


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
    if not ys:
        return 0.0, page_height
    return min(ys), max(ys)


def extract_strike_slots(omr_path: str | Path) -> dict[Hit, Strike]:
    """Read Audiveris score-time slots only as physical positions for known hits."""
    path = Path(omr_path)
    slots: dict[Hit, Strike] = {}
    measure_index = 0
    page_index = 0

    with ZipFile(path) as zf:
        members = sorted(
            [n for n in zf.namelist() if re.search(r'sheet#\d+/sheet#\d+\.xml$', n, re.I)],
            key=lambda n: (_sheet_number(n), n.lower()),
        )
        if not members:
            raise ValueError('Audiveris project contains no sheet XML')

        for member in members:
            root = etree.fromstring(zf.read(member))
            picture = next((e for e in root.iter() if _local(e.tag).lower() == 'picture'), None)
            if picture is None:
                raise ValueError(f'{member}: missing picture geometry')
            width = float(picture.get('width') or 0)
            height = float(picture.get('height') or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f'{member}: invalid picture geometry')

            pages = [e for e in root.iter() if _local(e.tag).lower() == 'page'] or [root]
            for page in pages:
                systems = [e for e in page if _local(e.tag).lower() == 'system']
                if not systems:
                    systems = [e for e in page.iter() if _local(e.tag).lower() == 'system']

                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    stacks = [e for e in system.iter() if _local(e.tag).lower() == 'stack']
                    for stack in stacks:
                        left = float(stack.get('left') or 0)
                        for slot in stack:
                            if _local(slot.tag).lower() != 'slot':
                                continue
                            time_text = slot.get('time-offset')
                            x_text = slot.get('x-offset')
                            if time_text is None or x_text is None:
                                continue
                            offset = Fraction(time_text) * 4
                            x = left + float(x_text)
                            hit = Hit(measure_index, offset)
                            slots[hit] = Strike(
                                hit=hit,
                                page_index=page_index,
                                system_index=system_index,
                                x=x,
                                system_top=top,
                                system_bottom=bottom,
                                omr_width=width,
                                omr_height=height,
                            )
                        measure_index += 1
                page_index += 1

    return slots


def map_hits_to_strikes(hits: list[Hit], omr_path: str | Path) -> list[Strike]:
    slots = extract_strike_slots(omr_path)
    missing = [hit for hit in hits if hit not in slots]
    if missing:
        sample = ', '.join(f'm{h.measure_index + 1}@{h.offset}' for h in missing[:12])
        raise ValueError(f'{len(missing)} hit(s) could not be placed exactly: {sample}')
    return [slots[hit] for hit in hits]
