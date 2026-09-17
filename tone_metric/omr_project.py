from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from lxml import etree
from PIL import Image


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1].lower()


def _frac(value) -> Fraction | None:
    if value is None:
        return None
    try:
        return Fraction(str(value).strip())
    except Exception:
        return None


def _intish(value, default=None):
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _floatish(value, default=None):
    try:
        return float(str(value))
    except Exception:
        return default


def _sheet_number(member: str) -> int:
    m = re.search(r'sheet#(\d+)', member, re.I)
    return int(m.group(1)) if m else 10**9


@dataclass(frozen=True)
class OmrSlot:
    global_measure_index: int
    page_index: int
    system_index: int
    stack_index_in_system: int
    stack_id: str
    left: int
    right: int
    slot_id: int | None
    time_offset_whole: Fraction
    time_offset_quarter: Fraction
    x_offset: int
    x_abs: int
    suspicious: bool = False


def _page_dimensions(zf: ZipFile, sheet_xml_member: str) -> tuple[int, int] | None:
    folder = str(PurePosixPath(sheet_xml_member).parent)
    for name in ('BINARY.png', 'GRAY.png'):
        member = f'{folder}/{name}'
        if member in zf.namelist():
            try:
                im = Image.open(BytesIO(zf.read(member)))
                im.load()
                return tuple(map(int, im.size))
            except Exception:
                pass
    return None


def _system_bounds(system, system_index: int) -> dict | None:
    """Read display-only system bounds from Audiveris staff-line geometry.

    These coordinates never establish musical time or event identity. They are used
    only after a symbolic event has already been fixed by MusicXML score time.
    """
    xs: list[float] = []
    ys: list[float] = []
    for staff in system.iter():
        if _local(staff.tag) != 'staff':
            continue
        left = _floatish(staff.get('left'))
        right = _floatish(staff.get('right'))
        if left is not None:
            xs.append(left)
        if right is not None:
            xs.append(right)
        for line in staff.iter():
            if _local(line.tag) != 'line':
                continue
            for point in line:
                if _local(point.tag) != 'point':
                    continue
                x = _floatish(point.get('x'))
                y = _floatish(point.get('y'))
                if x is not None:
                    xs.append(x)
                if y is not None:
                    ys.append(y)
    if not xs or not ys:
        return None
    return {
        'system_index': int(system_index),
        'left': min(xs),
        'right': max(xs),
        'top': min(ys),
        'bottom': max(ys),
    }


def read_omr_slots(omr_path: str | Path) -> tuple[list[OmrSlot], dict]:
    """Read exact Audiveris rhythmic slots plus display-only system geometry.

    The returned slots are not an event source. A caller may use a slot only after an
    event already exists symbolically and only when ``(measure_index, exact offset)``
    matches. This prevents OMR geometry from creating, deleting, splitting, merging,
    or retiming musical attacks.
    """
    path = Path(omr_path)
    slots: list[OmrSlot] = []
    meta = {
        'available': False,
        'sheet_xml_members': [],
        'pages': [],
        'stack_count': 0,
        'slot_count': 0,
        'warnings': [],
    }
    if not path.exists():
        meta['warnings'].append('Saved Audiveris .omr project was not found.')
        return slots, meta

    global_measure_index = 0
    global_page_index = 0
    try:
        with ZipFile(path, 'r') as zf:
            xml_members = sorted(
                [n for n in zf.namelist() if re.search(r'sheet#\d+/sheet#\d+\.xml$', n, re.I)],
                key=lambda n: (_sheet_number(n), n.lower()),
            )
            meta['sheet_xml_members'] = xml_members
            for member in xml_members:
                try:
                    root = etree.fromstring(zf.read(member))
                except Exception as exc:
                    meta['warnings'].append(f'Could not parse {member}: {exc}')
                    continue
                dims = _page_dimensions(zf, member)
                pages = [el for el in root.iter() if _local(el.tag) == 'page']
                if not pages:
                    pages = [root]
                for page_local_index, page in enumerate(pages):
                    systems = [el for el in page if _local(el.tag) == 'system']
                    if not systems:
                        systems = [el for el in page.iter() if _local(el.tag) == 'system']
                    system_bounds = []
                    for system_index, system in enumerate(systems):
                        b = _system_bounds(system, system_index)
                        if b is not None:
                            system_bounds.append(b)
                    page_meta = {
                        'page_index': global_page_index,
                        'sheet_member': member,
                        'page_local_index': page_local_index,
                        'width': dims[0] if dims else None,
                        'height': dims[1] if dims else None,
                        'systems': len(systems),
                        'system_bounds': system_bounds,
                    }
                    meta['pages'].append(page_meta)
                    for system_index, system in enumerate(systems):
                        stacks = [el for el in system.iter() if _local(el.tag) == 'stack']
                        seen = set()
                        unique_stacks = []
                        for stack in stacks:
                            oid = id(stack)
                            if oid not in seen:
                                seen.add(oid)
                                unique_stacks.append(stack)
                        for stack_index, stack in enumerate(unique_stacks):
                            left = _intish(stack.get('left'))
                            right = _intish(stack.get('right'))
                            if left is None or right is None:
                                meta['warnings'].append(
                                    f'{member}: stack {stack.get("id", "?")} lacks left/right coordinates.'
                                )
                                global_measure_index += 1
                                continue
                            sid = stack.get('id') or str(global_measure_index + 1)
                            stack_slots = [el for el in stack if _local(el.tag) == 'slot']
                            if not stack_slots:
                                stack_slots = [el for el in stack.iter() if _local(el.tag) == 'slot']
                            for slot in stack_slots:
                                toff = _frac(slot.get('time-offset'))
                                xoff = _intish(slot.get('x-offset'))
                                if toff is None or xoff is None:
                                    continue
                                slots.append(OmrSlot(
                                    global_measure_index=global_measure_index,
                                    page_index=global_page_index,
                                    system_index=system_index,
                                    stack_index_in_system=stack_index,
                                    stack_id=str(sid),
                                    left=left,
                                    right=right,
                                    slot_id=_intish(slot.get('id')),
                                    time_offset_whole=toff,
                                    time_offset_quarter=toff * 4,
                                    x_offset=xoff,
                                    x_abs=left + xoff,
                                    suspicious=str(slot.get('suspicious') or '').lower() in {'true', '1', 'yes'},
                                ))
                            global_measure_index += 1
                    global_page_index += 1
    except Exception as exc:
        meta['warnings'].append(f'Could not read Audiveris .omr project: {exc}')
        return [], meta

    meta['available'] = bool(slots)
    meta['stack_count'] = len({s.global_measure_index for s in slots})
    meta['slot_count'] = len(slots)
    return slots, meta


def omr_slots_debug_rows(slots: list[OmrSlot]) -> list[dict]:
    return [
        {
            'global_measure_index': s.global_measure_index,
            'measure_ordinal': s.global_measure_index + 1,
            'page_index': s.page_index,
            'system_index': s.system_index,
            'stack_index_in_system': s.stack_index_in_system,
            'stack_id': s.stack_id,
            'slot_id': s.slot_id,
            'time_offset_whole': str(s.time_offset_whole),
            'time_offset_quarter': str(s.time_offset_quarter),
            'left': s.left,
            'right': s.right,
            'x_offset': s.x_offset,
            'x_abs': s.x_abs,
            'suspicious': s.suspicious,
        }
        for s in slots
    ]
