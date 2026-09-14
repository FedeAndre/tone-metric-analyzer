from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from lxml import etree
from PIL import Image
from io import BytesIO


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1].lower()


def _frac(value) -> Fraction | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Fraction(text)
    except Exception:
        return None


def _intish(value, default=None):
    try:
        return int(float(str(value)))
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


def read_omr_slots(omr_path: str | Path) -> tuple[list[OmrSlot], dict]:
    """Read Audiveris measure-stack slots directly from a saved .omr project.

    Audiveris persists each measure ``stack`` with absolute ``left``/``right``
    coordinates, and each rhythmic ``slot`` with both ``time-offset`` and
    ``x-offset``.  ``x-offset`` is explicitly measured from the stack's left edge.
    This gives the identity we were missing in earlier builds:

        exact measure time -> Audiveris slot -> exact physical x

    No ordinal note matching, MusicXML default-x interpolation, or nearest-neighbor
    notehead pairing is involved in this path.
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
                # Normally one PDF sheet -> one Audiveris page.  The XML model also
                # permits several <page> nodes per sheet, so preserve their order.
                pages = [el for el in root.iter() if _local(el.tag) == 'page']
                if not pages:
                    pages = [root]
                for page_local_index, page in enumerate(pages):
                    systems = [el for el in page if _local(el.tag) == 'system']
                    if not systems:
                        # Be tolerant of wrapper elements used by different Audiveris versions.
                        systems = [el for el in page.iter() if _local(el.tag) == 'system']
                    page_meta = {
                        'page_index': global_page_index,
                        'sheet_member': member,
                        'page_local_index': page_local_index,
                        'width': dims[0] if dims else None,
                        'height': dims[1] if dims else None,
                        'systems': len(systems),
                    }
                    meta['pages'].append(page_meta)
                    for system_index, system in enumerate(systems):
                        stacks = [el for el in system.iter() if _local(el.tag) == 'stack']
                        # Avoid accidentally re-reading nested descendants if a future
                        # schema wraps stack-like content: document order is sufficient.
                        seen = set()
                        unique_stacks = []
                        for stack in stacks:
                            oid = id(stack)
                            if oid not in seen:
                                seen.add(oid); unique_stacks.append(stack)
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
                                    suspicious=str(slot.get('suspicious') or '').lower() in {'true','1','yes'},
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
