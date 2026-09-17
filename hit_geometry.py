from __future__ import annotations

import re

from hit_model import Box

def _local(tag: object) -> str:
    return tag.rsplit('}', 1)[-1].lower() if isinstance(tag, str) else ''


def _sheet_number(name: str) -> int:
    match = re.search(r'sheet#(\d+)', name, re.I)
    return int(match.group(1)) if match else 10**9


def _box(node) -> Box | None:
    bounds = next((c for c in node if _local(c.tag) == 'bounds'), None)
    if bounds is None:
        return None
    try:
        return Box(float(bounds.get('x')), float(bounds.get('y')), float(bounds.get('w')), float(bounds.get('h')))
    except (TypeError, ValueError):
        return None


def _interline(root) -> float:
    for scale in (n for n in root.iter() if _local(n.tag) == 'scale'):
        inter = next((c for c in scale if _local(c.tag) == 'interline'), None)
        if inter is not None and inter.get('main'):
            value = float(inter.get('main'))
            if value > 0:
                return value
    return 20.0


def _system_vertical_bounds(system, page_height: float) -> tuple[float, float]:
    ys: list[float] = []
    for staff in (n for n in system.iter() if _local(n.tag) == 'staff'):
        for line in (n for n in staff.iter() if _local(n.tag) == 'line'):
            for point in line:
                if _local(point.tag) == 'point' and point.get('y') is not None:
                    ys.append(float(point.get('y')))
    return (min(ys), max(ys)) if ys else (0.0, page_height)


def _assign_measure(chord_box: Box, stacks: list[tuple[float, float]]) -> int:
    direct = [
        (abs(chord_box.cx - (left + right) / 2.0), i)
        for i, (left, right) in enumerate(stacks)
        if left <= chord_box.cx <= right
    ]
    if direct:
        return min(direct)[1]
    distances: list[tuple[float, int]] = []
    for i, (left, right) in enumerate(stacks):
        if chord_box.cx < left:
            distance = left - chord_box.cx
        elif chord_box.cx > right:
            distance = chord_box.cx - right
        else:
            distance = 0.0
        distances.append((distance, i))
    distance, index = min(distances)
    if distance <= max(2.0, chord_box.w * 0.20):
        return index
    raise ValueError(f'Attack chord at x={chord_box.cx:.1f} cannot be assigned to one measure')
