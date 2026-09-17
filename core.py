from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import re

from lxml import etree

from hit_model import Chord, GlobalOnset, LocalOnset, Strike, Diagnostics
from hit_geometry import _local, _sheet_number, _box, _interline, _system_vertical_bounds, _assign_measure
from hit_align import _same_staff_local_sequence, _align_staff_sequence, _actual_strike_x

def extract_hit_strikes(omr_path: str | Path) -> tuple[int, list[Strike], Diagnostics]:
    """Return one strike for every global new-note onset.

    The engine is deliberately not a patch chain:
      1. Every semantic head-chord with a non-tied attack head enters a staff
         sequence exactly once.
      2. Known exact BEGIN time groups simultaneous voices *within a staff*.
         Untimed recognized chords remain present instead of being discarded.
      3. Staff sequences are merged with order-preserving one-to-one alignment.
         Known equal times corroborate matches; known different times forbid a
         match; untimed notes can match only by local geometric alignment.
      4. Every red strike is drawn through an actual attacking notehead. No
         synthetic, averaged, clustered-center, or invented x exists.
    """
    path = Path(omr_path)
    strikes: list[Strike] = []
    sounding_chords = 0
    timed_chords = 0
    untimed_chords = 0
    global_measure = 0

    with ZipFile(path) as archive:
        members = sorted(
            [name for name in archive.namelist() if re.search(r'sheet#\d+/sheet#\d+\.xml$', name, re.I)],
            key=lambda name: (_sheet_number(name), name.lower()),
        )
        if not members:
            raise ValueError('Audiveris project contains no sheet XML')

        page_index = 0
        for member_name in members:
            root = etree.fromstring(archive.read(member_name))
            picture = next((n for n in root.iter() if _local(n.tag) == 'picture'), None)
            if picture is None:
                raise ValueError(f'{member_name}: missing picture geometry')
            width = float(picture.get('width') or 0)
            height = float(picture.get('height') or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f'{member_name}: invalid picture geometry')
            interline = _interline(root)

            by_id = {n.get('id'): n for n in root.iter() if n.get('id')}
            chord_heads: dict[str, list[str]] = defaultdict(list)
            head_x: dict[str, float] = {}
            tied_right: set[str] = set()
            chord_stem: dict[str, str] = {}
            stem_x: dict[str, float] = {}

            for node in root.iter():
                if _local(node.tag) == 'head':
                    b = _box(node)
                    if b is not None:
                        head_x[node.get('id')] = b.cx
                elif _local(node.tag) == 'stem':
                    b = _box(node)
                    if b is not None:
                        stem_x[node.get('id')] = b.cx

            for relation in (n for n in root.iter() if _local(n.tag) == 'relation'):
                child = next(iter(relation), None)
                if child is None:
                    continue
                kind = _local(child.tag)
                source = relation.get('source')
                target = relation.get('target')
                if (
                    kind == 'containment'
                    and source in by_id and target in by_id
                    and _local(by_id[source].tag) == 'head-chord'
                    and _local(by_id[target].tag) == 'head'
                ):
                    chord_heads[source].append(target)
                elif kind == 'chord-stem' and source and target:
                    chord_stem[source] = target
                elif kind == 'slur-head' and (child.get('side') or '').upper() == 'RIGHT':
                    slur = by_id.get(source)
                    if slur is not None and _local(slur.tag) == 'slur' and (slur.get('tie') or '').lower() == 'true' and target:
                        tied_right.add(target)

            pages = [n for n in root.iter() if _local(n.tag) == 'page'] or [root]
            for page in pages:
                systems = [n for n in page if _local(n.tag) == 'system']
                if not systems:
                    systems = [n for n in page.iter() if _local(n.tag) == 'system']
                for system_index, system in enumerate(systems):
                    top, bottom = _system_vertical_bounds(system, height)
                    stacks = [n for n in system if _local(n.tag) == 'stack']
                    parts = [n for n in system if _local(n.tag) == 'part']
                    if not stacks:
                        continue
                    stack_bounds = [(float(s.get('left') or 0), float(s.get('right') or 0)) for s in stacks]
                    staff_ids = [
                        s.get('id') for s in system.iter()
                        if _local(s.tag) == 'staff' and s.get('id')
                    ]

                    # Collect exact BEGIN times by chord. Multiple CONTINUE
                    # entries are deliberately ignored; only the attack BEGIN matters.
                    begin_time_by_measure: list[dict[str, Fraction]] = [dict() for _ in stacks]
                    for part in parts:
                        measures = [n for n in part if _local(n.tag) == 'measure']
                        for stack_index, measure in enumerate(measures[:len(stacks)]):
                            slots = {
                                s.get('id'): s for s in stacks[stack_index]
                                if _local(s.tag) == 'slot' and s.get('id')
                            }
                            for entry in (n for n in measure.iter() if _local(n.tag) == 'entry'):
                                key = next((c for c in entry if _local(c.tag) == 'key'), None)
                                value = next((c for c in entry if _local(c.tag) == 'value'), None)
                                if key is None or value is None or (value.get('status') or '').upper() != 'BEGIN':
                                    continue
                                chord_id = value.get('chord')
                                slot = slots.get((key.text or '').strip())
                                if not chord_id or slot is None:
                                    continue
                                try:
                                    time = Fraction(slot.get('time-offset') or '0')
                                except Exception as exc:
                                    raise ValueError(f'{member_name}: invalid time-offset') from exc
                                previous = begin_time_by_measure[stack_index].get(chord_id)
                                if previous is not None and previous != time:
                                    raise ValueError(f'{member_name}: chord {chord_id} has conflicting BEGIN times')
                                begin_time_by_measure[stack_index][chord_id] = time

                    chords_by_measure_staff: dict[tuple[int, str], list[Chord]] = defaultdict(list)
                    for chord_id, node in by_id.items():
                        if _local(node.tag) != 'head-chord' or node.get('staff') not in staff_ids:
                            continue
                        b = _box(node)
                        if b is None:
                            continue
                        heads = chord_heads.get(chord_id, [])
                        attack_xs = tuple(head_x[h] for h in heads if h in head_x and h not in tied_right)
                        if not attack_xs:
                            continue
                        stack_index = _assign_measure(b, stack_bounds)
                        symbolic_time = begin_time_by_measure[stack_index].get(chord_id)
                        stem = chord_stem.get(chord_id)
                        visual_x = stem_x.get(stem, float(median(attack_xs)))
                        measure_index = global_measure + stack_index + 1
                        chord = Chord(
                            id=chord_id,
                            staff=node.get('staff') or '',
                            measure_index=measure_index,
                            attack_head_xs=attack_xs,
                            box=b,
                            visual_x=float(visual_x),
                            symbolic_time=symbolic_time,
                        )
                        chords_by_measure_staff[(stack_index, chord.staff)].append(chord)
                        sounding_chords += 1
                        if symbolic_time is None:
                            untimed_chords += 1
                        else:
                            timed_chords += 1

                    for stack_index in range(len(stacks)):
                        measure_index = global_measure + stack_index + 1
                        staff_sequences: list[list[LocalOnset]] = []
                        for staff in staff_ids:
                            chords = chords_by_measure_staff.get((stack_index, staff), [])
                            if chords:
                                staff_sequences.append(_same_staff_local_sequence(chords, interline))

                        if not staff_sequences:
                            continue
                        global_seq = [GlobalOnset([item]) for item in staff_sequences[0]]
                        for staff_seq in staff_sequences[1:]:
                            global_seq = _align_staff_sequence(global_seq, staff_seq, interline)

                        # No chord can vanish: alignment carries every local item
                        # into exactly one global onset.
                        carried = sum(len(item.chords) for event in global_seq for item in event.items)
                        expected = sum(len(item.chords) for seq in staff_sequences for item in seq)
                        if carried != expected:
                            raise AssertionError(f'Measure {measure_index}: alignment lost chords')

                        for event in global_seq:
                            x = _actual_strike_x(event)
                            if x not in event.attack_head_xs:
                                raise AssertionError('Synthetic strike position generated')
                            recovered = any(item.symbolic_time is None for item in event.items)
                            strikes.append(Strike(
                                page_index=page_index,
                                system_index=system_index,
                                measure_index=measure_index,
                                x=float(x),
                                system_top=top,
                                system_bottom=bottom,
                                omr_width=width,
                                omr_height=height,
                                recovered=recovered,
                            ))

                    global_measure += len(stacks)
                page_index += 1

    if not strikes:
        raise ValueError('No sounding note attacks were found')
    strikes.sort(key=lambda s: (s.page_index, s.system_index, s.measure_index, s.x))
    diagnostics = Diagnostics(
        global_hits=len(strikes),
        sounding_chords=sounding_chords,
        timed_chords=timed_chords,
        untimed_chords=untimed_chords,
        synthetic_strike_positions=0,
        sequence_alignment=True,
    )
    return len(strikes), strikes, diagnostics
