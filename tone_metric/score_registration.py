from __future__ import annotations

"""Exact visual registration for already-established symbolic attacks.

Musical event identity and score time are fixed upstream. This module performs one
strict display-only lookup:

    symbolic (measure_index, exact offset) -> existing Audiveris slot -> PDF x/system

No nearest-slot, x-cluster, MusicXML-layout, ordinal-note, or interpolation fallback
exists here. Extra OMR slots are ignored. Missing exact matches remain unmapped.
"""

from collections import defaultdict
from fractions import Fraction

from .omr_project import OmrSlot


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _event_targets(analysis_result: dict) -> list[dict]:
    rows = []
    for si, segment in enumerate(analysis_result.get('segments', [])):
        for event in segment.get('events', []):
            levels = sorted({int(x) for x in event.get('tone_metric_levels', []) if int(x) > 0})
            if not levels:
                continue
            mi = int(event.get('measure_index', -1))
            off = _frac(event.get('offset_in_measure_quarter'))
            if mi < 0 or off is None:
                continue
            rows.append({
                'segment_index': si,
                'event': event,
                'levels': levels,
                'measure_index': mi,
                'offset': off,
                'attack_key': str(event.get('attack_key') or f'{mi}:{off}'),
            })
    return rows


def build_layer_anchors_from_exact_omr_slots(
    analysis_result: dict,
    omr_slots: list[OmrSlot],
    omr_meta: dict,
):
    by_key: dict[tuple[int, Fraction], list[OmrSlot]] = defaultdict(list)
    for slot in omr_slots:
        by_key[(int(slot.global_measure_index), Fraction(slot.time_offset_quarter))].append(slot)

    pages = {
        int(p.get('page_index', -1)): p
        for p in (omr_meta.get('pages', []) or [])
        if int(p.get('page_index', -1)) >= 0
    }
    anchors_by_page = defaultdict(list)
    per_level = defaultdict(lambda: {'expected': 0, 'mapped': 0, 'missing': 0})
    missing = defaultdict(list)
    mapped_events = 0
    ambiguous_exact_slot_keys = 0
    suspicious_exact_matches = 0

    for target in _event_targets(analysis_result):
        event = target['event']
        levels = target['levels']
        for level in levels:
            per_level[level]['expected'] += 1

        candidates = by_key.get((target['measure_index'], target['offset']), [])
        if len(candidates) > 1:
            ambiguous_exact_slot_keys += 1
        candidates = sorted(
            candidates,
            key=lambda s: (bool(s.suspicious), s.page_index, s.system_index, s.x_abs, s.slot_id or 0),
        )
        slot = candidates[0] if candidates else None
        if slot is None:
            for level in levels:
                per_level[level]['missing'] += 1
                missing[level].append({
                    'measure_index': event.get('measure_index'),
                    'measure_number': event.get('measure_number'),
                    'offset_in_measure_quarter': event.get('offset_in_measure_quarter'),
                    'attack_key': target['attack_key'],
                    'reason': 'exact-omr-slot-not-found',
                })
            continue

        page_meta = pages.get(int(slot.page_index), {})
        width = float(page_meta.get('width') or 0)
        height = float(page_meta.get('height') or 0)
        if width <= 0 or height <= 0:
            for level in levels:
                per_level[level]['missing'] += 1
                missing[level].append({
                    'measure_index': event.get('measure_index'),
                    'measure_number': event.get('measure_number'),
                    'offset_in_measure_quarter': event.get('offset_in_measure_quarter'),
                    'attack_key': target['attack_key'],
                    'reason': 'omr-page-dimensions-unavailable',
                })
            continue

        if slot.suspicious:
            suspicious_exact_matches += 1
        x_norm = max(0.0, min(1.0, float(slot.x_abs) / width))
        row = {
            'page_index': int(slot.page_index),
            'system_index': int(slot.system_index),
            'physical_system_index': int(slot.system_index),
            'segment_index': int(target['segment_index']),
            'event_index': int(event.get('event_index', -1) or -1),
            'measure_index': int(event.get('measure_index', -1)),
            'measure_number': event.get('measure_number'),
            'onset_quarter': event.get('onset_quarter'),
            'offset_in_measure_quarter': event.get('offset_in_measure_quarter'),
            'attack_key': target['attack_key'],
            'duration_quarter': event.get('duration_quarter', ''),
            'cx_norm': x_norm,
            'cy_norm': 0.0,
            'height': max(levels),
            'lowest_level': min(levels),
            'levels': levels,
            'parenthetical': False,
            'registration_source': 'exact-symbolic-measure-offset->audiveris-slot',
            'timing_source': 'musicxml-symbolic-global-onset',
            'timing_confidence': 'symbolic-exact',
            'recovered_x_abs': float(slot.x_abs),
            'visual_position_is_estimated': False,
            'vertical_position_is_system_estimate': False,
            'slot_id': slot.slot_id,
            'slot_suspicious': bool(slot.suspicious),
            'stack_left': slot.left,
            'stack_right': slot.right,
        }
        anchors_by_page[int(slot.page_index)].append(row)
        mapped_events += 1
        for level in levels:
            per_level[level]['mapped'] += 1

    for rows in anchors_by_page.values():
        rows.sort(key=lambda a: (
            int(a.get('system_index', 0)),
            int(a.get('measure_index', -1)),
            _frac(a.get('offset_in_measure_quarter'), Fraction(0)),
        ))

    missing_total = sum(v['missing'] for v in per_level.values())
    warnings = []
    if missing_total:
        warnings.append(
            f'{missing_total} event-level label(s) lacked an exact Audiveris slot at the same symbolic measure/offset; no fallback was used.'
        )
    if ambiguous_exact_slot_keys:
        warnings.append(
            f'{ambiguous_exact_slot_keys} symbolic event(s) had more than one exact OMR slot candidate; deterministic exact-key selection was used.'
        )

    attack_levels = sorted(per_level)
    stats = {
        'final_layer_anchors': sum(len(v) for v in anchors_by_page.values()),
        'layer_anchor_registration': 'symbolic-exact-measure-offset->audiveris-slot',
        'registration_timing_authority': 'musicxml-symbolic-global-onset',
        'registration_can_create_attacks': False,
        'registration_can_retime_attacks': False,
        'registration_can_nearest_match_time': False,
        'analysis_position_policy': analysis_result.get('analysis_position_policy', 'symbolic-attacks-plus-independent-metric-grid'),
        'levels_enabled': attack_levels,
        'max_layer_level': max(attack_levels, default=0),
        'layer_hits_expected': sum(v['expected'] for v in per_level.values()),
        'layer_hits_mapped': sum(v['mapped'] for v in per_level.values()),
        'layer_hits_without_visual_attack': missing_total,
        'symbolic_events_mapped': mapped_events,
        'exact_omr_slots_available': len(omr_slots),
        'ambiguous_exact_slot_keys': ambiguous_exact_slot_keys,
        'suspicious_exact_matches': suspicious_exact_matches,
        'musicxml_layout_coordinates_used': False,
        'manufactured_x_coordinates': 0,
        'semantic_attack_column_fallbacks': 0,
        'per_level': {str(k): dict(v) for k, v in sorted(per_level.items())},
        'missing_targets_by_level': {str(k): v for k, v in sorted(missing.items())},
        'structural_parenthetical_positions': 0,
        'structural_parenthetical_labels_expected': 0,
        'structural_parenthetical_labels_mapped': 0,
        'structural_parenthetical_labels_missing': 0,
        'structural_per_level': {},
        'structural_missing_targets_by_level': {},
        'structural_registration_sources': {},
        'structural_reasons': {},
    }
    for level, values in per_level.items():
        stats[f'level{level}_anchors_mapped'] = values['mapped']
        stats[f'level{level}_hits_expected'] = values['expected']
        stats[f'level{level}_hits_without_visual_attack'] = values['missing']
    return dict(anchors_by_page), {}, stats, warnings
