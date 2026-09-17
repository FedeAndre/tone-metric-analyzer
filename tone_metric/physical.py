"""Display-only PDF registration for the Tone-Metric Analyzer.

Musical score time and recursive Levels are complete before this module runs.
The only permitted geometric operation is an exact lookup of an existing symbolic
attack at the same Audiveris ``(measure_index, offset)`` slot. No geometry can
create, delete, split, merge, reorder, or retime events.
"""

from __future__ import annotations

from .omr_project import OmrSlot
from .score_registration import build_layer_anchors_from_exact_omr_slots
from .waves import build_wave_profile, register_wave_profile
from .pivots import build_pivot_profile, register_pivot_profile
from .trees import build_tree_profile, register_tree_profile


def _normalized_system_bounds(omr_meta: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for page in omr_meta.get('pages', []) or []:
        try:
            page_index = int(page.get('page_index', 0))
            width = float(page.get('width') or 0)
            height = float(page.get('height') or 0)
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        raw = sorted(
            list(page.get('system_bounds', []) or []),
            key=lambda b: int(b.get('system_index', 0)),
        )
        rows = []
        for pos, b in enumerate(raw):
            try:
                top = float(b['top'])
                bottom = float(b['bottom'])
                left = float(b['left'])
                right = float(b['right'])
                system_index = int(b.get('system_index', pos))
            except Exception:
                continue
            if pos + 1 < len(raw):
                try:
                    next_top = float(raw[pos + 1]['top'])
                    guard_bottom = (bottom + next_top) / 2.0
                except Exception:
                    guard_bottom = bottom
            else:
                guard_bottom = bottom
            rows.append({
                'system_index': system_index,
                'top_norm': max(0.0, min(1.0, top / height)),
                'bottom_norm': max(0.0, min(1.0, bottom / height)),
                'guard_bottom_norm': max(0.0, min(1.0, guard_bottom / height)),
                'left_norm': max(0.0, min(1.0, left / width)),
                'right_norm': max(0.0, min(1.0, right / width)),
            })
        out[page_index] = rows
    return out


def build_normalized_overlay(
    analysis_result: dict,
    omr_slots: list[OmrSlot],
    omr_meta: dict,
) -> dict:
    """Attach exact OMR display geometry to an already-complete analysis."""
    bounds_by_page = _normalized_system_bounds(omr_meta)
    page_indices = sorted({
        int(p.get('page_index', 0)) for p in (omr_meta.get('pages', []) or [])
    })
    per_page = {
        idx: {
            'page_index': idx,
            'overlays': [],
            'layer_anchors': [],
            'structural_anchors': [],
            'wave_anchors': [],
            'pivot_anchors': [],
            'tree_nodes': [],
            'tree_branches': [],
            'system_bounds': bounds_by_page.get(idx, []),
        }
        for idx in page_indices
    }

    anchors_by_page, structural_by_page, layer_stats, layer_warnings = (
        build_layer_anchors_from_exact_omr_slots(analysis_result, omr_slots, omr_meta)
    )

    wave_profile = build_wave_profile(analysis_result)
    wave_by_page, wave_stats, wave_warnings = register_wave_profile(
        wave_profile, anchors_by_page, structural_by_page
    )
    pivot_profile = build_pivot_profile(wave_profile)
    pivot_by_page, pivot_stats, pivot_warnings = register_pivot_profile(
        pivot_profile, wave_by_page
    )
    tree_profile = build_tree_profile(wave_profile)
    tree_nodes_by_page, tree_branches_by_page, tree_stats, tree_warnings = register_tree_profile(
        tree_profile, wave_by_page
    )

    layer_stats.update(wave_stats)
    layer_stats.update(pivot_stats)
    layer_stats.update(tree_stats)
    warnings = list(omr_meta.get('warnings', []) or [])
    warnings.extend(layer_warnings)
    warnings.extend(wave_warnings)
    warnings.extend(pivot_warnings)
    warnings.extend(tree_warnings)

    for page_idx, anchors in anchors_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]['layer_anchors'] = anchors
    for page_idx, anchors in structural_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]['structural_anchors'] = anchors
    for page_idx, anchors in wave_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]['wave_anchors'] = anchors
    for page_idx, anchors in pivot_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]['pivot_anchors'] = anchors
    for page_idx, anchors in tree_nodes_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]['tree_nodes'] = anchors
    for page_idx, branches in tree_branches_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]['tree_branches'] = branches

    stats = dict(layer_stats)
    stats['validation_scope'] = (
        'symbolic-attacks-plus-independent-metric-grid-plus-exact-omr-display-slots-'
        'plus-read-only-wave-pivot-tree-display'
    )
    stats['active_registration_pipeline'] = (
        'symbolic-score-time->recursive-levels->exact-measure-offset-omr-slot->pdf-display'
    )
    stats['physical_geometry_timing_authority'] = False
    stats['physical_geometry_event_authority'] = False
    stats['musicxml_engraving_coordinates_used'] = False
    stats['registration_policy'] = (
        'exact symbolic measure+offset to existing OMR slot; no nearest/time/layout fallback'
    )
    layer_anchor_count = sum(len(page.get('layer_anchors', [])) for page in per_page.values())
    return {
        'available': layer_anchor_count > 0,
        'pages': [per_page[k] for k in sorted(per_page)],
        'max_level': int(layer_stats.get('max_layer_level', 0) or 0),
        'matching': stats,
        'warnings': warnings,
    }
