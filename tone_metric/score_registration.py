from __future__ import annotations

"""Register analyzed symbolic attacks to PDF layout without deriving musical time.

Musical event identity is fixed upstream by ``parse_musicxml`` and the recursive
Tone-Metric engine.  This module may only attach already-existing events to the
layout metadata exported with those same symbolic notes.  It never creates,
removes, splits, merges, reorders, or retimes attacks.

The registration key is the symbolic ``measure_index:offset`` attack key.  Visual
x coordinates are used only after that key is established.  Multiple engraved
noteheads belonging to the same simultaneous global attack may have different x
positions; they therefore remain one event and merely provide several candidate
visual anchors for that already-fixed event.
"""

from collections import defaultdict
from fractions import Fraction
from statistics import median


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _event_targets(analysis_result: dict) -> list[dict]:
    rows = []
    for si, segment in enumerate(analysis_result.get("segments", [])):
        for event in segment.get("events", []):
            levels = sorted({int(x) for x in event.get("tone_metric_levels", []) if int(x) > 0})
            if not levels:
                continue
            key = str(event.get("attack_key") or "").strip()
            if not key:
                try:
                    mi = int(event.get("measure_index", -1))
                    off = _frac(event.get("offset_in_measure_quarter"))
                    if off is None:
                        continue
                    key = f"{mi}:{off}"
                except Exception:
                    continue
            rows.append({"segment_index": si, "event": event, "levels": levels, "attack_key": key})
    return rows


def _visual_candidates(analysis_result: dict) -> tuple[dict[str, list[dict]], dict[int, int]]:
    """Index MusicXML visual note groups by already-established symbolic attack key."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    systems_by_page: dict[int, set[int]] = defaultdict(set)
    for row in analysis_result.get("_symbolic_visual_groups", []) or []:
        try:
            page = int(row.get("layout_page", 0))
            system = int(row.get("layout_system", 0))
        except Exception:
            continue
        systems_by_page[page].add(system)
        if not bool(row.get("analyzed")):
            continue
        key = str(row.get("attack_key") or "").strip()
        if not key:
            continue
        try:
            x_abs = float(row.get("layout_x_abs"))
            page_width = float(row.get("layout_page_width"))
        except Exception:
            continue
        if page_width <= 0:
            continue
        copy = dict(row)
        copy["_page"] = page
        copy["_system"] = system
        copy["_x_abs"] = x_abs
        copy["_page_width"] = page_width
        by_key[key].append(copy)
    system_counts = {page: len(systems) for page, systems in systems_by_page.items() if systems}
    return dict(by_key), system_counts


def _representative_candidate(candidates: list[dict]) -> dict | None:
    """Choose one *existing* symbolic note position for a global simultaneous attack.

    The median is used only to select among real candidate note positions.  No new
    coordinate is averaged or synthesized between noteheads.
    """
    if not candidates:
        return None
    xs = [float(c["_x_abs"]) for c in candidates]
    mid = median(xs)
    return min(
        candidates,
        key=lambda c: (
            abs(float(c["_x_abs"]) - mid),
            int(c.get("_page", 0)),
            int(c.get("_system", 0)),
            float(c["_x_abs"]),
            int(c.get("serial", 0)),
        ),
    )


def build_layer_anchors_from_canonical_score(
    analysis_result: dict,
    page_dimensions: dict[int, tuple[float, float]],
):
    """Compatibility entry point: register from symbolic attack keys, never canonical time.

    The historical function name is retained only to avoid an interface break in the
    physical rendering module.  No canonical-score reconstruction is consulted.
    """
    visual_by_key, system_counts = _visual_candidates(analysis_result)

    anchors_by_page = defaultdict(list)
    per_level = defaultdict(lambda: {"expected": 0, "mapped": 0, "missing": 0})
    missing = defaultdict(list)
    mapped_events = 0
    multi_position_events = 0
    max_candidate_spread = 0.0

    for target in _event_targets(analysis_result):
        event = target["event"]
        key = target["attack_key"]
        levels = target["levels"]
        for level in levels:
            per_level[level]["expected"] += 1

        candidates = visual_by_key.get(key, [])
        if len(candidates) > 1:
            multi_position_events += 1
            xs = [float(c["_x_abs"]) for c in candidates]
            max_candidate_spread = max(max_candidate_spread, max(xs) - min(xs))
        chosen = _representative_candidate(candidates)
        if chosen is None:
            for level in levels:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": event.get("measure_index"),
                    "measure_number": event.get("measure_number"),
                    "offset_in_measure_quarter": event.get("offset_in_measure_quarter"),
                    "attack_key": key,
                    "reason": "symbolic-note-layout-not-found",
                })
            continue

        page = int(chosen["_page"])
        system = int(chosen["_system"])
        dims = page_dimensions.get(page)
        if not dims or float(dims[0]) <= 0 or float(dims[1]) <= 0:
            for level in levels:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": event.get("measure_index"),
                    "measure_number": event.get("measure_number"),
                    "offset_in_measure_quarter": event.get("offset_in_measure_quarter"),
                    "attack_key": key,
                    "reason": "page-dimensions-unavailable",
                })
            continue

        pw, ph = float(dims[0]), float(dims[1])
        x_norm = max(0.0, min(1.0, float(chosen["_x_abs"]) / float(chosen["_page_width"])))
        # Audiveris commonly omits default-y in MusicXML.  Vertical placement is
        # therefore deliberately system-level only.  The actual analysis label rows
        # are laid out against the physical system bounds later; this y value cannot
        # affect event identity or score time.
        n_systems = max(1, int(system_counts.get(page, system + 1)))
        cy_norm = max(0.0, min(1.0, (system + 0.5) / n_systems))

        row = {
            "page_index": page,
            "system_index": system,
            "physical_system_index": system,
            "segment_index": int(target["segment_index"]),
            "event_index": int(event.get("event_index", -1) or -1),
            "measure_index": int(event.get("measure_index", -1)),
            "measure_number": event.get("measure_number"),
            "onset_quarter": event.get("onset_quarter"),
            "offset_in_measure_quarter": event.get("offset_in_measure_quarter"),
            "attack_key": key,
            "duration_quarter": event.get("duration_quarter", ""),
            "cx_norm": x_norm,
            "cy_norm": cy_norm,
            "height": max(levels),
            "lowest_level": min(levels),
            "levels": levels,
            "parenthetical": False,
            "registration_source": "symbolic-musicxml-note-layout-after-attack-key",
            "timing_source": "musicxml-symbolic-global-onset",
            "timing_confidence": "symbolic-exact",
            "recovered_x_abs": x_norm * pw,
            "visual_position_is_estimated": False,
            "vertical_position_is_system_estimate": True,
            "candidate_visual_note_count": len(candidates),
            "candidate_layout_x_abs": [float(c["_x_abs"]) for c in candidates],
        }
        anchors_by_page[page].append(row)
        mapped_events += 1
        for level in levels:
            per_level[level]["mapped"] += 1

    for rows in anchors_by_page.values():
        rows.sort(key=lambda a: (
            int(a.get("system_index", 0)),
            int(a.get("measure_index", -1)),
            _frac(a.get("offset_in_measure_quarter"), Fraction(0)),
        ))

    warnings: list[str] = []
    missing_total = sum(v["missing"] for v in per_level.values())
    if missing_total:
        warnings.append(
            f"{missing_total} event-level label(s) could not be attached to a symbolic note-layout position; no geometric fallback was used."
        )

    attack_levels = sorted(per_level)
    stats = {
        "final_layer_anchors": sum(len(v) for v in anchors_by_page.values()),
        "layer_anchor_registration": "symbolic-attack-key->musicxml-note-layout",
        "registration_timing_authority": "musicxml-symbolic-global-onset",
        "registration_can_create_attacks": False,
        "registration_can_retime_attacks": False,
        "analysis_position_policy": "symbolic-attacks-plus-independent-metric-grid",
        "levels_enabled": attack_levels,
        "max_layer_level": max(attack_levels, default=0),
        "layer_hits_expected": sum(v["expected"] for v in per_level.values()),
        "layer_hits_mapped": sum(v["mapped"] for v in per_level.values()),
        "layer_hits_without_visual_attack": missing_total,
        "symbolic_events_mapped": mapped_events,
        "symbolic_visual_attack_keys_available": len(visual_by_key),
        "simultaneous_events_with_multiple_visual_positions": multi_position_events,
        "max_simultaneous_visual_x_spread": max_candidate_spread,
        "manufactured_x_coordinates": 0,
        "semantic_attack_column_fallbacks": 0,
        "per_level": {str(k): dict(v) for k, v in sorted(per_level.items())},
        "missing_targets_by_level": {str(k): v for k, v in sorted(missing.items())},
        "structural_parenthetical_positions": 0,
        "structural_parenthetical_labels_expected": 0,
        "structural_parenthetical_labels_mapped": 0,
        "structural_parenthetical_labels_missing": 0,
        "structural_per_level": {},
        "structural_missing_targets_by_level": {},
        "structural_registration_sources": {},
        "structural_reasons": {},
    }
    for level, values in per_level.items():
        stats[f"level{level}_anchors_mapped"] = values["mapped"]
        stats[f"level{level}_hits_expected"] = values["expected"]
        stats[f"level{level}_hits_without_visual_attack"] = values["missing"]

    return dict(anchors_by_page), {}, stats, warnings
