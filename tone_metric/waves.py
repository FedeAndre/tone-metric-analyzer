from __future__ import annotations

"""Tone-Metric wave derivation from actual attacks only.

The recursive Levels engine remains authoritative.  The wave is sampled only at
new note/chord attacks already present in the analyzed event stream.  Sustained
notes, augmentation-dot continuations, ties, rests, barlines, accidentals, and
empty metric positions can never become wave points or alter pivot derivation.
"""

from collections import defaultdict
from fractions import Fraction


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def build_wave_profile(analysis_result: dict) -> list[dict]:
    """Return an attack-only Tone-Metric wave in score time."""
    out = []
    seen = set()
    for si, seg in enumerate(analysis_result.get("segments", [])):
        for event in seg.get("events", []):
            levels = sorted({int(x) for x in event.get("tone_metric_levels", []) if int(x) > 0})
            if not levels:
                continue
            try:
                mi = int(event.get("measure_index", -1))
            except Exception:
                continue
            off = _frac(event.get("offset_in_measure_quarter"))
            onset = _frac(event.get("onset_quarter"))
            if off is None or onset is None:
                continue
            key = (mi, off)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "segment_index": si,
                "measure_index": mi,
                "measure_number": event.get("measure_number"),
                "onset_quarter": str(event.get("onset_quarter")),
                "offset_in_measure_quarter": str(event.get("offset_in_measure_quarter")),
                "levels": levels,
                "height": max(levels),
                "density": len(levels),
                "lowest_level": min(levels),
                "attack": True,
                "parenthetical": False,
                "wave_source": "actual-attacks-only-recursive-level-envelope",
            })
    out.sort(key=lambda r: (
        int(r.get("segment_index", 0)),
        _frac(r.get("onset_quarter"), Fraction(0)),
        int(r.get("measure_index", -1)),
    ))
    return out


def register_wave_profile(
    wave_profile: list[dict],
    attack_anchors_by_page: dict[int, list[dict]],
    structural_anchors_by_page: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict, list[str]]:
    """Attach every wave point to its already-validated exact attack anchor."""
    lookup = {}
    duplicate_keys = []
    for page, rows in attack_anchors_by_page.items():
        for r in rows:
            mi = int(r.get("measure_index", -1))
            off = _frac(r.get("offset_in_measure_quarter"))
            if off is None:
                continue
            key = (mi, off)
            if key in lookup:
                duplicate_keys.append(key)
                continue
            lookup[key] = (int(page), r)

    by_page = defaultdict(list)
    missing = []
    for p in wave_profile:
        mi = int(p.get("measure_index", -1))
        off = _frac(p.get("offset_in_measure_quarter"))
        if off is None:
            missing.append({**p, "reason": "invalid-wave-offset"})
            continue
        found = lookup.get((mi, off))
        if found is None:
            missing.append({**p, "reason": "validated-attack-anchor-not-found"})
            continue
        page, anchor = found
        row = {
            **{k: v for k, v in p.items() if k not in {"density", "lowest_level"}},
            "page_index": int(anchor.get("page_index", page)),
            "system_index": int(anchor.get("system_index", 0)),
            "physical_system_index": int(anchor.get("physical_system_index", anchor.get("system_index", 0))),
            "cx_norm": float(anchor.get("cx_norm")),
            "cy_norm": float(anchor.get("cy_norm", 0.0)),
            "recovered_x_abs": float(anchor.get("recovered_x_abs")),
            "registration_source": "existing-exact-attacking-notehead-anchor",
            "anchor_registration_source": anchor.get("registration_source"),
            "visual_position_is_estimated": False,
        }
        by_page[row["page_index"]].append(row)

    for rows in by_page.values():
        rows.sort(key=lambda r: (
            int(r.get("system_index", 0)),
            int(r.get("measure_index", -1)),
            _frac(r.get("offset_in_measure_quarter"), Fraction(0)),
        ))

    warnings = []
    if duplicate_keys:
        warnings.append(
            f"{len(duplicate_keys)} duplicate exact attack-anchor key(s) were found while registering the wave profile."
        )
    if missing:
        warnings.append(
            f"{len(missing)} attack-only wave point(s) could not be matched to an existing exact attack anchor."
        )

    heights = [int(p.get("height", 0)) for p in wave_profile]
    stats = {
        "wave_profile_points_expected": len(wave_profile),
        "wave_profile_points_mapped": sum(len(v) for v in by_page.values()),
        "wave_profile_points_missing": len(missing),
        "wave_attack_points_mapped": sum(len(v) for v in by_page.values()),
        "wave_parenthetical_points_mapped": 0,
        "wave_max_height": max(heights, default=0),
        "wave_registration": "attack-only-score-time-profile->exact-attacking-notehead-anchors",
        "wave_coordinate_synthesis": False,
        "wave_missing_points": missing,
        "wave_duplicate_existing_anchor_keys": [f"{mi}:{off}" for mi, off in duplicate_keys],
        "analysis_position_policy": "actual-attacks-only",
    }
    return dict(by_page), stats, warnings
