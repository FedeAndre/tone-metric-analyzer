from __future__ import annotations

"""Adapter from the clean optical-rhythm result to the validated Tone-Metric
Levels / wave / pivot / tree mathematics.

No score parsing or timing reconstruction occurs here.  This module accepts
only rational attacks that were already proven by rhythm_reconstructor.
"""

from fractions import Fraction

from tone_metric.engine import analyze as analyze_levels
from tone_metric.models import Hit, MeasureInfo, NoteAttack
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tone_metric.waves import build_wave_profile


def _frac(value) -> Fraction:
    return Fraction(str(value))


def analyze_tone_metric(rhythm: dict) -> dict:
    if not bool(rhythm.get("rhythmic_attacks_ready")):
        raise ValueError(
            "Tone-Metric analysis requires a fully resolved exact attack timeline."
        )

    meter = rhythm.get("meter", {})
    numerator = int(meter["numerator"])
    denominator = int(meter["denominator"])
    full_duration = _frac(meter["measure_duration_quarter"])

    measures: list[MeasureInfo] = []
    for row in rhythm.get("measures", []):
        start = _frac(row["start_quarter"])
        pickup_shift = _frac(row.get("pickup_shift_quarter", "0"))
        actual = full_duration - pickup_shift if pickup_shift > 0 else full_duration
        measures.append(
            MeasureInfo(
                index=int(row["measure_index"]),
                number=str(row.get("measure_number", int(row["measure_index"]) + 1)),
                start=start,
                full_duration=full_duration,
                actual_duration=actual,
                pickup_shift=pickup_shift,
                numerator=numerator,
                denominator=denominator,
                implicit=pickup_shift > 0,
                opening_anacrusis=bool(
                    int(row["measure_index"]) == 0 and pickup_shift > 0
                ),
            )
        )

    hits: list[Hit] = []
    for row in rhythm.get("hits", []):
        measure_index = int(row["measure_index"])
        onset = _frac(row["onset_quarter"])
        offset = _frac(row["offset_in_measure_quarter"])
        sources: list[NoteAttack] = []
        source_durations: list[Fraction] = []
        for index, source in enumerate(row.get("sources", [])):
            duration = _frac(source.get("duration_quarter", "0"))
            source_durations.append(duration)
            notehead_ids = source.get("notehead_ids", []) or []
            visual_name = (
                "visual:" + ",".join(str(value) for value in notehead_ids)
                if notehead_ids
                else f"visual-source-{index}"
            )
            sources.append(
                NoteAttack(
                    onset=onset,
                    duration=duration,
                    measure_index=measure_index,
                    measure_number=str(row.get("measure_number", measure_index + 1)),
                    offset_in_measure=offset,
                    part_id=f"page-{source.get('page', 0)}",
                    voice=str(source.get("voice", "visual")),
                    staff=str(source.get("staff_id", "")),
                    pitch=visual_name,
                    tie_start=False,
                    tie_stop=False,
                    tuplet_actual=None,
                    tuplet_normal=None,
                    page_index=int(source.get("page", 0) or 0),
                    x_norm=None,
                    y_norm=None,
                )
            )
        hit_duration = min(source_durations) if source_durations else Fraction(0)
        hits.append(
            Hit(
                onset=onset,
                duration=hit_duration,
                measure_index=measure_index,
                measure_number=str(row.get("measure_number", measure_index + 1)),
                offset_in_measure=offset,
                sources=sources,
                canonical_recovered=False,
            )
        )

    levels = analyze_levels(hits, measures)
    wave = build_wave_profile(levels)
    pivots = build_pivot_profile(wave)
    trees = build_tree_profile(wave)

    max_level = max(
        (
            int(event.get("tone_metric_height", 0))
            for segment in levels.get("segments", [])
            for event in segment.get("events", [])
        ),
        default=0,
    )
    return {
        "engine": "tma-clean-exact-attacks-v1",
        "timing_source": "clean-rational-constraint-solver",
        "semantic_timing_used": False,
        "x_position_used_as_time": False,
        "meter": rhythm["meter"],
        "levels": levels,
        "wave": wave,
        "pivots": pivots,
        "trees": trees,
        "stats": {
            "measure_count": len(measures),
            "hit_count": len(hits),
            "max_level": max_level,
            "wave_points": len(wave),
            "pivot_count": len(pivots),
            "tree_nodes": len(trees.get("nodes", [])),
            "tree_branches": len(trees.get("branches", [])),
        },
        "ready": True,
    }
