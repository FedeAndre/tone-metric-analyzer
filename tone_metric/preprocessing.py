"""Notation/OMR preprocessing contract for the v0.16 Tone-Metric engine.

The mathematical Levels engine accepts only ordinary sonic attacks on a fixed score-
time grid.  This module is the explicit boundary between notation interpretation and
mathematics.  No rule here is presented as part of the dissertation Levels algorithm.

Strict v0.16 policy:
- ties: parser/canonical recovery emits only the initial attack;
- grace notes and rests: no attack;
- simultaneous ordinary attacks: one merged hit;
- tuplets: attacks belonging only to a tuplet voice are excluded from the dissertation
  core; ordinary attacks in other voices at the same score time are retained;
- pickup placement and meter are fixed before Levels analysis begins.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

from .models import Hit, NoteAttack


def _source_is_tuplet(source: NoteAttack) -> bool:
    return bool(
        getattr(source, "tuplet_actual_notes", None)
        or getattr(source, "tuplet_normal_notes", None)
        or getattr(source, "tuplet_group", None)
    )


def prepare_hits_for_dissertation_core(hits: Iterable[Hit]) -> tuple[list[Hit], dict]:
    """Remove notation-only tuplet attacks without modifying ordinary simultaneous hits.

    This is intentionally conservative.  A hit with no source-note detail but explicit
    tuplet metadata is excluded entirely.  When source-note detail exists, only the
    tuplet sources are removed; the same physical/score-time hit survives if at least
    one ordinary source remains.
    """
    hits = list(hits)
    prepared: list[Hit] = []
    ignored_tuplet_hit_count = 0
    ignored_tuplet_source_count = 0
    retained_mixed_simultaneous_hit_count = 0
    metadata_only_tuplet_hit_count = 0

    for h in hits:
        sources = list(h.sources or [])
        if sources:
            ordinary = [s for s in sources if not _source_is_tuplet(s)]
            tuple_sources = [s for s in sources if _source_is_tuplet(s)]
            ignored_tuplet_source_count += len(tuple_sources)
            # If the canonical hit itself is explicitly a tuplet but the attached
            # symbolic sources do not identify which source belongs to it, strict
            # mode refuses to reinterpret that ambiguity as an ordinary attack.
            if getattr(h, "tuplet_arity", None) and not tuple_sources:
                metadata_only_tuplet_hit_count += 1
                ignored_tuplet_hit_count += 1
                continue
            if not ordinary:
                if tuple_sources:
                    ignored_tuplet_hit_count += 1
                    continue
                ordinary = sources
            elif tuple_sources:
                retained_mixed_simultaneous_hit_count += 1

            prepared.append(Hit(
                onset=h.onset,
                duration=max((s.duration for s in ordinary), default=h.duration),
                measure_index=h.measure_index,
                measure_number=h.measure_number,
                offset_in_measure=h.offset_in_measure,
                sources=ordinary,
                tuplet_arity=None,
                tuplet_group=None,
                tuplet_span_start=None,
                tuplet_span_end=None,
            ))
            continue

        # Canonical OMR can recover a column without symbolic source-note details.
        # Explicit tuplet metadata is still enough to keep it out of the core.
        if getattr(h, "tuplet_arity", None) or getattr(h, "tuplet_group", None):
            ignored_tuplet_hit_count += 1
            continue
        prepared.append(Hit(
            onset=h.onset,
            duration=h.duration,
            measure_index=h.measure_index,
            measure_number=h.measure_number,
            offset_in_measure=h.offset_in_measure,
            sources=[],
        ))

    # The parser/canonical stage should already have merged simultaneities.  Enforce
    # the contract deterministically here rather than trusting upstream details.
    grouped: dict[Fraction, list[Hit]] = {}
    for h in prepared:
        grouped.setdefault(h.onset, []).append(h)

    merged: list[Hit] = []
    remerged_count = 0
    for onset in sorted(grouped):
        rows = grouped[onset]
        if len(rows) == 1:
            merged.append(rows[0])
            continue
        remerged_count += len(rows) - 1
        first = rows[0]
        sources = [s for row in rows for s in row.sources]
        merged.append(Hit(
            onset=onset,
            duration=max((row.duration for row in rows), default=Fraction(0)),
            measure_index=first.measure_index,
            measure_number=first.measure_number,
            offset_in_measure=first.offset_in_measure,
            sources=sources,
        ))

    audit = {
        "contract": "notation-before-levels-v0.16",
        "input_hit_count": len(hits),
        "output_hit_count": len(merged),
        "tuplet_policy": "exclude-tuplet-voice-from-dissertation-core",
        "ignored_tuplet_hit_count": ignored_tuplet_hit_count,
        "ignored_tuplet_source_count": ignored_tuplet_source_count,
        "retained_mixed_simultaneous_hit_count": retained_mixed_simultaneous_hit_count,
        "metadata_only_tuplet_hit_count": metadata_only_tuplet_hit_count,
        "remerged_simultaneous_hit_count": remerged_count,
        "grace_policy": "ignored-upstream",
        "rest_policy": "no-attack-upstream",
        "tie_policy": "initial-attack-only-upstream",
        "spacing_based_arity_inference": False,
    }
    return merged, audit
