from __future__ import annotations

"""Register symbolic tone-metric events to exact Audiveris noteheads.

Musical time is never reconstructed from graphics in this module. The event list
has already been established by ``musicxml.parse_musicxml`` and globally merged by
exact symbolic onset before Levels are computed.

For PDF placement we read the saved Audiveris ``.omr`` project only as a semantic
geometry graph:

    symbolic (measure, exact offset)
        -> Audiveris stack slot with the same persisted time-offset
        -> voice slot entry whose status is BEGIN
        -> referenced head-chord
        -> containment relation
        -> exact head bounds

The Audiveris slot time is used only as an exact lookup key for an already-existing
symbolic event. It can never add, remove, split, merge, reorder, or retime an event.
No x-ordering, x-clustering, tolerance-based onset reconstruction, nearest-column
matching, MusicXML default-x timing, or canonical-score recovery is allowed here.
"""

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile
import re

from lxml import etree


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _frac(value, default=None):
    try:
        return Fraction(str(value))
    except Exception:
        return default


def _intish(value, default=None):
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _sheet_number(member: str) -> int:
    m = re.search(r"sheet#(\d+)", member, re.I)
    return int(m.group(1)) if m else 10**9


def _bounds(el) -> dict | None:
    if el is None:
        return None
    for child in el:
        if _local(child.tag) == "bounds":
            x = _intish(child.get("x"))
            y = _intish(child.get("y"))
            w = _intish(child.get("w"))
            h = _intish(child.get("h"))
            if None not in (x, y, w, h) and w > 0 and h > 0:
                return {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                }
    return None


def build_symbolic_registration_meta(omr_path: str | Path) -> dict:
    """Read exact slot/chord/head geometry from a saved Audiveris project.

    This function does not receive or construct musical events. It produces a
    geometry lookup table that can later be joined to the already-authoritative
    symbolic event list by exact ``(measure_index, offset_quarter)`` equality.
    """
    path = Path(omr_path)
    meta = {
        "available": False,
        "architecture": "symbolic-time-authoritative-semantic-omr-geometry-only",
        "timing_authority": "musicxml-symbolic-global-onset",
        "geometry_source": "audiveris-slot->voice-BEGIN->head-chord->contained-head",
        "slot_rows": [],
        "slot_count": 0,
        "slots_with_begin_chords": 0,
        "slots_with_exact_heads": 0,
        "begin_chord_count": 0,
        "exact_head_count": 0,
        "duplicate_time_keys": 0,
        "warnings": [],
    }
    if not path.exists():
        meta["warnings"].append("Saved Audiveris .omr project was not found.")
        return meta

    global_measure_index = 0
    global_page_index = 0
    seen_keys: set[tuple[int, Fraction]] = set()

    try:
        with ZipFile(path, "r") as zf:
            members = sorted(
                [n for n in zf.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
                key=lambda n: (_sheet_number(n), n.lower()),
            )
            if not members:
                meta["warnings"].append("No Audiveris sheet XML was found in the saved .omr project.")
                return meta

            for member in members:
                try:
                    root = etree.fromstring(zf.read(member))
                except Exception as exc:
                    meta["warnings"].append(f"Could not parse {member}: {exc}")
                    continue

                pages = [el for el in root.iter() if _local(el.tag) == "page"] or [root]
                for page in pages:
                    systems = [el for el in page if _local(el.tag) == "system"]
                    if not systems:
                        systems = [el for el in page.iter() if _local(el.tag) == "system"]

                    for system_index, system in enumerate(systems):
                        stacks = [el for el in system if _local(el.tag) == "stack"]
                        parts = [el for el in system if _local(el.tag) == "part"]
                        part_measures = [
                            [el for el in part if _local(el.tag) == "measure"]
                            for part in parts
                        ]

                        sig = next((el for el in system if _local(el.tag) == "sig"), None)
                        id_map = {}
                        relations = []
                        if sig is not None:
                            for el in sig.iter():
                                if el.get("id"):
                                    id_map[str(el.get("id"))] = el
                                if _local(el.tag) == "relation":
                                    relations.append(el)

                        chord_to_heads: dict[str, list[str]] = defaultdict(list)
                        for rel in relations:
                            if not any(_local(c.tag) == "containment" for c in rel):
                                continue
                            source = str(rel.get("source") or "")
                            target = str(rel.get("target") or "")
                            source_el = id_map.get(source)
                            target_el = id_map.get(target)
                            if source_el is None or target_el is None:
                                continue
                            if _local(source_el.tag) != "head-chord" or _local(target_el.tag) != "head":
                                continue
                            chord_to_heads[source].append(target)

                        for stack_index, stack in enumerate(stacks):
                            slot_time: dict[int, Fraction] = {}
                            slot_x_abs: dict[int, int] = {}
                            left = _intish(stack.get("left"), 0)
                            for slot in stack:
                                if _local(slot.tag) != "slot":
                                    continue
                                slot_id = _intish(slot.get("id"))
                                time_whole = _frac(slot.get("time-offset"))
                                x_offset = _intish(slot.get("x-offset"))
                                if slot_id is None or time_whole is None or x_offset is None:
                                    continue
                                slot_time[slot_id] = time_whole * 4
                                slot_x_abs[slot_id] = left + x_offset

                            begin_chords_by_slot: dict[int, list[str]] = defaultdict(list)
                            for measures in part_measures:
                                if stack_index >= len(measures):
                                    continue
                                measure = measures[stack_index]
                                for voice in [el for el in measure if _local(el.tag) == "voice"]:
                                    for slots_el in [el for el in voice if _local(el.tag) == "slots"]:
                                        for entry in [el for el in slots_el if _local(el.tag) == "entry"]:
                                            key_el = next((el for el in entry if _local(el.tag) == "key"), None)
                                            value_el = next((el for el in entry if _local(el.tag) == "value"), None)
                                            if key_el is None or value_el is None:
                                                continue
                                            slot_id = _intish(key_el.text)
                                            if slot_id is None:
                                                continue
                                            if str(value_el.get("status") or "").upper() != "BEGIN":
                                                continue
                                            chord_id = str(value_el.get("chord") or "")
                                            if chord_id:
                                                begin_chords_by_slot[slot_id].append(chord_id)

                            for slot_id, offset_quarter in sorted(slot_time.items(), key=lambda kv: (kv[1], kv[0])):
                                chord_ids = sorted(
                                    set(begin_chords_by_slot.get(slot_id, [])),
                                    key=lambda x: (_intish(x, 10**9), x),
                                )
                                heads = []
                                for chord_id in chord_ids:
                                    for head_id in sorted(
                                        set(chord_to_heads.get(chord_id, [])),
                                        key=lambda x: (_intish(x, 10**9), x),
                                    ):
                                        head_el = id_map.get(head_id)
                                        box = _bounds(head_el)
                                        if box is None:
                                            continue
                                        heads.append({
                                            "head_id": head_id,
                                            "chord_id": chord_id,
                                            **box,
                                        })

                                key = (global_measure_index, offset_quarter)
                                if key in seen_keys:
                                    meta["duplicate_time_keys"] += 1
                                    meta["warnings"].append(
                                        f"Duplicate OMR slot time key at measure {global_measure_index + 1}, offset {offset_quarter}."
                                    )
                                seen_keys.add(key)

                                meta["slot_rows"].append({
                                    "global_measure_index": global_measure_index,
                                    "page_index": global_page_index,
                                    "system_index": system_index,
                                    "stack_index_in_system": stack_index,
                                    "stack_id": str(stack.get("id") or ""),
                                    "slot_id": slot_id,
                                    "offset_quarter": str(offset_quarter),
                                    "slot_x_abs": slot_x_abs.get(slot_id),
                                    "begin_chord_ids": chord_ids,
                                    "heads": heads,
                                })
                                meta["slot_count"] += 1
                                if chord_ids:
                                    meta["slots_with_begin_chords"] += 1
                                    meta["begin_chord_count"] += len(chord_ids)
                                if heads:
                                    meta["slots_with_exact_heads"] += 1
                                    meta["exact_head_count"] += len(heads)

                            global_measure_index += 1
                    global_page_index += 1
    except Exception as exc:
        meta["warnings"].append(f"Could not read semantic OMR registration graph: {exc}")
        return meta

    meta["available"] = bool(meta["slot_rows"]) and meta["duplicate_time_keys"] == 0
    return meta


def _event_targets(analysis_result: dict) -> list[dict]:
    rows = []
    for segment_index, segment in enumerate(analysis_result.get("segments", [])):
        for event in segment.get("events", []):
            levels = sorted({int(x) for x in event.get("tone_metric_levels", []) if int(x) > 0})
            if not levels:
                continue
            onset = _frac(event.get("onset_quarter"))
            offset = _frac(event.get("offset_in_measure_quarter"))
            if onset is None or offset is None:
                continue
            rows.append({
                "segment_index": segment_index,
                "event": event,
                "onset": onset,
                "offset": offset,
                "levels": levels,
            })
    return rows


def build_layer_anchors_from_symbolic_slots(
    analysis_result: dict,
    page_dimensions: dict[int, tuple[float, float]],
):
    """Join already-established symbolic events to exact semantic notehead geometry."""
    meta = analysis_result.get("_symbolic_registration_meta") or {}
    by_key = {}
    duplicate_keys = set()
    for row in meta.get("slot_rows", []):
        mi = _intish(row.get("global_measure_index"), -1)
        offset = _frac(row.get("offset_quarter"))
        if mi < 0 or offset is None:
            continue
        key = (mi, offset)
        if key in by_key:
            duplicate_keys.add(key)
            continue
        by_key[key] = row
    for key in duplicate_keys:
        by_key.pop(key, None)

    anchors_by_page = defaultdict(list)
    per_level = defaultdict(lambda: {"expected": 0, "mapped": 0, "missing": 0})
    missing = defaultdict(list)
    event_targets = _event_targets(analysis_result)
    mapped_events = 0
    missing_events = 0
    exact_head_anchors = 0

    for target in event_targets:
        event = target["event"]
        mi = _intish(event.get("measure_index"), -1)
        offset = target["offset"]
        levels = target["levels"]
        for level in levels:
            per_level[level]["expected"] += 1

        row = by_key.get((mi, offset))
        reason = None
        if row is None:
            reason = "exact-omr-slot-time-key-not-found"
        elif not row.get("heads"):
            reason = "slot-has-no-BEGIN-chord-contained-head"

        if reason is not None:
            missing_events += 1
            for level in levels:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": reason,
                })
            continue

        slot_x = row.get("slot_x_abs")
        try:
            slot_x_f = float(slot_x)
        except Exception:
            slot_x_f = None

        def head_key(head: dict):
            try:
                cx = float(head.get("cx"))
                cy = float(head.get("cy"))
            except Exception:
                return (float("inf"), float("inf"), float("inf"), str(head.get("head_id") or ""))
            dx = abs(cx - slot_x_f) if slot_x_f is not None else 0.0
            return (dx, cx, cy, str(head.get("head_id") or ""))

        head = min(row["heads"], key=head_key)
        try:
            x_abs = float(head["cx"])
            y_abs = float(head["cy"])
        except Exception:
            missing_events += 1
            for level in levels:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": "contained-head-bounds-invalid",
                })
            continue

        page = _intish(row.get("page_index"), -1)
        system = _intish(row.get("system_index"), -1)
        dims = page_dimensions.get(page)
        if not dims or float(dims[0]) <= 0 or float(dims[1]) <= 0:
            missing_events += 1
            for level in levels:
                per_level[level]["missing"] += 1
                missing[level].append({
                    "measure_index": mi,
                    "measure_number": event.get("measure_number", str(mi + 1)),
                    "offset_in_measure_quarter": str(offset),
                    "reason": "page-dimensions-unavailable",
                })
            continue

        pw, ph = float(dims[0]), float(dims[1])
        mapped_events += 1
        exact_head_anchors += 1
        anchors_by_page[page].append({
            "page_index": page,
            "system_index": system,
            "physical_system_index": system,
            "segment_index": int(target["segment_index"]),
            "event_index": _intish(event.get("event_index"), -1),
            "measure_index": mi,
            "measure_number": event.get("measure_number", str(mi + 1)),
            "onset_quarter": event.get("onset_quarter"),
            "offset_in_measure_quarter": event.get("offset_in_measure_quarter"),
            "attack_key": event.get("attack_key", f"{mi}:{offset}"),
            "duration_quarter": event.get("duration_quarter", ""),
            "cx_norm": x_abs / pw,
            "cy_norm": y_abs / ph,
            "height": max(levels),
            "lowest_level": min(levels),
            "levels": levels,
            "parenthetical": False,
            "registration_source": "audiveris-voice-BEGIN-chord-contained-notehead",
            "timing_source": "musicxml-symbolic-global-onset",
            "geometry_time_key_source": "exact-audiveris-slot-time-offset-match",
            "slot_id": row.get("slot_id"),
            "slot_x_abs": slot_x,
            "head_id": head.get("head_id"),
            "chord_id": head.get("chord_id"),
            "simultaneous_begin_chord_ids": list(row.get("begin_chord_ids") or []),
            "simultaneous_head_ids": [h.get("head_id") for h in row.get("heads", [])],
            "visual_position_is_estimated": False,
        })
        for level in levels:
            per_level[level]["mapped"] += 1

    for rows in anchors_by_page.values():
        rows.sort(key=lambda a: (
            int(a.get("system_index", 0)),
            int(a.get("measure_index", -1)),
            _frac(a.get("offset_in_measure_quarter"), Fraction(0)),
        ))

    warnings = list(meta.get("warnings", []))
    if duplicate_keys:
        warnings.append(
            f"{len(duplicate_keys)} duplicate OMR measure/time key(s) were rejected rather than guessed."
        )
    if missing_events:
        warnings.append(
            f"{missing_events} symbolic event(s) could not be mapped to an exact semantic OMR notehead and were left unmapped."
        )

    missing_total = sum(v["missing"] for v in per_level.values())
    levels_enabled = sorted(per_level)
    stats = {
        "final_layer_anchors": sum(len(v) for v in anchors_by_page.values()),
        "layer_anchor_registration": "symbolic-event->exact-omr-slot->voice-BEGIN->head-chord->contained-notehead",
        "analysis_position_policy": analysis_result.get(
            "analysis_position_policy", "symbolic-attacks-plus-independent-metric-grid"
        ),
        "timing_authority": "musicxml-symbolic-global-onset",
        "geometry_authority": "audiveris-semantic-slot/chord/head-relations",
        "x_order_used_for_timing": False,
        "x_clustering_used_for_timing": False,
        "nearest_graphical_symbol_fallback": False,
        "levels_enabled": levels_enabled,
        "max_layer_level": max(levels_enabled, default=0),
        "symbolic_events_expected": len(event_targets),
        "symbolic_events_mapped": mapped_events,
        "symbolic_events_unmapped": missing_events,
        "layer_hits_expected": sum(v["expected"] for v in per_level.values()),
        "layer_hits_mapped": sum(v["mapped"] for v in per_level.values()),
        "layer_hits_without_visual_attack": missing_total,
        "exact_semantic_notehead_anchors": exact_head_anchors,
        "omr_slot_count": int(meta.get("slot_count", 0) or 0),
        "omr_slots_with_exact_heads": int(meta.get("slots_with_exact_heads", 0) or 0),
        "duplicate_time_keys": len(duplicate_keys),
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


# Temporary import compatibility for physical.py only. The old canonical-score
# implementation is removed from this branch; this alias points exclusively to the
# symbolic-slot implementation above and contains no legacy timing behavior.
build_layer_anchors_from_canonical_score = build_layer_anchors_from_symbolic_slots
