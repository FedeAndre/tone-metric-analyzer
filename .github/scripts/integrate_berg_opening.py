from pathlib import Path
import hashlib


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{path}: expected one replacement target, found {count}')
    p.write_text(text.replace(old, new), encoding='utf-8')

# 1) Carry an explicit opening-anacrusis flag in the authoritative score model.
replace_once(
    'tone_metric/models.py',
    '    implicit: bool = False\n\n    @property\n',
    '    implicit: bool = False\n    # True only when the notation itself supplies the conservative opening\n    # anacrusis cue used by the dissertation-aligned structural grid.  This is\n    # score metadata, not a second timing algorithm or a filename exception.\n    opening_anacrusis: bool = False\n\n    @property\n',
)
replace_once(
    'tone_metric/models.py',
    '    compound: bool\n    hits: List[Hit] = field(default_factory=list)\n',
    '    compound: bool\n    opening_anacrusis: bool = False\n    hits: List[Hit] = field(default_factory=list)\n',
)

# 2) Derive the flag from an explicit sectional opening barline in MusicXML.
replace_once(
    'tone_metric/musicxml.py',
    '        implicit = (measure.get("implicit") or "").lower() == "yes"\n        short_first = mi == 0 and max_cursor < full\n        pickup_shift = full - max_cursor if (implicit or short_first) and max_cursor < full else Fraction(0)\n        measure_templates.append({\n',
    '        implicit = (measure.get("implicit") or "").lower() == "yes"\n        short_first = mi == 0 and max_cursor < full\n        pickup_shift = full - max_cursor if (implicit or short_first) and max_cursor < full else Fraction(0)\n\n        # A metrically full opening can still carry an explicit sectional\n        # anacrusis cue.  Preserve only the concrete notation case of a first\n        # measure closed by a heavy-light sectional double barline without a\n        # repeat sign.  This supplies score metadata for one pre-entry Level-1\n        # articulation; no score title, filename, pitch, or hard-coded measure\n        # identity participates in the rule.\n        right_bar_style = ""\n        right_bar_has_repeat = False\n        for barline in measure.findall("barline"):\n            if (barline.get("location") or "right").strip().lower() != "right":\n                continue\n            right_bar_style = (barline.findtext("bar-style") or "").strip().lower()\n            right_bar_has_repeat = barline.find("repeat") is not None\n        opening_anacrusis = bool(\n            mi == 0\n            and pickup_shift == 0\n            and right_bar_style == "heavy-light"\n            and not right_bar_has_repeat\n        )\n        measure_templates.append({\n',
)
replace_once(
    'tone_metric/musicxml.py',
    '            "implicit": implicit,\n        })\n',
    '            "implicit": implicit,\n            "opening_anacrusis": opening_anacrusis,\n        })\n',
)
replace_once(
    'tone_metric/musicxml.py',
    '        actual_duration=m["actual"], pickup_shift=m["shift"], numerator=m["num"], denominator=m["den"], implicit=m["implicit"]\n',
    '        actual_duration=m["actual"], pickup_shift=m["shift"], numerator=m["num"], denominator=m["den"],\n        implicit=m["implicit"], opening_anacrusis=bool(m.get("opening_anacrusis", False))\n',
)

# 3) Integrate the pre-entry Level 1 into the single recursive structural grid.
# Existing event Levels are intentionally untouched: the dissertation shows one
# parenthetical L1 before the first sounding pickup, then the already-present L1.
replace_once(
    'tone_metric/engine.py',
    '                beat_count=beat_count, top_base=top_base, compound=compound,\n                hits=seg_hits, warnings=warns,\n',
    '                beat_count=beat_count, top_base=top_base, compound=compound,\n                opening_anacrusis=bool(start_i == 0 and getattr(measures[start_i], "opening_anacrusis", False)),\n                hits=seg_hits, warnings=warns,\n',
)
replace_once(
    'tone_metric/engine.py',
    '    event_rows = []\n',
    '    # Dissertation opening-anacrusis rule.  This is part of the authoritative\n    # structural grid, not a rendering patch: one silent Level-1 articulation\n    # precedes the first scored attack while all existing attack Levels retain\n    # their validated score-time assignments.\n    pre_entry_time = None\n    if segment.opening_anacrusis:\n        pre_entry_time = segment.start - beat\n        structural.setdefault(pre_entry_time, set()).add(1)\n\n    event_rows = []\n',
)
replace_once(
    'tone_metric/engine.py',
    "            'lowest_level': min(v),\n        }\n        for t, v in sorted(structural.items()) if segment.start <= t <= segment.end\n    ]\n",
    "            'lowest_level': min(v),\n            'opening_anacrusis_pre_entry': bool(pre_entry_time is not None and t == pre_entry_time),\n        }\n        for t, v in sorted(structural.items())\n        if segment.start <= t <= segment.end or (pre_entry_time is not None and t == pre_entry_time)\n    ]\n",
)
replace_once(
    'tone_metric/engine.py',
    "    for t, levels in sorted(structural.items()):\n        if not (segment.start <= t < segment.end):\n            continue\n",
    "    for t, levels in sorted(structural.items()):\n        if not (\n            segment.start <= t < segment.end\n            or (pre_entry_time is not None and t == pre_entry_time)\n        ):\n            continue\n",
)
replace_once(
    'tone_metric/engine.py',
    "        'compound': segment.compound,\n",
    "        'compound': segment.compound,\n        'opening_anacrusis': bool(segment.opening_anacrusis),\n        'opening_anacrusis_pre_entry_quarter': frac_to_str(pre_entry_time) if pre_entry_time is not None else None,\n",
)
replace_once(
    'tone_metric/engine.py',
    "                'pickup_shift_quarter': frac_to_str(m.pickup_shift),\n                'meter': f\"{m.numerator}/{m.denominator}\",\n",
    "                'pickup_shift_quarter': frac_to_str(m.pickup_shift),\n                'opening_anacrusis': bool(getattr(m, 'opening_anacrusis', False)),\n                'meter': f\"{m.numerator}/{m.denominator}\",\n",
)

# 4) Register that engine-generated negative-time structural point to the opening
# measure layout.  The point is parenthetical and cannot alter attack timing.
replace_once(
    'tone_metric/score_registration.py',
    '            "shift": shift,\n        })\n',
    '            "shift": shift,\n            "opening_anacrusis": bool(row.get("opening_anacrusis", False)),\n        })\n',
)
replace_once(
    'tone_metric/score_registration.py',
    '            measure = next((m for m in measures if m["start"] <= t < m["end"]), None)\n            if measure is None:\n                continue\n            offset = t - measure["start"]\n',
    '            pre_entry = bool(point.get("opening_anacrusis_pre_entry", False))\n            measure = next((m for m in measures if m["start"] <= t < m["end"]), None)\n            if measure is None and pre_entry and measures:\n                first = measures[0]\n                if first.get("opening_anacrusis") and t < first["start"]:\n                    measure = first\n            if measure is None:\n                continue\n            offset = t - measure["start"]\n',
)
replace_once(
    'tone_metric/score_registration.py',
    '                "offset": offset,\n            })\n',
    '                "offset": offset,\n                "opening_anacrusis_pre_entry": pre_entry,\n            })\n',
)
insert_before = 'def build_layer_anchors_from_canonical_score(analysis_result: dict, page_dimensions: dict[int, tuple[float, float]]):\n'
helper = '''def _opening_anacrusis_pre_entry_layout_position(measure_meta: dict) -> tuple[float | None, str, float]:\n    """Place the engine-generated silent Level-1 articulation before the first attack.\n\n    The score-time point is already fixed by the Levels engine.  This function only\n    maps it into the visible opening space between Audiveris' measure boundary and\n    the first canonical column.  It cannot create or move a sonic event.\n    """\n    left = float(measure_meta.get("stack_left", 0.0) or 0.0)\n    right = float(measure_meta.get("stack_right", left + 1.0) or (left + 1.0))\n    xs = []\n    for col in measure_meta.get("columns", []):\n        try:\n            xs.append(float(col.get("x_abs")))\n        except Exception:\n            pass\n    if xs:\n        first_x = min(xs)\n        if first_x > left + 2.0:\n            # Midway through the explicit pre-attack opening space keeps the label\n            # clear of both the meter signature and the first sounding notehead.\n            return left + 0.5 * (first_x - left), "opening-anacrusis-pre-entry-layout", 0.90\n    if right > left:\n        return left + 1.0, "opening-anacrusis-measure-layout-fallback", 0.62\n    return None, "opening-anacrusis-layout-unavailable", 0.0\n\n\n'''
replace_once('tone_metric/score_registration.py', insert_before, helper + insert_before)
replace_once(
    'tone_metric/score_registration.py',
    '        is_opening_pre_pickup = (\n            mi == 0\n            and m["shift"] > 0\n            and Fraction(0) <= offset < m["shift"]\n        )\n        if cm is None or ((local_time < 0 and not is_opening_pre_pickup) or local_time >= m["full"]):\n',
    '        is_opening_pre_pickup = (\n            mi == 0\n            and m["shift"] > 0\n            and Fraction(0) <= offset < m["shift"]\n        )\n        is_opening_pre_entry = bool(\n            mi == 0\n            and m.get("opening_anacrusis")\n            and target.get("opening_anacrusis_pre_entry")\n            and offset < 0\n        )\n        if cm is None or ((local_time < 0 and not (is_opening_pre_pickup or is_opening_pre_entry)) or local_time >= m["full"]):\n',
)
replace_once(
    'tone_metric/score_registration.py',
    '        if is_opening_pre_pickup:\n            x_abs, source, confidence = _pre_pickup_structural_layout_position(\n                cm, offset, m["shift"]\n            )\n            exact_col = None\n        else:\n            x_abs, source, confidence, exact_col = _structural_layout_position(cm, local_time)\n',
    '        if is_opening_pre_entry:\n            x_abs, source, confidence = _opening_anacrusis_pre_entry_layout_position(cm)\n            exact_col = None\n        elif is_opening_pre_pickup:\n            x_abs, source, confidence = _pre_pickup_structural_layout_position(\n                cm, offset, m["shift"]\n            )\n            exact_col = None\n        else:\n            x_abs, source, confidence, exact_col = _structural_layout_position(cm, local_time)\n',
)
replace_once(
    'tone_metric/score_registration.py',
    '        reason = (\n            "opening-pickup-pre-attack-metric-position"\n            if is_opening_pre_pickup\n            else _structural_reason(cm, local_time, exact_col)\n        )\n',
    '        reason = (\n            "opening-anacrusis-pre-entry-metric-position"\n            if is_opening_pre_entry\n            else (\n                "opening-pickup-pre-attack-metric-position"\n                if is_opening_pre_pickup\n                else _structural_reason(cm, local_time, exact_col)\n            )\n        )\n',
)

# Refresh the existing runtime checksum manifest for exactly the four modified
# modules.  No other path is touched by this integration script.
manifest = Path('v0152_sha256.txt')
lines = manifest.read_text(encoding='utf-8').splitlines()
changed = {
    'tone_metric/models.py',
    'tone_metric/musicxml.py',
    'tone_metric/engine.py',
    'tone_metric/score_registration.py',
}
out = []
for line in lines:
    parts = line.split(None, 1)
    if len(parts) == 2 and parts[1] in changed:
        digest = hashlib.sha256(Path(parts[1]).read_bytes()).hexdigest()
        out.append(f'{digest}  {parts[1]}')
    else:
        out.append(line)
manifest.write_text('\n'.join(out) + '\n', encoding='utf-8')

print('BERG OPENING INTEGRATION APPLIED')
