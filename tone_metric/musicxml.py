from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile
from lxml import etree

from .models import NoteAttack, Hit, MeasureInfo, frac_to_str


def _read_musicxml_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".mxl":
        with ZipFile(path, "r") as zf:
            rootfile = None
            try:
                container = etree.fromstring(zf.read("META-INF/container.xml"))
                ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                node = container.find(".//c:rootfile", namespaces=ns)
                if node is not None:
                    rootfile = node.get("full-path")
            except Exception:
                rootfile = None
            if not rootfile:
                candidates = [n for n in zf.namelist() if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF/")]
                if not candidates:
                    raise ValueError("No MusicXML document found inside .mxl archive")
                rootfile = candidates[0]
            return zf.read(rootfile)
    return path.read_bytes()


def _text_int(node, xpath: str, default: int = 0) -> int:
    el = node.find(xpath)
    if el is None or el.text is None:
        return default
    return int(el.text.strip())


def _text_float(node, xpath: str, default: float) -> float:
    el = node.find(xpath)
    if el is None or el.text is None:
        return default
    try:
        return float(el.text.strip())
    except Exception:
        return default




def _normalized_time_signature(time, current_num: int, current_den: int, warnings: List[str] | None = None) -> tuple[int, int]:
    """Read a MusicXML <time> element and honor semantic common/cut symbols.

    Audiveris may preserve the semantic symbol even when the exported beats/beat-type
    pair is generic.  For tone-metric analysis the notated tactus is decisive, so a
    cut-time symbol is normalized to 2/2 and common time to 4/4.
    """
    if time is None:
        return current_num, current_den
    num, den = current_num, current_den
    try:
        if time.find("beats") is not None and time.find("beat-type") is not None:
            num = int(time.findtext("beats"))
            den = int(time.findtext("beat-type"))
    except Exception:
        pass
    symbol = (time.get("symbol") or "").strip().lower().replace("_", "-")
    if symbol in {"cut", "cut-time", "alla-breve", "allabreve"} or ("cut" in symbol and "time" in symbol):
        if (num, den) != (2, 2) and warnings is not None:
            warnings.append(f"Cut-time symbol detected; normalized exported {num}/{den} to 2/2 for tone-metric tactus.")
        return 2, 2
    if symbol in {"common", "common-time"}:
        if (num, den) != (4, 4) and warnings is not None:
            warnings.append(f"Common-time symbol detected; normalized exported {num}/{den} to 4/4.")
        return 4, 4
    return num, den


def _parse_meter_override(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        try:
            n, d = int(value[0]), int(value[1])
            return (n, d) if n > 0 and d > 0 else None
        except Exception:
            return None
    text = str(value).strip().lower()
    if not text or text == "auto":
        return None
    if "/" not in text:
        return None
    try:
        n, d = [int(x.strip()) for x in text.split("/", 1)]
    except Exception:
        return None
    return (n, d) if n > 0 and d > 0 else None

def _pitch_string(note) -> str:
    pitch = note.find("pitch")
    if pitch is None:
        unpitched = note.find("unpitched")
        return "unpitched" if unpitched is not None else "unknown"
    step = (pitch.findtext("step") or "?").strip()
    alter = int(pitch.findtext("alter") or "0")
    octave = (pitch.findtext("octave") or "?").strip()
    accidental = {2:"##",1:"#",0:"",-1:"b",-2:"bb"}.get(alter, f"({alter:+d})")
    return f"{step}{accidental}{octave}"


def _has_tie(note, tie_type: str) -> bool:
    return any((t.get("type") or "").lower() == tie_type for t in note.findall("tie")) or any(
        (t.get("type") or "").lower() == tie_type for t in note.findall("notations/tied")
    )


def _tuplet_ratio(note) -> tuple[int | None, int | None]:
    """Return MusicXML ``actual-notes : normal-notes`` when explicitly present.

    MusicXML ``duration`` already encodes performed score time.  The ratio is
    therefore metadata for the recursive tone-metric subdivision, not a duration
    correction.  In particular, 3:2 triplets and 2:3 duplets remain genuine
    sounding attacks; they are not filtered out.
    """
    tm = note.find("time-modification")
    if tm is None:
        return None, None
    try:
        actual = int((tm.findtext("actual-notes") or "").strip())
        normal = int((tm.findtext("normal-notes") or "").strip())
    except Exception:
        return None, None
    if actual < 2 or normal < 1:
        return None, None
    return actual, normal


def _layout_defaults(root) -> dict:
    defaults = root.find("defaults")
    if defaults is None:
        return {
            "page_width": 1190.0, "page_height": 1684.0,
            "left_margin": 80.0, "right_margin": 80.0,
            "top_margin": 80.0, "bottom_margin": 80.0,
            "system_left": 0.0, "top_system_distance": 100.0,
            "system_distance": 220.0, "staff_distance": 80.0,
        }
    pl = defaults.find("page-layout")
    page_width = _text_float(pl, "page-width", 1190.0) if pl is not None else 1190.0
    page_height = _text_float(pl, "page-height", 1684.0) if pl is not None else 1684.0
    left = right = top = bottom = 80.0
    if pl is not None:
        margins = pl.find("page-margins")
        if margins is not None:
            left = _text_float(margins, "left-margin", left)
            right = _text_float(margins, "right-margin", right)
            top = _text_float(margins, "top-margin", top)
            bottom = _text_float(margins, "bottom-margin", bottom)
    sl = defaults.find("system-layout")
    system_left = 0.0
    top_system_distance = 100.0
    system_distance = 220.0
    if sl is not None:
        sm = sl.find("system-margins")
        if sm is not None:
            system_left = _text_float(sm, "left-margin", 0.0)
        top_system_distance = _text_float(sl, "top-system-distance", top_system_distance)
        system_distance = _text_float(sl, "system-distance", system_distance)
    staff_distance = 80.0
    staff_layouts = defaults.findall("staff-layout")
    if staff_layouts:
        vals = [_text_float(x, "staff-distance", staff_distance) for x in staff_layouts]
        vals = [x for x in vals if x > 0]
        if vals:
            staff_distance = sum(vals) / len(vals)
    return {
        "page_width": page_width, "page_height": page_height,
        "left_margin": left, "right_margin": right,
        "top_margin": top, "bottom_margin": bottom,
        "system_left": system_left, "top_system_distance": top_system_distance,
        "system_distance": system_distance, "staff_distance": staff_distance,
    }


def _part_staff_offsets(root, default_staff_distance: float) -> dict:
    offsets = {}
    running = 0.0
    for part in root.findall("part"):
        pid = part.get("id") or "P?"
        max_staves = 1
        for m in part.findall("measure"):
            attrs = m.find("attributes")
            if attrs is not None and attrs.find("staves") is not None:
                try:
                    max_staves = max(max_staves, int(attrs.findtext("staves")))
                except Exception:
                    pass
        offsets[pid] = running
        running += max_staves * default_staff_distance + default_staff_distance * 0.5
    return offsets


def _measure_layouts(root) -> list[dict]:
    cfg = _layout_defaults(root)
    first_part = root.find("part")
    if first_part is None:
        return []
    page = 0
    system = 0
    current_system_left = cfg["system_left"]
    current_system_distance = cfg["system_distance"]
    current_top_system_distance = cfg["top_system_distance"]
    system_top = cfg["top_margin"] + current_top_system_distance
    x_cursor = cfg["left_margin"] + current_system_left
    layouts = []
    for mi, measure in enumerate(first_part.findall("measure")):
        pr = measure.find("print")
        if pr is not None:
            new_page = (pr.get("new-page") or "").lower() == "yes"
            new_system = (pr.get("new-system") or "").lower() == "yes"
            sys_layout = pr.find("system-layout")
            if sys_layout is not None:
                sm = sys_layout.find("system-margins")
                if sm is not None:
                    current_system_left = _text_float(sm, "left-margin", current_system_left)
                current_system_distance = _text_float(sys_layout, "system-distance", current_system_distance)
                current_top_system_distance = _text_float(sys_layout, "top-system-distance", current_top_system_distance)
            if new_page:
                page += 1
                system = 0
                system_top = cfg["top_margin"] + current_top_system_distance
                x_cursor = cfg["left_margin"] + current_system_left
            elif new_system:
                system += 1
                system_top += current_system_distance
                x_cursor = cfg["left_margin"] + current_system_left
        width = float(measure.get("width") or 100.0)
        layouts.append({
            "page_index": page,
            "system_index": system,
            "x_start": x_cursor,
            "width": width,
            "system_top": system_top,
            **cfg,
        })
        x_cursor += width
    return layouts


def parse_musicxml(path: str | Path, initial_meter_override=None) -> Tuple[List[Hit], List[MeasureInfo], List[str]]:
    """Parse MusicXML/MXL into globally merged rhythmic hits.

    Project rules:
    - ties: only initial attack; duration extended through tied continuation
    - grace notes: ignored
    - tuplets (time-modification): sounding attacks are retained with their explicit local ratio
    - rests: no attack
    - simultaneous attacks: merged globally into one hit
    - pickup: first short/implicit measure is right-aligned inside a full measure
    - visual coordinates: when MusicXML layout positions are available, preserve page/x/y
      so the analysis can be overlaid on the original uploaded PDF.
    """
    path = Path(path)
    root = etree.fromstring(_read_musicxml_bytes(path))
    if root.tag.split("}")[-1] != "score-partwise":
        raise ValueError("Only score-partwise MusicXML is currently supported")

    parts = root.findall("part")
    if not parts:
        raise ValueError("No <part> elements found in MusicXML")

    warnings: List[str] = []
    measure_layouts = _measure_layouts(root)
    cfg = _layout_defaults(root)
    part_offsets = _part_staff_offsets(root, cfg["staff_distance"])

    measure_templates: List[dict] = []
    divisions = 1
    meter_override = _parse_meter_override(initial_meter_override)
    num, den = meter_override or (4, 4)
    cumulative = Fraction(0)
    first_part = parts[0]
    for mi, measure in enumerate(first_part.findall("measure")):
        attrs = measure.find("attributes")
        if attrs is not None:
            if attrs.find("divisions") is not None:
                divisions = int(attrs.findtext("divisions"))
            time = attrs.find("time")
            if time is not None:
                if mi == 0 and meter_override is not None:
                    xml_num, xml_den = _normalized_time_signature(time, num, den, warnings)
                    if (xml_num, xml_den) != meter_override:
                        warnings.append(f"Initial meter override {meter_override[0]}/{meter_override[1]} used instead of exported {xml_num}/{xml_den}.")
                    num, den = meter_override
                else:
                    num, den = _normalized_time_signature(time, num, den, warnings)
        full = Fraction(num * 4, den)
        cursor = Fraction(0)
        max_cursor = Fraction(0)
        last_nonchord_onset = Fraction(0)
        for child in measure:
            tag = child.tag.split("}")[-1]
            if tag == "note":
                dur_div = _text_int(child, "duration", 0)
                dur = Fraction(dur_div, divisions) if divisions else Fraction(0)
                if child.find("chord") is not None:
                    onset = last_nonchord_onset
                else:
                    onset = cursor
                    last_nonchord_onset = onset
                    if child.find("grace") is None:
                        cursor += dur
                        max_cursor = max(max_cursor, cursor)
            elif tag == "backup":
                cursor -= Fraction(_text_int(child, "duration", 0), divisions)
            elif tag == "forward":
                cursor += Fraction(_text_int(child, "duration", 0), divisions)
                max_cursor = max(max_cursor, cursor)
        implicit = (measure.get("implicit") or "").lower() == "yes"
        short_first = mi == 0 and max_cursor < full
        pickup_shift = full - max_cursor if (implicit or short_first) and max_cursor < full else Fraction(0)
        measure_templates.append({
            "index": mi,
            "number": measure.get("number") or str(mi + 1),
            "start": cumulative,
            "full": full,
            "actual": max_cursor,
            "shift": pickup_shift,
            "num": num,
            "den": den,
            "implicit": implicit,
        })
        cumulative += full

    measures = [MeasureInfo(
        index=m["index"], number=m["number"], start=m["start"], full_duration=m["full"],
        actual_duration=m["actual"], pickup_shift=m["shift"], numerator=m["num"], denominator=m["den"], implicit=m["implicit"]
    ) for m in measure_templates]

    raw_attacks: List[NoteAttack] = []
    open_ties: Dict[tuple, NoteAttack] = {}
    layout_note_count = 0

    for part in parts:
        part_id = part.get("id") or "P?"
        divisions = 1
        p_measures = part.findall("measure")
        if len(p_measures) != len(measure_templates):
            warnings.append(f"Part {part_id} has {len(p_measures)} measures; canonical part has {len(measure_templates)}. Alignment uses measure index.")
        for mi, measure in enumerate(p_measures):
            if mi >= len(measure_templates):
                break
            tmpl = measure_templates[mi]
            layout = measure_layouts[mi] if mi < len(measure_layouts) else None
            attrs = measure.find("attributes")
            if attrs is not None and attrs.find("divisions") is not None:
                divisions = int(attrs.findtext("divisions"))
            cursor = Fraction(0)
            last_nonchord_onset = Fraction(0)
            for child in measure:
                tag = child.tag.split("}")[-1]
                if tag == "note":
                    dur_div = _text_int(child, "duration", 0)
                    dur = Fraction(dur_div, divisions) if divisions else Fraction(0)
                    is_grace = child.find("grace") is not None
                    tuplet_actual, tuplet_normal = _tuplet_ratio(child)
                    is_chord = child.find("chord") is not None
                    if is_chord:
                        onset_local = last_nonchord_onset
                    else:
                        onset_local = cursor
                        last_nonchord_onset = onset_local
                    if not is_chord and not is_grace:
                        cursor += dur

                    if child.find("rest") is not None or is_grace:
                        continue

                    voice = (child.findtext("voice") or "1").strip()
                    staff = (child.findtext("staff") or "1").strip()
                    pitch = _pitch_string(child)
                    tie_start = _has_tie(child, "start")
                    tie_stop = _has_tie(child, "stop")
                    key = (part_id, staff, voice, pitch)
                    global_onset = tmpl["start"] + tmpl["shift"] + onset_local

                    page_index = None
                    x_norm = None
                    y_norm = None
                    if layout is not None:
                        try:
                            dx = float(child.get("default-x")) if child.get("default-x") is not None else None
                            dy = float(child.get("default-y")) if child.get("default-y") is not None else None
                            if dx is not None:
                                x_abs = layout["x_start"] + dx
                                x_norm = max(0.0, min(1.0, x_abs / max(layout["page_width"], 1.0)))
                            if dy is not None:
                                staff_no = max(1, int(staff))
                                staff_top = layout["system_top"] + part_offsets.get(part_id, 0.0) + (staff_no - 1) * layout["staff_distance"]
                                y_abs = staff_top - dy
                                y_norm = max(0.0, min(1.0, y_abs / max(layout["page_height"], 1.0)))
                            page_index = layout["page_index"]
                            if x_norm is not None and y_norm is not None:
                                layout_note_count += 1
                        except Exception:
                            pass

                    if tie_stop:
                        prev = open_ties.get(key)
                        if prev is not None:
                            prev.duration += dur
                        else:
                            warnings.append(f"Unmatched tie-stop at measure {tmpl['number']} for {pitch}; treated as continuation with no new attack.")
                        if not tie_start:
                            open_ties.pop(key, None)
                        continue

                    attack = NoteAttack(
                        onset=global_onset,
                        duration=dur,
                        measure_index=mi,
                        measure_number=tmpl["number"],
                        offset_in_measure=tmpl["shift"] + onset_local,
                        part_id=part_id,
                        voice=voice,
                        staff=staff,
                        pitch=pitch,
                        tie_start=tie_start,
                        tie_stop=tie_stop,
                        tuplet_actual=tuplet_actual,
                        tuplet_normal=tuplet_normal,
                        page_index=page_index,
                        x_norm=x_norm,
                        y_norm=y_norm,
                    )
                    raw_attacks.append(attack)
                    if tie_start:
                        open_ties[key] = attack
                elif tag == "backup":
                    cursor -= Fraction(_text_int(child, "duration", 0), divisions)
                elif tag == "forward":
                    cursor += Fraction(_text_int(child, "duration", 0), divisions)

    by_onset: Dict[Fraction, List[NoteAttack]] = {}
    for a in raw_attacks:
        by_onset.setdefault(a.onset, []).append(a)

    hits: List[Hit] = []
    for onset in sorted(by_onset):
        src = by_onset[onset]
        first = src[0]
        hits.append(Hit(
            onset=onset,
            duration=max((s.duration for s in src), default=Fraction(0)),
            measure_index=first.measure_index,
            measure_number=first.measure_number,
            offset_in_measure=first.offset_in_measure,
            sources=src,
        ))

    if layout_note_count == 0:
        warnings.append("The MusicXML export did not provide usable note layout coordinates, so coloring on the original PDF may be unavailable or approximate for this score.")
    return hits, measures, warnings


def extract_visual_groups(path: str | Path, initial_meter_override=None) -> tuple[list[dict], bool]:
    """Return symbolic notehead/chord columns in visual reading order.

    Unlike analysis hits, this includes grace notes and tied continuations because those
    noteheads are physically present on the score. Tuplet attacks are analysis attacks too;
    their ratio changes the recursive subdivision, not whether the note sounds. Each group
    is marked ``analyzed`` only when at least one contained note creates a Tone-Metric attack.
    This lets Audiveris physical notehead columns be matched without shifting the overlay.
    """
    path = Path(path)
    root = etree.fromstring(_read_musicxml_bytes(path))
    if root.tag.split("}")[-1] != "score-partwise":
        return [], False
    parts = root.findall("part")
    if not parts:
        return [], False

    measure_layouts = _measure_layouts(root)
    layout_known = any(
        m.find("print") is not None or m.get("width") is not None
        for m in parts[0].findall("measure")
    )

    # Canonical measure timing/pickup shifts from first part.
    templates = []
    divisions = 1
    meter_override = _parse_meter_override(initial_meter_override)
    num, den = meter_override or (4, 4)
    cumulative = Fraction(0)
    for mi, measure in enumerate(parts[0].findall("measure")):
        attrs = measure.find("attributes")
        if attrs is not None:
            if attrs.find("divisions") is not None:
                divisions = int(attrs.findtext("divisions"))
            time = attrs.find("time")
            if time is not None:
                if mi == 0 and meter_override is not None:
                    num, den = meter_override
                else:
                    num, den = _normalized_time_signature(time, num, den)
        full = Fraction(num * 4, den)
        cursor = Fraction(0); max_cursor = Fraction(0); last_nonchord = Fraction(0)
        for child in measure:
            tag = child.tag.split("}")[-1]
            if tag == "note":
                dur = Fraction(_text_int(child, "duration", 0), divisions) if divisions else Fraction(0)
                if child.find("chord") is not None:
                    onset = last_nonchord
                else:
                    onset = cursor; last_nonchord = onset
                    if child.find("grace") is None:
                        cursor += dur; max_cursor = max(max_cursor, cursor)
            elif tag == "backup":
                cursor -= Fraction(_text_int(child, "duration", 0), divisions)
            elif tag == "forward":
                cursor += Fraction(_text_int(child, "duration", 0), divisions); max_cursor = max(max_cursor, cursor)
        implicit = (measure.get("implicit") or "").lower() == "yes"
        short_first = mi == 0 and max_cursor < full
        shift = full - max_cursor if (implicit or short_first) and max_cursor < full else Fraction(0)
        templates.append({
            "start": cumulative,
            "shift": shift,
            "full": full,
            "number": measure.get("number") or str(mi + 1),
        })
        cumulative += full

    groups: dict[tuple, dict] = {}
    serial = 0
    for part in parts:
        divisions = 1
        for mi, measure in enumerate(part.findall("measure")):
            if mi >= len(templates):
                break
            attrs = measure.find("attributes")
            if attrs is not None and attrs.find("divisions") is not None:
                divisions = int(attrs.findtext("divisions"))
            cursor = Fraction(0); last_nonchord = Fraction(0)
            for child in measure:
                tag = child.tag.split("}")[-1]
                if tag == "note":
                    dur = Fraction(_text_int(child, "duration", 0), divisions) if divisions else Fraction(0)
                    is_chord = child.find("chord") is not None
                    is_grace = child.find("grace") is not None
                    if is_chord:
                        onset_local = last_nonchord
                    else:
                        onset_local = cursor; last_nonchord = onset_local
                    if not is_chord and not is_grace:
                        cursor += dur
                    if child.find("rest") is not None:
                        continue

                    # Prefer exported horizontal engraving position when present.
                    # Audiveris MusicXML default-x is local to the measure; combining it
                    # with the exported measure x_start gives us an engraving-aware
                    # horizontal coordinate that can later be registered to the PDF.
                    dx = child.get("default-x")
                    try:
                        xkey = round(float(dx), 2) if dx is not None else None
                    except Exception:
                        xkey = None
                    # Grace note at same rhythmic onset must remain a separate visual column.
                    visual_kind = "grace" if is_grace else "main"
                    global_onset = templates[mi]["start"] + templates[mi]["shift"] + onset_local
                    layout = measure_layouts[mi] if mi < len(measure_layouts) else {}
                    layout_x_abs = None
                    if xkey is not None and layout.get("x_start") is not None:
                        try:
                            layout_x_abs = float(layout["x_start"]) + float(xkey)
                        except Exception:
                            layout_x_abs = None
                    key = (mi, layout.get("page_index", 0), layout.get("system_index", 0), xkey if xkey is not None else global_onset, visual_kind)
                    row = groups.setdefault(key, {
                        "serial": serial,
                        "measure_index": mi,
                        "measure_number": templates[mi]["number"],
                        "layout_page": layout.get("page_index", 0),
                        "layout_system": layout.get("system_index", 0),
                        "layout_x_abs": layout_x_abs,
                        "layout_default_x": xkey,
                        "layout_measure_x_start": layout.get("x_start"),
                        "layout_measure_width": layout.get("width"),
                        "layout_page_width": layout.get("page_width"),
                        "layout_page_height": layout.get("page_height"),
                        "measure_start_quarter": frac_to_str(templates[mi]["start"]),
                        "measure_full_quarter": frac_to_str(templates[mi]["full"]),
                        "measure_shift_quarter": frac_to_str(templates[mi]["shift"]),
                        "offset_in_measure_quarter": frac_to_str(templates[mi]["shift"] + onset_local),
                        "duration_quarter": frac_to_str(dur),
                        "attack_key": f"{mi}:{frac_to_str(templates[mi]['shift'] + onset_local)}",
                        "visual_kind": visual_kind,
                        "onset_quarter": frac_to_str(global_onset),
                        "analyzed": False,
                        "pitches": [],
                    })
                    serial += 1
                    row["pitches"].append(_pitch_string(child))
                    tuplet_actual, tuplet_normal = _tuplet_ratio(child)
                    if tuplet_actual is not None:
                        row["tuplet_actual"] = tuplet_actual
                        row["tuplet_normal"] = tuplet_normal
                    tie_stop = _has_tie(child, "stop")
                    if not is_grace and not tie_stop:
                        row["analyzed"] = True
                elif tag == "backup":
                    cursor -= Fraction(_text_int(child, "duration", 0), divisions)
                elif tag == "forward":
                    cursor += Fraction(_text_int(child, "duration", 0), divisions)

    rows = list(groups.values())
    rows.sort(key=lambda r: (r["layout_page"], r["layout_system"], r["measure_index"], r["serial"]))
    return rows, layout_known
