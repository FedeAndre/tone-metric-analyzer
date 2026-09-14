from __future__ import annotations

import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

APP_VERSION = "0.8.0-recursive-levels-standalone-clean"
MAX_UPLOAD = 80 * 1024 * 1024
PDF_ZOOM = 2.0

# CLEAN-RUNTIME CONTRACT
# ----------------------
# This file is self-contained. It deliberately imports NONE of:
#   level1_clean_v4 / v5 / v6 / recursive_levels_clean_v7
#   tone_metric/*
#   dissertation_*
#   legacy renderer/overlay modules
# Level 1, Level 2, recursive higher Levels, MusicXML parsing, Audiveris OMR
# registration, PDF rendering, HTTP routes, and UI are all defined here.
# Attack labels are NEVER proportionally interpolated: an attack must resolve
# to the exact Audiveris OMR time slot or rendering stops with an explicit error.
app = FastAPI(title="Tone-Metric Recursive Levels — Clean", version=APP_VERSION)


@dataclass(frozen=True)
class MeterProfile:
    numerator: int
    denominator: int
    beat_unit: Fraction
    arity: int


PROFILES: dict[tuple[int, int], MeterProfile] = {
    (2, 2): MeterProfile(2, 2, Fraction(2), 2),
    (4, 4): MeterProfile(4, 4, Fraction(1), 2),
    (3, 4): MeterProfile(3, 4, Fraction(1), 3),
    (6, 8): MeterProfile(6, 8, Fraction(3, 2), 2),
    (9, 8): MeterProfile(9, 8, Fraction(3, 2), 3),
    (12, 8): MeterProfile(12, 8, Fraction(3, 2), 2),
}


@dataclass(frozen=True)
class Measure:
    index: int
    number: str
    start: Fraction
    end: Fraction
    pickup_shift: Fraction
    meter: tuple[int, int]


@dataclass(frozen=True)
class Attack:
    onset: Fraction


def lname(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def child(el, name: str):
    for c in el:
        if lname(c.tag) == name:
            return c
    return None


def children(el, name: str):
    return [c for c in el if lname(c.tag) == name]


def text(el, name: str, default: str | None = None) -> str | None:
    c = child(el, name)
    return default if c is None or c.text is None else c.text.strip()


def ftxt(v: Fraction) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def parse_fraction(value: str) -> Fraction:
    return Fraction(str(value))


def parse_meter(value: str | None) -> tuple[int, int]:
    raw = (value or "").strip().lower()
    if raw in {"", "auto"}:
        return (4, 4)
    if "/" not in raw:
        raise ValueError("Meter must be numerator/denominator.")
    a, b = raw.split("/", 1)
    meter = (int(a), int(b))
    if meter not in PROFILES:
        raise ValueError(f"Unsupported meter {meter[0]}/{meter[1]}.")
    return meter


def sequence(arity: int, limit: int) -> list[int]:
    if arity not in (2, 3):
        raise ValueError("Only binary and ternary dissertation sequences are supported.")
    if limit < 1:
        return []
    out = [1]
    if limit == 1:
        return out
    out.append(2)
    while out[-1] < limit:
        out.append(arity * out[-1] - (arity - 1))
    return out


def xml_root(path: Path):
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            rootfile = None
            if "META-INF/container.xml" in names:
                root = etree.fromstring(zf.read("META-INF/container.xml"))
                for n in root.iter():
                    if lname(n.tag) == "rootfile":
                        rootfile = n.get("full-path")
                        if rootfile:
                            break
            if not rootfile:
                candidates = [
                    n for n in names
                    if n.lower().endswith((".xml", ".musicxml"))
                    and not n.startswith("META-INF/")
                ]
                if not candidates:
                    raise ValueError("MXL contains no MusicXML score.")
                rootfile = sorted(candidates)[0]
            return etree.fromstring(zf.read(rootfile))
    return etree.parse(str(path)).getroot()


def first_part(root):
    parts = [n for n in root.iter() if lname(n.tag) == "part" and n.getparent() is root]
    if not parts:
        parts = [n for n in root.iter() if lname(n.tag) == "part"]
    if not parts:
        raise ValueError("No MusicXML part found.")
    return parts[0], parts


def time_signature(attrs):
    if attrs is None:
        return None
    t = child(attrs, "time")
    if t is None:
        return None
    beats, beat_type = text(t, "beats"), text(t, "beat-type")
    if not beats or not beat_type or "+" in beats:
        return None
    return int(beats), int(beat_type)


def scan_measure(measure_el, divisions: int):
    cursor = Fraction(0)
    max_cursor = Fraction(0)
    previous_note_onset = Fraction(0)
    note_rows: list[tuple[Fraction, bool]] = []

    for item in measure_el:
        tag = lname(item.tag)
        if tag == "note":
            if child(item, "grace") is not None:
                continue
            chord = child(item, "chord") is not None
            dur = Fraction(int(text(item, "duration", "0") or "0"), max(1, divisions))
            onset = previous_note_onset if chord else cursor
            previous_note_onset = onset
            if not chord:
                cursor += dur
            max_cursor = max(max_cursor, onset + dur, cursor)

            rest = child(item, "rest") is not None
            tied_stop = any((n.get("type") or "").lower() == "stop" for n in children(item, "tie"))
            notations = child(item, "notations")
            if notations is not None:
                for n in notations.iter():
                    if lname(n.tag) == "tied" and (n.get("type") or "").lower() == "stop":
                        tied_stop = True
            note_rows.append((onset, (not rest) and (not tied_stop)))
        elif tag == "backup":
            cursor -= Fraction(int(text(item, "duration", "0") or "0"), max(1, divisions))
            cursor = max(cursor, Fraction(0))
        elif tag == "forward":
            cursor += Fraction(int(text(item, "duration", "0") or "0"), max(1, divisions))
            max_cursor = max(max_cursor, cursor)
    return max_cursor, note_rows


def parse_score(path: Path, opening_meter: tuple[int, int]):
    root = xml_root(path)
    first, parts = first_part(root)

    measures: list[Measure] = []
    divisions = 1
    inherited = opening_meter
    global_start = Fraction(0)

    for mi, m in enumerate(children(first, "measure")):
        attrs = child(m, "attributes")
        if attrs is not None:
            d = text(attrs, "divisions")
            if d:
                divisions = max(1, int(d))
            explicit = time_signature(attrs)
            if explicit in PROFILES:
                inherited = explicit
        profile = PROFILES[inherited]
        full = Fraction(profile.numerator * 4, profile.denominator)
        actual, _ = scan_measure(m, divisions)
        if actual <= 0:
            actual = full
        pickup = full - actual if mi == 0 and actual < full else Fraction(0)
        measures.append(
            Measure(
                mi,
                m.get("number") or str(mi + 1),
                global_start,
                global_start + full,
                pickup,
                inherited,
            )
        )
        global_start += full

    attacks: set[Fraction] = set()
    for part in parts:
        divisions = 1
        inherited = opening_meter
        for mi, m in enumerate(children(part, "measure")[: len(measures)]):
            attrs = child(m, "attributes")
            if attrs is not None:
                d = text(attrs, "divisions")
                if d:
                    divisions = max(1, int(d))
                explicit = time_signature(attrs)
                if explicit in PROFILES:
                    inherited = explicit
            _, rows = scan_measure(m, divisions)
            mm = measures[mi]
            for local_onset, is_attack in rows:
                if is_attack:
                    t = mm.start + mm.pickup_shift + local_onset
                    if mm.start <= t < mm.end:
                        attacks.add(t)
    return measures, [Attack(t) for t in sorted(attacks)]


def meter_segments(measures: list[Measure]):
    if not measures:
        return []
    out = []
    start = 0
    cur = measures[0].meter
    for i in range(1, len(measures) + 1):
        changed = i == len(measures) or measures[i].meter != cur
        if changed:
            out.append((start, i - 1))
            if i < len(measures):
                start, cur = i, measures[i].meter
    return out


def _measure_for_time(measures: list[Measure], a: int, b: int, t: Fraction):
    return next(m for m in measures[a : b + 1] if m.start <= t < m.end)


def analyze_level1(measures: list[Measure], attacks: list[Attack]) -> dict:
    attack_times = {a.onset for a in attacks}
    points: list[dict] = []
    segments: list[dict] = []
    for si, (a, b) in enumerate(meter_segments(measures)):
        first, last = measures[a], measures[b]
        profile = PROFILES[first.meter]
        start, end = first.start, last.end
        count = int((end - start) / profile.beat_unit)
        seq = sequence(profile.arity, count + 1)
        for pos in seq:
            t = start + (pos - 1) * profile.beat_unit
            if not (start <= t < end):
                continue
            m = _measure_for_time(measures, a, b, t)
            is_attack = t in attack_times
            points.append({
                "segment_index": si,
                "sequence_position": pos,
                "time_quarter": ftxt(t),
                "measure_index": m.index,
                "measure_number": m.number,
                "beat": int((t - m.start) / profile.beat_unit) + 1,
                "level": 1,
                "attack": is_attack,
                "parenthetical": not is_attack,
                "label": "1" if is_attack else "(1)",
            })
        segments.append({
            "segment_index": si,
            "start_measure": first.number,
            "end_measure": last.number,
            "meter": f"{profile.numerator}/{profile.denominator}",
            "arity": profile.arity,
            "sequence": seq,
        })
    return {
        "engine_contract": "dissertation-recursive-levels-standalone-v0.8",
        "points": points,
        "segments": segments,
        "measures": [
            {
                "index": m.index,
                "number": m.number,
                "start_quarter": ftxt(m.start),
                "end_quarter": ftxt(m.end),
                "meter": f"{m.meter[0]}/{m.meter[1]}",
            }
            for m in measures
        ],
        "measure_count": len(measures),
        "attack_count": len(attack_times),
    }


def analyze_level2(measures: list[Measure], attacks: list[Attack], level1_result: dict) -> list[dict]:
    attack_times = {a.onset for a in attacks}
    l1_by_segment: dict[int, list[Fraction]] = {}
    for p in level1_result.get("points", []):
        l1_by_segment.setdefault(int(p["segment_index"]), []).append(parse_fraction(p["time_quarter"]))

    level2_by_time: dict[tuple[int, Fraction], dict] = {}

    def add_point(si: int, a: int, b: int, profile, t: Fraction, seq_pos: int,
                  span_start: Fraction, span_end: Fraction | None):
        if not (measures[a].start <= t < measures[b].end):
            return
        key = (si, t)
        if key in level2_by_time:
            return
        m = _measure_for_time(measures, a, b, t)
        is_attack = t in attack_times
        level2_by_time[key] = {
            "segment_index": si,
            "sequence_position": seq_pos,
            "time_quarter": ftxt(t),
            "measure_index": m.index,
            "measure_number": m.number,
            "beat": int((t - m.start) / profile.beat_unit) + 1,
            "level": 2,
            "attack": is_attack,
            "parenthetical": not is_attack,
            "label": "2" if is_attack else "(2)",
            "recursive_span_start_quarter": ftxt(span_start),
            "recursive_span_end_quarter": None if span_end is None else ftxt(span_end),
        }

    for si, (a, b) in enumerate(meter_segments(measures)):
        first, last = measures[a], measures[b]
        profile = PROFILES[first.meter]
        beat = profile.beat_unit
        seg_end = last.end
        boundaries = sorted(set(l1_by_segment.get(si, [])))
        if not boundaries:
            continue

        for t0, t1 in zip(boundaries, boundaries[1:]):
            steps = int((t1 - t0) / beat)
            if steps <= 1:
                continue
            position_count = steps + 1
            for pos in sequence(profile.arity, position_count):
                if pos > position_count:
                    break
                t = t0 + (pos - 1) * beat
                if t > t1:
                    break
                add_point(si, a, b, profile, t, pos, t0, t1)

        t0 = boundaries[-1]
        position_count = int((seg_end - t0) / beat)
        if position_count > 1:
            for pos in sequence(profile.arity, position_count):
                if pos > position_count:
                    break
                t = t0 + (pos - 1) * beat
                if t0 <= t < seg_end:
                    add_point(si, a, b, profile, t, pos, t0, None)

    return [level2_by_time[k] for k in sorted(level2_by_time, key=lambda x: (x[0], x[1]))]


def _group_times_by_segment(points: list[dict]) -> dict[int, list[Fraction]]:
    out: dict[int, list[Fraction]] = {}
    for p in points:
        out.setdefault(int(p["segment_index"]), []).append(parse_fraction(p["time_quarter"]))
    return {si: sorted(set(times)) for si, times in out.items()}


def _all_tactus_times(measures: list[Measure]) -> dict[int, set[Fraction]]:
    out: dict[int, set[Fraction]] = {}
    for si, (a, b) in enumerate(meter_segments(measures)):
        profile = PROFILES[measures[a].meter]
        beat = profile.beat_unit
        start = measures[a].start
        end = measures[b].end
        count = int((end - start) / beat)
        out[si] = {start + k * beat for k in range(count)}
    return out


def analyze_next_recursive_level(
    measures: list[Measure], attacks: list[Attack], previous_points: list[dict],
    covered_before: dict[int, set[Fraction]], level_number: int,
) -> list[dict]:
    attack_times = {a.onset for a in attacks}
    previous_by_segment = _group_times_by_segment(previous_points)
    next_by_time: dict[tuple[int, Fraction], dict] = {}

    def add_point(si: int, a: int, b: int, profile, t: Fraction, seq_pos: int,
                  span_start: Fraction, span_end: Fraction | None):
        if not (measures[a].start <= t < measures[b].end):
            return
        key = (si, t)
        if key in next_by_time:
            return
        m = _measure_for_time(measures, a, b, t)
        is_attack = t in attack_times
        next_by_time[key] = {
            "segment_index": si,
            "sequence_position": seq_pos,
            "time_quarter": ftxt(t),
            "measure_index": m.index,
            "measure_number": m.number,
            "beat": int((t - m.start) / profile.beat_unit) + 1,
            "level": level_number,
            "attack": is_attack,
            "parenthetical": not is_attack,
            "label": str(level_number) if is_attack else f"({level_number})",
            "recursive_span_start_quarter": ftxt(span_start),
            "recursive_span_end_quarter": None if span_end is None else ftxt(span_end),
        }

    for si, (a, b) in enumerate(meter_segments(measures)):
        profile = PROFILES[measures[a].meter]
        beat = profile.beat_unit
        seg_end = measures[b].end
        boundaries = previous_by_segment.get(si, [])
        if not boundaries:
            continue
        covered = covered_before.get(si, set())

        for t0, t1 in zip(boundaries, boundaries[1:]):
            steps = int((t1 - t0) / beat)
            if steps <= 1:
                continue
            interior = [t0 + k * beat for k in range(1, steps)]
            if not any(t not in covered for t in interior):
                continue
            position_count = steps + 1
            for pos in sequence(profile.arity, position_count):
                if pos > position_count:
                    break
                t = t0 + (pos - 1) * beat
                if t > t1:
                    break
                add_point(si, a, b, profile, t, pos, t0, t1)

        t0 = boundaries[-1]
        position_count = int((seg_end - t0) / beat)
        if position_count > 1:
            interior = [t0 + k * beat for k in range(1, position_count)]
            if any(t not in covered for t in interior):
                for pos in sequence(profile.arity, position_count):
                    if pos > position_count:
                        break
                    t = t0 + (pos - 1) * beat
                    if t0 <= t < seg_end:
                        add_point(si, a, b, profile, t, pos, t0, None)

    return [next_by_time[k] for k in sorted(next_by_time, key=lambda x: (x[0], x[1]))]


def analyze_recursive_levels(measures: list[Measure], attacks: list[Attack]) -> dict:
    result = analyze_level1(measures, attacks)
    l1 = result["points"]
    l2 = analyze_level2(measures, attacks, result)
    levels: dict[str, list[dict]] = {"1": l1, "2": l2}

    all_tactus = _all_tactus_times(measures)
    covered: dict[int, set[Fraction]] = {si: set() for si in all_tactus}
    for pts in (l1, l2):
        for p in pts:
            covered.setdefault(int(p["segment_index"]), set()).add(parse_fraction(p["time_quarter"]))

    def complete() -> bool:
        return all(all_tactus.get(si, set()) <= covered.get(si, set()) for si in all_tactus)

    previous = l2
    level_number = 3
    while not complete():
        if not previous:
            missing = sum(len(all_tactus[si] - covered.get(si, set())) for si in all_tactus)
            raise RuntimeError(f"Recursive construction stalled with {missing} unmapped tactus positions.")
        next_points = analyze_next_recursive_level(measures, attacks, previous, covered, level_number)
        if not next_points:
            missing = sum(len(all_tactus[si] - covered.get(si, set())) for si in all_tactus)
            raise RuntimeError(
                f"Recursive construction made no progress at Level {level_number}; "
                f"{missing} tactus positions remain unmapped."
            )
        before = sum(len(v) for v in covered.values())
        for p in next_points:
            covered.setdefault(int(p["segment_index"]), set()).add(parse_fraction(p["time_quarter"]))
        after = sum(len(v) for v in covered.values())
        if after <= before:
            raise RuntimeError(f"Recursive Level {level_number} added no newly mapped tactus position.")
        levels[str(level_number)] = next_points
        previous = next_points
        level_number += 1

    highest = max(int(k) for k, pts in levels.items() if pts) if any(levels.values()) else 1
    result["level1_points"] = l1
    result["level2_points"] = l2
    result["levels"] = levels
    result["highest_level"] = highest
    result["recursive_levels_complete"] = True
    result["recursive_scope"] = "tactus-denomination-only"
    for k, pts in levels.items():
        result[f"level{k}_points"] = pts
    return result


def find_audiveris() -> str:
    configured = os.environ.get("AUDIVERIS_CMD", "").strip()
    if configured and Path(configured).exists():
        return configured
    for c in ("Audiveris", "audiveris", "/opt/audiveris/bin/Audiveris"):
        found = shutil.which(c) if not c.startswith("/") else c
        if found and Path(found).exists():
            return str(found)
    raise RuntimeError("Audiveris is not available.")


def pdf_to_musicxml(pdf: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_audiveris(), "-batch", "-transcribe", "-save", "-export",
        "-output", str(out), "--", str(pdf),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    if p.returncode != 0:
        raise RuntimeError(f"Audiveris failed with exit code {p.returncode}.\n\n{(p.stdout or '')[-12000:]}")
    candidates: list[Path] = []
    for pattern in ("*.mxl", "*.musicxml", "*.xml"):
        candidates.extend(out.rglob(pattern))
    candidates = [x for x in candidates if "container.xml" not in str(x)]
    if not candidates:
        raise RuntimeError("Audiveris completed but no MusicXML export was found.")
    candidates.sort(key=lambda x: (0 if x.suffix.lower() == ".mxl" else 1, len(str(x))))
    return candidates[0]


def _font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _sheet_number(path: str) -> int:
    m = re.search(r"sheet#(\d+)/sheet#\1\.xml$", path)
    return int(m.group(1)) if m else 10**9


def find_omr(out_dir: Path, symbolic_path: Path | None = None) -> Path:
    candidates = sorted(out_dir.rglob("*.omr"))
    if not candidates:
        raise RuntimeError("Audiveris completed but no OMR save file was found for PDF registration.")
    if symbolic_path is not None:
        stem = symbolic_path.stem.lower()
        matching = [p for p in candidates if p.stem.lower() == stem]
        if matching:
            candidates = matching
    candidates.sort(key=lambda p: (-p.stat().st_size, len(str(p))))
    return candidates[0]


def parse_omr_geometry(omr_path: Path, measure_count: int) -> list[dict]:
    pages: list[dict] = []
    measure_index = 0
    with zipfile.ZipFile(omr_path) as zf:
        sheet_xmls = sorted(
            [n for n in zf.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n)],
            key=_sheet_number,
        )
        if not sheet_xmls:
            raise RuntimeError("OMR save file contains no sheet XML geometry.")

        for page_index, name in enumerate(sheet_xmls):
            root = etree.fromstring(zf.read(name))
            picture = next((n for n in root if lname(n.tag) == "picture"), None)
            page = next((n for n in root if lname(n.tag) == "page"), None)
            if picture is None or page is None:
                raise RuntimeError(f"OMR sheet {page_index + 1} is missing picture/page geometry.")
            picture_width = float(picture.get("width") or 0)
            picture_height = float(picture.get("height") or 0)
            if picture_width <= 0 or picture_height <= 0:
                raise RuntimeError(f"OMR sheet {page_index + 1} has invalid source dimensions.")

            systems: list[dict] = []
            for system_index, system in enumerate([n for n in page if lname(n.tag) == "system"]):
                staff_x: list[float] = []
                staff_y: list[float] = []
                for node in system.iter():
                    if lname(node.tag) == "staff":
                        try:
                            staff_x.extend([float(node.get("left")), float(node.get("right"))])
                        except Exception:
                            pass
                        for point in node.iter():
                            if lname(point.tag) == "point":
                                try:
                                    staff_y.append(float(point.get("y")))
                                except Exception:
                                    pass

                stacks: list[dict] = []
                for stack in [n for n in system if lname(n.tag) == "stack"]:
                    if measure_index >= measure_count:
                        break
                    left = float(stack.get("left") or 0.0)
                    right = float(stack.get("right") or left)
                    slots: dict[Fraction, float] = {}
                    for slot in [n for n in stack if lname(n.tag) == "slot"]:
                        raw_time = slot.get("time-offset")
                        raw_x = slot.get("x-offset")
                        if raw_time is None or raw_x is None:
                            continue
                        try:
                            slots[Fraction(raw_time)] = left + float(raw_x)
                        except Exception:
                            continue
                    stacks.append({
                        "measure_index": measure_index,
                        "left": left,
                        "right": right,
                        "slots": slots,
                    })
                    measure_index += 1

                if not stacks:
                    continue
                if not staff_x:
                    staff_x = [min(s["left"] for s in stacks), max(s["right"] for s in stacks)]
                if not staff_y:
                    raise RuntimeError(
                        f"OMR sheet {page_index + 1}, system {system_index + 1} has no staff-line coordinates."
                    )
                systems.append({
                    "system_index": system_index,
                    "staff_left": min(staff_x),
                    "staff_right": max(staff_x),
                    "staff_top": min(staff_y),
                    "staff_bottom": max(staff_y),
                    "stacks": stacks,
                })

            pages.append({
                "page_index": page_index,
                "source_width": picture_width,
                "source_height": picture_height,
                "systems": systems,
            })

    if measure_index != measure_count:
        raise RuntimeError(
            f"OMR/PDF registration has {measure_index} measure stacks but analysis has {measure_count} measures."
        )
    return pages


def exact_slot_x(stack: dict, local_whole: Fraction) -> float | None:
    slots: dict[Fraction, float] = stack.get("slots", {})
    if local_whole in slots:
        return float(slots[local_whole])
    return None


def structural_x(stack: dict, local_whole: Fraction, duration_whole: Fraction) -> float:
    """Interpolation is allowed ONLY for structural non-attack positions."""
    left = float(stack["left"])
    right = float(stack["right"])
    if duration_whole <= 0:
        return left
    t = max(Fraction(0), min(duration_whole, local_whole))
    anchors: list[tuple[Fraction, float]] = [(Fraction(0), left)]
    anchors.extend(sorted((Fraction(k), float(v)) for k, v in stack.get("slots", {}).items()))
    anchors.append((duration_whole, right))
    merged: dict[Fraction, float] = {}
    for tt, xx in anchors:
        merged[tt] = xx
    ordered = sorted(merged.items())
    if t <= ordered[0][0]:
        return ordered[0][1]
    if t >= ordered[-1][0]:
        return ordered[-1][1]
    for (t0, x0), (t1, x1) in zip(ordered, ordered[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return x0
            frac = float((t - t0) / (t1 - t0))
            return x0 + frac * (x1 - x0)
    return left + float(t / duration_whole) * (right - left)


def _draw_point(draw, x: int, row_top: int, row_bottom: int, label: str, font):
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    y = row_top + max(1, (row_bottom - row_top - th) // 2 - 1)
    draw.text((x - tw // 2, y), label, fill=(0, 0, 0), font=font)


def render_original_pdf_with_recursive_levels_exact(
    pdf_path: Path, omr_path: Path, measures: list[Measure], result: dict,
) -> tuple[list[str], list[dict]]:
    doc = fitz.open(str(pdf_path))
    geometry = parse_omr_geometry(omr_path, len(measures))
    if len(geometry) != len(doc):
        doc.close()
        raise RuntimeError(
            f"OMR/PDF registration page mismatch: OMR has {len(geometry)} pages, PDF has {len(doc)}."
        )

    levels: dict[int, list[dict]] = {int(k): v for k, v in result.get("levels", {}).items() if v}
    highest = max(levels) if levels else 1
    points_by_level_measure: dict[int, dict[int, list[dict]]] = {
        level: {} for level in range(1, highest + 1)
    }
    for level, pts in levels.items():
        for p in pts:
            points_by_level_measure[level].setdefault(int(p["measure_index"]), []).append(p)

    pages_out: list[str] = []
    diagnostics: list[dict] = []
    unmatched_attack_errors: list[str] = []

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_ZOOM, PDF_ZOOM), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image)
        geom = geometry[page_index]
        scale_x = image.width / float(geom["source_width"])
        scale_y = image.height / float(geom["source_height"])
        font_label = _font(max(22, int(image.width / 72)), bold=True)
        font_title = _font(max(18, int(image.width / 90)), bold=True)
        row_height = max(48, int(image.width / 34))
        page_diag = {
            "page_index": page_index,
            "registration": "strict-audiveris-omr-time-slot-for-attacks",
            "highest_level": highest,
            "systems": [],
        }

        for system in geom["systems"]:
            left = int(round(float(system["staff_left"]) * scale_x))
            right = int(round(float(system["staff_right"]) * scale_x))
            staff_top = int(round(float(system["staff_top"]) * scale_y))
            l1_bottom = max(20, staff_top - 10)
            l1_top = max(4, l1_bottom - row_height)
            rows: dict[int, tuple[int, int]] = {1: (l1_top, l1_bottom)}
            for level in range(2, highest + 1):
                lower_top = rows[level - 1][0]
                rows[level] = (max(4, lower_top - row_height), lower_top)

            fills = [
                (242,242,242),(225,225,225),(208,208,208),(191,191,191),(174,174,174),
                (157,157,157),(140,140,140),(123,123,123),(106,106,106),(89,89,89),
            ]
            label_x = max(4, left - int(image.width * 0.055))
            for level in range(1, highest + 1):
                top, bottom = rows[level]
                fill = fills[min(level - 1, len(fills) - 1)]
                draw.rounded_rectangle([left, top, right, bottom], radius=6, fill=fill,
                                       outline=(170,170,170), width=1)
                title_y = top + max(2, (bottom - top - font_title.size) // 2)
                draw.text((label_x, title_y), f"L{level}", fill=(30,30,30), font=font_title)

            system_diag = {
                "system_index": system["system_index"],
                "measure_numbers": [],
                "levels": {
                    str(level): {
                        "exact_attack_slots": 0,
                        "missing_attack_slots": 0,
                        "structural_nonattack_labels": 0,
                    }
                    for level in range(1, highest + 1)
                },
            }

            for stack in system["stacks"]:
                mi = int(stack["measure_index"])
                if not (0 <= mi < len(measures)):
                    continue
                measure = measures[mi]
                system_diag["measure_numbers"].append(measure.number)
                duration_whole = Fraction(measure.end - measure.start, 4)

                for level in range(1, highest + 1):
                    row_top, row_bottom = rows[level]
                    for point in points_by_level_measure[level].get(mi, []):
                        t = parse_fraction(point["time_quarter"])
                        local_quarter = t - measure.start - measure.pickup_shift
                        local_whole = Fraction(local_quarter, 4)

                        if bool(point.get("attack")):
                            x_omr = exact_slot_x(stack, local_whole)
                            if x_omr is None:
                                system_diag["levels"][str(level)]["missing_attack_slots"] += 1
                                unmatched_attack_errors.append(
                                    f"page {page_index+1}, system {system['system_index']+1}, "
                                    f"measure {measure.number}, Level {level}, time {ftxt(t)}"
                                )
                                continue
                            system_diag["levels"][str(level)]["exact_attack_slots"] += 1
                        else:
                            system_diag["levels"][str(level)]["structural_nonattack_labels"] += 1
                            x_omr = structural_x(stack, local_whole, duration_whole)

                        x = int(round(float(x_omr) * scale_x))
                        _draw_point(draw, x, row_top, row_bottom, str(point["label"]), font_label)

            page_diag["systems"].append(system_diag)

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        pages_out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        diagnostics.append(page_diag)

    doc.close()
    if unmatched_attack_errors:
        sample = "; ".join(unmatched_attack_errors[:12])
        raise RuntimeError(
            f"Strict OMR registration refused to interpolate {len(unmatched_attack_errors)} attack label(s). "
            f"Examples: {sample}"
        )
    return pages_out, diagnostics


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Tone-Metric Recursive Levels — Clean</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;color:#111;background:#fff}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}.note{font-size:12px;color:#444;margin-top:8px}#score{margin-top:28px}.score-title{font-size:20px;font-weight:700;margin:0 0 8px}.score-page{display:block;max-width:100%;height:auto;margin:18px auto;border:1px solid #ccc;box-shadow:0 2px 10px rgba(0,0,0,.08);background:white}.score-note{font-size:13px;color:#444;margin-bottom:12px}table{border-collapse:collapse;margin-top:18px;font-size:13px}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}
</style></head><body>
<h1>Tone-Metric Analyzer — Recursive Levels</h1>
<div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Opening meter<select id="meter"><option selected>4/4</option><option>2/2</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze Recursive Levels</button></div>
<div id="status">No analysis yet.</div>
<div class="note">Single self-contained runtime. No previous analyzer version is imported. Attack labels require exact Audiveris OMR note/chord slots; only non-attack structural positions may be interpolated.</div>
<div id="table"></div><div id="score"></div>
<script>
const $=id=>document.getElementById(id);
function render(d){const levels=d.levels||{};let h='<table><thead><tr><th>Level</th><th>Sequence pos.</th><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Label</th><th>Attack?</th></tr></thead><tbody>';for(let lev=1;lev<=d.highest_level;lev++)for(const p of (levels[String(lev)]||[]))h+=`<tr><td>${lev}</td><td>${p.sequence_position??''}</td><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;h+='</tbody></table>';$('table').innerHTML=h;const pages=d.pdf_pages||[];if(pages.length){let q=`<div class="score-title">Original PDF + recursive Levels 1–${d.highest_level}</div><div class="score-note">All attack labels are registered only to exact OMR note/chord slots.</div>`;pages.forEach((src,i)=>q+=`<img class="score-page" src="${src}" alt="Analyzed score page ${i+1}">`);$('score').innerHTML=q}else $('score').innerHTML=''}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Building recursive Levels…';$('table').innerHTML='';$('score').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Complete — ${d.measure_count} measures, Levels 1–${d.highest_level}.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}};
</script></body></html>'''


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "mode": "standalone-clean-recursive-levels",
        "imports_previous_analyzer_versions": False,
        "imports_tone_metric_package": False,
        "recursive_scope": "tactus-denomination-only",
        "attack_alignment": "strict-exact-omr-slot-no-fallback",
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), initial_meter: str = Form("4/4")):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".mxl", ".musicxml", ".xml"}:
        raise HTTPException(400, "Upload PDF, MXL, MusicXML, or XML.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Upload exceeds 80 MB.")
    try:
        meter = parse_meter(initial_meter)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="tm-recursive-v8-clean-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            omr_path = None
            if suffix == ".pdf":
                audiveris_out = work / "audiveris"
                symbolic = pdf_to_musicxml(source, audiveris_out)
                omr_path = find_omr(audiveris_out, symbolic)
            else:
                symbolic = source

            measures, attacks = parse_score(symbolic, meter)
            result = analyze_recursive_levels(measures, attacks)
            result["version"] = APP_VERSION
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "opening_meter": f"{meter[0]}/{meter[1]}",
            }
            if suffix == ".pdf" and omr_path is not None:
                pages, diagnostics = render_original_pdf_with_recursive_levels_exact(
                    source, omr_path, measures, result
                )
                result["pdf_pages"] = pages
                result["pdf_overlay_diagnostics"] = diagnostics
            else:
                result["pdf_pages"] = []
                result["pdf_overlay_diagnostics"] = []
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Recursive-level analysis failed: {exc}") from exc
