from __future__ import annotations

import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

APP_VERSION = "0.9.0-recursive-canonical-omr-clean"
MAX_UPLOAD = 80 * 1024 * 1024
PDF_ZOOM = 2.0

# CLEAN RUNTIME CONTRACT
# ----------------------
# This file imports no previous Tone-Metric runtime/version and no tone_metric package.
# PDF input is interpreted through Audiveris MusicXML + the saved OMR notation graph.
# For PDFs, the OMR graph is the attack authority: note/rest/chord columns, ties,
# augmentation dots, beams/flags, voices/slots and physical x/y are read before
# recursive Levels are computed. MusicXML provides the measure/meter framework.
# Attack labels are drawn only at the physical canonical OMR column that generated
# the attack. Structural non-attack labels may be interpolated between canonical
# columns because no printed note/chord exists at those structural positions.
app = FastAPI(title="Tone-Metric Recursive Levels — Canonical OMR", version=APP_VERSION)


@dataclass(frozen=True)
class MeterProfile:
    numerator: int
    denominator: int
    beat_unit: Fraction
    arity: int


PROFILES = {
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

    @property
    def full_duration(self) -> Fraction:
        return self.end - self.start


@dataclass(frozen=True)
class Attack:
    onset: Fraction


@dataclass
class CanonicalColumn:
    measure_index: int
    page_index: int
    system_index: int
    stack_index: int
    x_abs: float
    y_abs: float
    onset_quarter: Fraction
    duration_quarter: Fraction
    attack: bool
    rest: bool
    tie_continuation_only: bool
    timing_source: str
    confidence: float
    notation_events: list[dict]

    def row(self) -> dict:
        d = asdict(self)
        d["onset_quarter"] = str(self.onset_quarter)
        d["duration_quarter"] = str(self.duration_quarter)
        return d


def lname(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def child(el, name: str):
    return next((c for c in el if lname(c.tag) == name), None)


def children(el, name: str):
    return [c for c in el if lname(c.tag) == name]


def text(el, name: str, default: str | None = None):
    c = child(el, name)
    return default if c is None or c.text is None else c.text.strip()


def ftxt(v: Fraction) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def parse_fraction(v) -> Fraction:
    return Fraction(str(v))


def parse_meter(v: str | None) -> tuple[int, int]:
    raw = (v or "").strip().lower()
    if raw in {"", "auto"}:
        return (4, 4)
    if "/" not in raw:
        raise ValueError("Meter must be numerator/denominator.")
    a, b = map(int, raw.split("/", 1))
    meter = (a, b)
    if meter not in PROFILES:
        raise ValueError(f"Unsupported meter {a}/{b}.")
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
                r = etree.fromstring(zf.read("META-INF/container.xml"))
                rootfile = next(
                    (n.get("full-path") for n in r.iter() if lname(n.tag) == "rootfile" and n.get("full-path")),
                    None,
                )
            if not rootfile:
                candidates = [
                    n for n in names
                    if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF/")
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
    if beats and beat_type and "+" not in beats:
        return int(beats), int(beat_type)
    symbol = (t.get("symbol") or "").lower()
    if symbol == "common":
        return (4, 4)
    if symbol == "cut":
        return (2, 2)
    return None


def scan_measure(measure_el, divisions: int):
    cursor = Fraction(0)
    max_cursor = Fraction(0)
    previous_note_onset = Fraction(0)
    rows: list[tuple[Fraction, bool, bool]] = []
    for item in measure_el:
        tag = lname(item.tag)
        if tag == "note":
            if child(item, "grace") is not None:
                continue
            chord = child(item, "chord") is not None
            dur = Fraction(int(text(item, "duration", "0") or 0), max(1, divisions))
            onset = previous_note_onset if chord else cursor
            previous_note_onset = onset
            if not chord:
                cursor += dur
            max_cursor = max(max_cursor, onset + dur, cursor)
            rest = child(item, "rest") is not None
            tied_stop = any((n.get("type") or "").lower() == "stop" for n in children(item, "tie"))
            notations = child(item, "notations")
            if notations is not None:
                tied_stop = tied_stop or any(
                    lname(n.tag) == "tied" and (n.get("type") or "").lower() == "stop"
                    for n in notations.iter()
                )
            tuplet = child(item, "time-modification") is not None
            rows.append((onset, (not rest) and (not tied_stop), tuplet))
        elif tag == "backup":
            cursor -= Fraction(int(text(item, "duration", "0") or 0), max(1, divisions))
            cursor = max(Fraction(0), cursor)
        elif tag == "forward":
            cursor += Fraction(int(text(item, "duration", "0") or 0), max(1, divisions))
            max_cursor = max(max_cursor, cursor)
    return max_cursor, rows


def parse_score_framework(path: Path, opening_meter: tuple[int, int]):
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
            Measure(mi, m.get("number") or str(mi + 1), global_start, global_start + full, pickup, inherited)
        )
        global_start += full

    # Symbolic attack map is used for symbolic files only. For PDF input the OMR
    # canonical notation graph below is the attack authority.
    attack_status = defaultdict(lambda: [False, False])
    for part in parts:
        divisions = 1
        for mi, m in enumerate(children(part, "measure")[: len(measures)]):
            attrs = child(m, "attributes")
            if attrs is not None:
                d = text(attrs, "divisions")
                if d:
                    divisions = max(1, int(d))
            _, rows = scan_measure(m, divisions)
            for onset, sounding, tuplet in rows:
                if not sounding:
                    continue
                key = (mi, onset)
                attack_status[key][0] = True
                if not tuplet:
                    attack_status[key][1] = True
    attacks: list[Attack] = []
    for (mi, onset), (_sounding, eligible) in sorted(attack_status.items()):
        if not eligible:
            continue
        mm = measures[mi]
        t = mm.start + mm.pickup_shift + onset
        if mm.start <= t < mm.end:
            attacks.append(Attack(t))
    return measures, attacks


def _bounds(el):
    if el is None:
        return None
    b = next((c for c in el if lname(c.tag) == "bounds"), None)
    if b is None:
        return None
    try:
        return tuple(float(b.get(k)) for k in ("x", "y", "w", "h"))
    except Exception:
        return None


def _frac(v):
    try:
        return Fraction(str(v))
    except Exception:
        return None


def _rest_duration(shape: str):
    s = (shape or "").upper()
    fixed = {
        "LONG_REST": Fraction(8),
        "WHOLE_REST": Fraction(4),
        "HALF_REST": Fraction(2),
        "QUARTER_REST": Fraction(1),
        "EIGHTH_REST": Fraction(1, 2),
    }
    for name, duration in fixed.items():
        if name in s:
            return duration
    m = re.search(r"(?:ONE_)?(\d+)(?:TH|ND|ST|RD)_REST", s)
    return Fraction(4, int(m.group(1))) if m else None


def _dot_multiplier(count: int) -> Fraction:
    out = Fraction(1)
    add = Fraction(1, 2)
    for _ in range(max(0, count)):
        out += add
        add /= 2
    return out


def _fraction_gcd(values: list[Fraction]) -> Fraction:
    vals = [Fraction(v) for v in values if v and v > 0]
    if not vals:
        return Fraction(1, 2)
    den = 1
    for v in vals:
        den = den * v.denominator // gcd(den, v.denominator)
    nums = [int(v * den) for v in vals]
    g = nums[0]
    for n in nums[1:]:
        g = gcd(g, n)
    return Fraction(g, den)


def _sheet_no(name: str) -> int:
    m = re.search(r"sheet#(\d+)", name, re.I)
    return int(m.group(1)) if m else 10**9


def recover_canonical_columns(omr_path: Path, measures: list[Measure], cluster_tolerance_px: float = 12.0):
    """Recover a metric score grid from the saved Audiveris notation graph.

    This follows the v0.15.2 canonical-score design: physical note/rest columns are
    read first; note value, dots, beams/flags and ties determine event semantics;
    Audiveris BEGIN slots are semantic anchors; missing/contradictory slot timing is
    filled on a rational metric lattice without creating new attack columns.
    """
    out: list[CanonicalColumn] = []
    meta = {
        "available": False,
        "architecture": "metric-grid-first/notation-semantics/global-attack-map",
        "warnings": [],
        "symbol_counts": Counter(),
        "measures": [],
    }
    global_measure = 0
    with zipfile.ZipFile(omr_path) as zf:
        members = sorted(
            [n for n in zf.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=_sheet_no,
        )
        for page_index, member in enumerate(members):
            root = etree.fromstring(zf.read(member))
            for node in root.iter():
                meta["symbol_counts"][lname(node.tag)] += 1
            page = next((e for e in root if lname(e.tag) == "page"), None)
            if page is None:
                continue
            for system_index, system in enumerate([e for e in page if lname(e.tag) == "system"]):
                objects = {str(e.get("id")): e for e in system.iter() if e.get("id") is not None}
                chord_members = defaultdict(list)
                chord_stem = {}
                stem_beams = defaultdict(set)
                stem_flags = defaultdict(list)
                dotted_heads = Counter()
                tie_right_heads = set()

                for rel in [e for e in system.iter() if lname(e.tag) == "relation"]:
                    typ = next((lname(c.tag) for c in rel if isinstance(c.tag, str)), "")
                    source = str(rel.get("source") or "")
                    target = str(rel.get("target") or "")
                    if typ == "containment" and source in objects and target in objects:
                        if lname(objects[source].tag) in {"head-chord", "rest-chord"}:
                            chord_members[source].append(target)
                    elif typ == "chord-stem":
                        chord_stem[source] = target
                    elif typ == "beam-stem":
                        stem_beams[target].add(source)
                    elif typ == "flag-stem":
                        stem_flags[target].append(source)
                    elif typ == "augmentation":
                        if source in objects and lname(objects[source].tag) == "head":
                            dotted_heads[source] += 1
                        if target in objects and lname(objects[target].tag) == "head":
                            dotted_heads[target] += 1
                    elif typ == "slur-head" and source in objects and target in objects:
                        slur = objects[source]
                        if lname(slur.tag) == "slur" and (slur.get("tie") or "").lower() == "true":
                            spec = next((c for c in rel if lname(c.tag) == "slur-head"), None)
                            if spec is not None and (spec.get("side") or "").upper() == "RIGHT":
                                tie_right_heads.add(target)

                stacks = [e for e in system if lname(e.tag) == "stack"]
                parts = [e for e in system if lname(e.tag) == "part"]
                part_measures = [[m for m in p if lname(m.tag) == "measure"] for p in parts]

                for stack_index, stack in enumerate(stacks):
                    if global_measure >= len(measures):
                        break
                    measure = measures[global_measure]
                    full = measure.full_duration
                    left = float(stack.get("left") or 0.0)
                    right = float(stack.get("right") or left + 1.0)
                    slot_time = {}
                    for sl in [e for e in stack if lname(e.tag) == "slot"]:
                        sid = str(sl.get("id") or "")
                        t = _frac(sl.get("time-offset"))
                        if sid and t is not None:
                            slot_time[sid] = t * 4

                    measure_nodes = [pm[stack_index] for pm in part_measures if stack_index < len(pm)]
                    head_chords: list[str] = []
                    rest_chords: list[str] = []
                    begin_times = defaultdict(list)
                    for meas in measure_nodes:
                        for tag, bucket in (("head-chords", head_chords), ("rest-chords", rest_chords)):
                            el = next((c for c in meas if lname(c.tag) == tag), None)
                            if el is not None and el.text:
                                bucket.extend(el.text.split())
                        for voice in [c for c in meas if lname(c.tag) == "voice"]:
                            slots_el = next((c for c in voice if lname(c.tag) == "slots"), None)
                            if slots_el is None:
                                continue
                            for entry in [c for c in slots_el if lname(c.tag) == "entry"]:
                                key = next((c for c in entry if lname(c.tag) == "key"), None)
                                val = next((c for c in entry if lname(c.tag) == "value"), None)
                                if key is None or val is None or not key.text:
                                    continue
                                cid = val.get("chord")
                                t = slot_time.get(key.text.strip())
                                if cid and t is not None and (val.get("status") or "").upper() == "BEGIN":
                                    begin_times[str(cid)].append(t)

                    events: list[dict] = []
                    for cid in dict.fromkeys(head_chords):
                        cel = objects.get(cid)
                        cb = _bounds(cel)
                        if cb is None:
                            continue
                        x, y, w, h = cb
                        members = [objects[m] for m in chord_members.get(cid, []) if m in objects]
                        heads = [m for m in members if lname(m.tag) == "head"]
                        shapes = " ".join((head.get("shape") or "").upper() for head in heads)
                        base = Fraction(4) if "WHOLE" in shapes else Fraction(2) if ("VOID" in shapes or "HALF" in shapes) else Fraction(1)
                        stem_id = chord_stem.get(cid)
                        divisions = len(stem_beams.get(stem_id, set())) if stem_id else 0
                        for flag_id in stem_flags.get(stem_id, []):
                            shape = (objects.get(flag_id).get("shape") if flag_id in objects else "") or ""
                            m = re.search(r"FLAG_(\d+)", shape.upper())
                            divisions = max(divisions, int(m.group(1)) if m else 1)
                        duration = base / (2 ** divisions) if divisions and base <= 1 else base
                        dot_count = max((dotted_heads.get(str(hd.get("id")), 0) for hd in heads), default=0)
                        duration *= _dot_multiplier(dot_count)
                        all_tied = bool(heads) and all(str(hd.get("id")) in tie_right_heads for hd in heads)
                        events.append({
                            "kind": "head",
                            "cid": cid,
                            "x": x + w / 2,
                            "y": y + h / 2,
                            "duration": duration,
                            "dot_count": dot_count,
                            "attack": not all_tied,
                            "tie": all_tied,
                            "rest": False,
                            "begin": list(begin_times.get(cid, [])),
                        })

                    for cid in dict.fromkeys(rest_chords):
                        cel = objects.get(cid)
                        cb = _bounds(cel)
                        if cb is None:
                            continue
                        x, y, w, h = cb
                        members = [objects[m] for m in chord_members.get(cid, []) if m in objects]
                        rests = [m for m in members if lname(m.tag) == "rest"]
                        shape = rests[0].get("shape", "") if rests else ""
                        duration = _rest_duration(shape) or Fraction(1)
                        if "WHOLE_REST" in shape.upper() and duration >= full:
                            continue
                        events.append({
                            "kind": "rest",
                            "cid": cid,
                            "x": x + w / 2,
                            "y": y + h / 2,
                            "duration": duration,
                            "dot_count": 0,
                            "attack": False,
                            "tie": False,
                            "rest": True,
                            "begin": list(begin_times.get(cid, [])),
                        })

                    events.sort(key=lambda e: e["x"])
                    clusters: list[list[dict]] = []
                    for event in events:
                        if not clusters or event["x"] - clusters[-1][-1]["x"] > cluster_tolerance_px:
                            clusters.append([event])
                        else:
                            clusters[-1].append(event)

                    columns = []
                    for group in clusters:
                        rep = min(group, key=lambda a: sum(abs(a["x"] - b["x"]) for b in group))
                        head_events = [e for e in group if e["kind"] == "head"]
                        durations = [e["duration"] for e in group if e["duration"] > 0]
                        columns.append({
                            "x": rep["x"],
                            "y": rep["y"],
                            "events": group,
                            "duration": max(durations, default=Fraction(1)),
                            "attack": any(e["attack"] for e in head_events),
                            "rest": any(e["rest"] for e in group),
                            "tie_only": bool(head_events) and all(e["tie"] for e in head_events),
                            "begin": [t for e in group for t in e["begin"]],
                        })

                    duration_values = [
                        e["duration"] for c in columns for e in c["events"] if e["duration"] > 0
                    ]
                    assigned_unique = sorted({t for c in columns for t in c["begin"] if 0 <= t < full})
                    duration_values.extend(b - a for a, b in zip(assigned_unique, assigned_unique[1:]) if b > a)
                    quantum = _fraction_gcd(duration_values)
                    if quantum > 1:
                        quantum = Fraction(1)
                    while columns and int(full / quantum) < len(columns):
                        quantum /= 2
                    if quantum < Fraction(1, 32):
                        quantum = Fraction(1, 32)
                        while columns and int(full / quantum) < len(columns):
                            quantum /= 2
                    units = max(1, int(full / quantum))

                    positions = []
                    prev = -1
                    for i, col in enumerate(columns):
                        candidate_units = sorted({
                            int(t / quantum)
                            for t in col["begin"]
                            if 0 <= t < full and (t / quantum).denominator == 1
                        })
                        hi = units - (len(columns) - i)
                        feasible = [j for j in candidate_units if j > prev and j <= hi]
                        if feasible:
                            j = feasible[0]
                            timing_source = "semantic-begin-anchor"
                            confidence = 0.95
                        else:
                            lo = prev + 1
                            frac = 0.0 if right <= left else (col["x"] - left) / (right - left)
                            ideal = int(round(frac * (units - 1)))
                            j = max(lo, min(hi, ideal))
                            timing_source = "canonical-grid-interpolation"
                            confidence = 0.72
                        positions.append((j, timing_source, confidence))
                        prev = j

                    measure_rows = []
                    for col, (grid_index, timing_source, confidence) in zip(columns, positions):
                        canonical = CanonicalColumn(
                            measure_index=global_measure,
                            page_index=page_index,
                            system_index=system_index,
                            stack_index=stack_index,
                            x_abs=float(col["x"]),
                            y_abs=float(col["y"]),
                            onset_quarter=Fraction(grid_index) * quantum,
                            duration_quarter=col["duration"],
                            attack=bool(col["attack"]),
                            rest=bool(col["rest"]),
                            tie_continuation_only=bool(col["tie_only"]),
                            timing_source=timing_source,
                            confidence=confidence,
                            notation_events=[
                                {
                                    "kind": e["kind"],
                                    "duration_quarter": str(e["duration"]),
                                    "dot_count": int(e["dot_count"]),
                                    "attack": bool(e["attack"]),
                                    "tie_continuation": bool(e["tie"]),
                                    "rest": bool(e["rest"]),
                                }
                                for e in col["events"]
                            ],
                        )
                        out.append(canonical)
                        measure_rows.append(canonical.row())

                    meta["measures"].append({
                        "measure_index": global_measure,
                        "measure_number": measure.number,
                        "page_index": page_index,
                        "system_index": system_index,
                        "stack_index": stack_index,
                        "stack_left": left,
                        "stack_right": right,
                        "quantum_quarter": str(quantum),
                        "physical_column_count": len(columns),
                        "attack_column_count": sum(1 for c in columns if c["attack"]),
                        "columns": measure_rows,
                    })
                    global_measure += 1

    meta["symbol_counts"] = dict(meta["symbol_counts"])
    meta["available"] = bool(out)
    meta["column_count"] = len(out)
    meta["attack_column_count"] = sum(1 for c in out if c.attack)
    meta["rest_column_count"] = sum(1 for c in out if c.rest)
    meta["tie_continuation_column_count"] = sum(1 for c in out if c.tie_continuation_only)
    meta["augmentation_dot_count"] = sum(
        e.get("dot_count", 0) for c in out for e in c.notation_events
    )
    return out, meta


def attacks_from_canonical(columns: list[CanonicalColumn], measures: list[Measure]) -> list[Attack]:
    times = set()
    for c in columns:
        if not c.attack:
            continue
        if not (0 <= c.measure_index < len(measures)):
            continue
        measure = measures[c.measure_index]
        t = measure.start + measure.pickup_shift + c.onset_quarter
        if measure.start <= t < measure.end:
            times.add(t)
    return [Attack(t) for t in sorted(times)]


def meter_segments(measures: list[Measure]):
    if not measures:
        return []
    out = []
    start = 0
    current = measures[0].meter
    for i in range(1, len(measures) + 1):
        if i == len(measures) or measures[i].meter != current:
            out.append((start, i - 1))
            if i < len(measures):
                start = i
                current = measures[i].meter
    return out


def _measure_for_time(measures, a, b, t):
    return next(m for m in measures[a : b + 1] if m.start <= t < m.end)


def _point(measures, a, b, profile, si, t, pos, level, attack_times, span_start=None, span_end=None):
    m = _measure_for_time(measures, a, b, t)
    is_attack = t in attack_times
    row = {
        "segment_index": si,
        "sequence_position": pos,
        "time_quarter": ftxt(t),
        "measure_index": m.index,
        "measure_number": m.number,
        "beat": int((t - m.start) / profile.beat_unit) + 1,
        "level": level,
        "attack": is_attack,
        "parenthetical": not is_attack,
        "label": str(level) if is_attack else f"({level})",
    }
    if span_start is not None:
        row["recursive_span_start_quarter"] = ftxt(span_start)
        row["recursive_span_end_quarter"] = None if span_end is None else ftxt(span_end)
    return row


def analyze_level1(measures, attacks):
    attack_times = {a.onset for a in attacks}
    points = []
    segments = []
    for si, (a, b) in enumerate(meter_segments(measures)):
        profile = PROFILES[measures[a].meter]
        start, end = measures[a].start, measures[b].end
        seq = sequence(profile.arity, int((end - start) / profile.beat_unit) + 1)
        for pos in seq:
            t = start + (pos - 1) * profile.beat_unit
            if start <= t < end:
                points.append(_point(measures, a, b, profile, si, t, pos, 1, attack_times))
        segments.append({
            "segment_index": si,
            "start_measure": measures[a].number,
            "end_measure": measures[b].number,
            "meter": f"{profile.numerator}/{profile.denominator}",
            "arity": profile.arity,
            "sequence": seq,
        })
    return {
        "engine_contract": "canonical-omr+recursive-levels-v0.9",
        "points": points,
        "segments": segments,
        "measure_count": len(measures),
        "attack_count": len(attack_times),
    }


def _group(points):
    grouped = defaultdict(list)
    for p in points:
        grouped[int(p["segment_index"])].append(parse_fraction(p["time_quarter"]))
    return {k: sorted(set(v)) for k, v in grouped.items()}


def _all_tactus_times(measures):
    out = {}
    for si, (a, b) in enumerate(meter_segments(measures)):
        profile = PROFILES[measures[a].meter]
        count = int((measures[b].end - measures[a].start) / profile.beat_unit)
        out[si] = {measures[a].start + k * profile.beat_unit for k in range(count)}
    return out


def analyze_next_level(measures, attacks, previous, covered_before, level):
    attack_times = {a.onset for a in attacks}
    by_segment = _group(previous)
    out = {}
    for si, (a, b) in enumerate(meter_segments(measures)):
        profile = PROFILES[measures[a].meter]
        beat = profile.beat_unit
        end = measures[b].end
        boundaries = by_segment.get(si, [])
        covered = covered_before.get(si, set())

        def add(t, pos, span_start, span_end):
            if measures[a].start <= t < end:
                out[(si, t)] = _point(
                    measures, a, b, profile, si, t, pos, level,
                    attack_times, span_start, span_end,
                )

        for t0, t1 in zip(boundaries, boundaries[1:]):
            steps = int((t1 - t0) / beat)
            if steps <= 1:
                continue
            if not any(t0 + k * beat not in covered for k in range(1, steps)):
                continue
            for pos in sequence(profile.arity, steps + 1):
                t = t0 + (pos - 1) * beat
                if t > t1:
                    break
                add(t, pos, t0, t1)

        if boundaries:
            t0 = boundaries[-1]
            count = int((end - t0) / beat)
            if count > 1 and any(t0 + k * beat not in covered for k in range(1, count)):
                for pos in sequence(profile.arity, count):
                    t = t0 + (pos - 1) * beat
                    if t < end:
                        add(t, pos, t0, None)

    return [out[k] for k in sorted(out)]


def analyze_recursive_levels(measures, attacks):
    result = analyze_level1(measures, attacks)
    level1 = result["points"]
    levels = {"1": level1}
    all_tactus = _all_tactus_times(measures)
    covered = {si: set() for si in all_tactus}
    for p in level1:
        covered[int(p["segment_index"])].add(parse_fraction(p["time_quarter"]))

    previous = level1
    level = 2
    while not all(all_tactus[si] <= covered[si] for si in all_tactus):
        next_points = analyze_next_level(measures, attacks, previous, covered, level)
        if not next_points:
            raise RuntimeError(f"Recursive construction stalled at Level {level}.")
        before = sum(len(v) for v in covered.values())
        for p in next_points:
            covered[int(p["segment_index"])].add(parse_fraction(p["time_quarter"]))
        after = sum(len(v) for v in covered.values())
        if after <= before:
            raise RuntimeError(f"Recursive Level {level} added no new tactus position.")
        levels[str(level)] = next_points
        previous = next_points
        level += 1

    result["levels"] = levels
    result["highest_level"] = max(map(int, levels))
    result["recursive_levels_complete"] = True
    result["recursive_scope"] = "tactus-denomination-only"
    for k, pts in levels.items():
        result[f"level{k}_points"] = pts
    return result


def find_audiveris():
    configured = os.environ.get("AUDIVERIS_CMD", "").strip()
    if configured and Path(configured).exists():
        return configured
    for candidate in ("Audiveris", "audiveris", "/opt/audiveris/bin/Audiveris"):
        found = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if found and Path(found).exists():
            return str(found)
    raise RuntimeError("Audiveris is not available.")


def pdf_to_musicxml(pdf: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_audiveris(), "-batch", "-transcribe", "-save", "-export",
        "-output", str(out), "--", str(pdf),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    if p.returncode != 0:
        raise RuntimeError(f"Audiveris failed with exit code {p.returncode}.\n\n{(p.stdout or '')[-12000:]}")
    candidates = []
    for pattern in ("*.mxl", "*.musicxml", "*.xml"):
        candidates.extend(out.rglob(pattern))
    candidates = [x for x in candidates if "container.xml" not in str(x)]
    if not candidates:
        raise RuntimeError("Audiveris produced no MusicXML export.")
    candidates.sort(key=lambda x: (0 if x.suffix.lower() == ".mxl" else 1, len(str(x))))
    return candidates[0]


def find_omr(out: Path):
    candidates = list(out.rglob("*.omr"))
    if not candidates:
        raise RuntimeError("Audiveris produced no saved OMR project.")
    return sorted(candidates, key=lambda p: -p.stat().st_size)[0]


def _font(size: int, bold=False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def _omr_page_geometry(omr: Path, measure_count: int):
    geometry = []
    global_measure = 0
    with zipfile.ZipFile(omr) as zf:
        sheets = sorted(
            [n for n in zf.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=_sheet_no,
        )
        for page_index, name in enumerate(sheets):
            root = etree.fromstring(zf.read(name))
            picture = next((n for n in root if lname(n.tag) == "picture"), None)
            page = next((n for n in root if lname(n.tag) == "page"), None)
            if picture is None or page is None:
                raise RuntimeError(f"OMR page {page_index + 1} lacks picture/page geometry.")
            systems = []
            for system_index, system in enumerate([n for n in page if lname(n.tag) == "system"]):
                xs, ys = [], []
                for staff in [n for n in system.iter() if lname(n.tag) == "staff"]:
                    try:
                        xs.extend([float(staff.get("left")), float(staff.get("right"))])
                    except Exception:
                        pass
                    for p in staff.iter():
                        if lname(p.tag) == "point":
                            try:
                                ys.append(float(p.get("y")))
                            except Exception:
                                pass
                stacks = []
                for stack in [n for n in system if lname(n.tag) == "stack"]:
                    if global_measure >= measure_count:
                        break
                    stacks.append({
                        "measure_index": global_measure,
                        "left": float(stack.get("left") or 0),
                        "right": float(stack.get("right") or 0),
                    })
                    global_measure += 1
                if stacks:
                    systems.append({
                        "system_index": system_index,
                        "left": min(xs) if xs else stacks[0]["left"],
                        "right": max(xs) if xs else stacks[-1]["right"],
                        "top": min(ys) if ys else 0,
                        "stacks": stacks,
                    })
            geometry.append({
                "width": float(picture.get("width")),
                "height": float(picture.get("height")),
                "systems": systems,
            })
    if global_measure != measure_count:
        raise RuntimeError(
            f"OMR has {global_measure} measure stacks while analysis has {measure_count} measures."
        )
    return geometry


def render_pdf(pdf: Path, omr: Path, measures: list[Measure], columns: list[CanonicalColumn], result: dict):
    by_measure = defaultdict(list)
    for c in columns:
        by_measure[c.measure_index].append(c)
    attack_x = {
        (c.measure_index, measures[c.measure_index].pickup_shift + c.onset_quarter): c.x_abs
        for c in columns if c.attack and 0 <= c.measure_index < len(measures)
    }
    geometry = _omr_page_geometry(omr, len(measures))
    doc = fitz.open(str(pdf))
    if len(doc) != len(geometry):
        doc.close()
        raise RuntimeError("PDF/OMR page count mismatch.")
    levels = {int(k): v for k, v in result["levels"].items()}
    pages_out, diagnostics = [], []

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_ZOOM, PDF_ZOOM), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image)
        geom = geometry[page_index]
        sx, sy = image.width / geom["width"], image.height / geom["height"]
        font = _font(max(22, int(image.width / 72)), True)
        title_font = _font(max(18, int(image.width / 90)), True)
        page_diag = {"page_index": page_index, "systems": []}

        for system in geom["systems"]:
            left, right = int(system["left"] * sx), int(system["right"] * sx)
            staff_top = int(system["top"] * sy)
            row_h = max(44, int(image.width / 36))
            rows = {1: (max(4, staff_top - 10 - row_h), max(20, staff_top - 10))}
            for level in range(2, result["highest_level"] + 1):
                rows[level] = (max(4, rows[level - 1][0] - row_h), rows[level - 1][0])
            for level, (top, bottom) in rows.items():
                shade = max(105, 245 - level * 10)
                draw.rectangle([left, top, right, bottom], fill=(shade, shade, shade))
                draw.text((max(4, left - int(image.width * 0.05)), top + 5), f"L{level}", font=title_font, fill=0)

            system_diag = {
                "system_index": system["system_index"],
                "levels": {
                    str(level): {"canonical_attack_aligned": 0, "missing_canonical_attack": 0}
                    for level in rows
                },
            }

            for stack in system["stacks"]:
                mi = stack["measure_index"]
                measure = measures[mi]
                duration = measure.full_duration
                anchors = sorted(
                    (measure.pickup_shift + c.onset_quarter, c.x_abs)
                    for c in by_measure.get(mi, [])
                )

                def structural_x(offset: Fraction):
                    points = [(Fraction(0), stack["left"])] + anchors + [(duration, stack["right"])]
                    points = sorted(dict(points).items())
                    for (t0, x0), (t1, x1) in zip(points, points[1:]):
                        if t0 <= offset <= t1:
                            if t1 == t0:
                                return x0
                            frac = float((offset - t0) / (t1 - t0))
                            return x0 + frac * (x1 - x0)
                    return stack["right"]

                for level, points in levels.items():
                    for p in points:
                        if int(p["measure_index"]) != mi:
                            continue
                        t = parse_fraction(p["time_quarter"])
                        offset = t - measure.start
                        if p["attack"]:
                            x = attack_x.get((mi, offset))
                            if x is None:
                                system_diag["levels"][str(level)]["missing_canonical_attack"] += 1
                                raise RuntimeError(
                                    f"No canonical OMR attack column for measure {measure.number}, "
                                    f"offset {ftxt(offset)}, Level {level}."
                                )
                            system_diag["levels"][str(level)]["canonical_attack_aligned"] += 1
                        else:
                            x = structural_x(offset)
                        top, bottom = rows[level]
                        label = str(p["label"])
                        bb = draw.textbbox((0, 0), label, font=font)
                        tw, th = bb[2] - bb[0], bb[3] - bb[1]
                        y = top + max(1, (bottom - top - th) // 2 - 1)
                        draw.text((int(x * sx) - tw // 2, y), label, font=font, fill=0)

            page_diag["systems"].append(system_diag)

        buf = io.BytesIO()
        image.save(buf, "PNG", optimize=True)
        pages_out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        diagnostics.append(page_diag)

    doc.close()
    return pages_out, diagnostics


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Tone-Metric Canonical OMR</title>
<style>body{font-family:Arial,sans-serif;margin:28px;color:#111}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}.note{font-size:12px;color:#444;margin-top:8px}.score-page{display:block;max-width:100%;height:auto;margin:18px auto;border:1px solid #ccc;box-shadow:0 2px 10px rgba(0,0,0,.08)}</style></head><body>
<h1>Tone-Metric Analyzer — Canonical OMR + Recursive Levels</h1>
<div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Opening meter<select id="meter"><option selected>4/4</option><option>2/2</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze</button></div>
<div id="status">No analysis yet.</div>
<div class="note">PDF analysis uses the Audiveris OMR notation graph to reconstruct canonical score time before the recursive Levels algorithm is run. No previous Tone-Metric runtime is imported.</div>
<div id="score"></div>
<script>const $=id=>document.getElementById(id);$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Reading notation and reconstructing canonical score time…';$('score').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Complete — ${d.measure_count} measures, ${d.attack_count} attacks, Levels 1–${d.highest_level}.`;let h='';(d.pdf_pages||[]).forEach((src,i)=>h+=`<img class="score-page" src="${src}" alt="Analyzed score page ${i+1}">`);$('score').innerHTML=h}catch(e){$('status').textContent='Analysis stopped: '+e.message}};</script></body></html>'''


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
        "runtime": "single-clean-file",
        "score_reader": "audiveris-omr-canonical-notation-graph",
        "previous_version_imports": False,
        "attack_authority_for_pdf": "canonical-omr-notation-columns",
        "recursive_scope": "tactus-denomination-only",
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

    with tempfile.TemporaryDirectory(prefix="tm-v9-canonical-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            omr_path = None
            if suffix == ".pdf":
                audiveris_out = work / "audiveris"
                symbolic = pdf_to_musicxml(source, audiveris_out)
                omr_path = find_omr(audiveris_out)
            else:
                symbolic = source

            measures, symbolic_attacks = parse_score_framework(symbolic, meter)
            if omr_path is not None:
                columns, canonical_meta = recover_canonical_columns(omr_path, measures)
                attacks = attacks_from_canonical(columns, measures)
                if not attacks and symbolic_attacks:
                    raise RuntimeError(
                        "Canonical OMR reader recovered no attacks; stopped instead of silently falling back to MusicXML attacks."
                    )
            else:
                columns = []
                canonical_meta = {"available": False, "architecture": "musicxml-symbolic-input"}
                attacks = symbolic_attacks

            result = analyze_recursive_levels(measures, attacks)
            result["version"] = APP_VERSION
            result["canonical_score_meta"] = canonical_meta
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "opening_meter": f"{meter[0]}/{meter[1]}",
            }
            if omr_path is not None:
                result["pdf_pages"], result["pdf_overlay_diagnostics"] = render_pdf(
                    source, omr_path, measures, columns, result
                )
            else:
                result["pdf_pages"] = []
                result["pdf_overlay_diagnostics"] = []
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Canonical recursive analysis failed: {exc}") from exc
