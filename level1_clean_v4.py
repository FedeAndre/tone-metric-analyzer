from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import fitz
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

APP_VERSION = "0.4.0-level1-clean-pdf"
MAX_UPLOAD = 80 * 1024 * 1024
PDF_ZOOM = 2.0

# CLEAN-ROOM RUNTIME:
# This file deliberately imports no tone_metric package, no dissertation_*
# module, no previous level1_clean_* module, and no legacy overlay code.
app = FastAPI(title="Tone-Metric Level 1", version=APP_VERSION)


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


@dataclass(frozen=True)
class Attack:
    onset: Fraction


@dataclass(frozen=True)
class LayoutMeasure:
    measure_index: int
    page_index: int
    system_index: int
    width: float


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
        raise ValueError(f"Unsupported Level-1 meter {meter[0]}/{meter[1]}.")
    return meter


def sequence(arity: int, limit: int) -> list[int]:
    if arity not in (2, 3):
        raise ValueError("Level 1 supports only binary/ternary dissertation sequences.")
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


def extract_layout_plan(path: Path, measure_count: int) -> list[LayoutMeasure]:
    """Read only page/system boundaries and measure widths from Audiveris MusicXML.

    This does not alter the analytical result. It is used solely to place the
    already-computed Level-1 labels over the original uploaded PDF pages.
    """
    root = xml_root(path)
    first, _ = first_part(root)
    page_index = 0
    system_index = 0
    out: list[LayoutMeasure] = []

    for mi, m in enumerate(children(first, "measure")[:measure_count]):
        pr = child(m, "print")
        if mi > 0 and pr is not None:
            new_page = (pr.get("new-page") or "").lower() == "yes"
            new_system = (pr.get("new-system") or "").lower() == "yes"
            if new_page:
                page_index += 1
                system_index = 0
            elif new_system:
                system_index += 1
        try:
            width = float(m.get("width") or 0.0)
        except Exception:
            width = 0.0
        out.append(LayoutMeasure(mi, page_index, system_index, width))

    # Some exports omit all print/system information. Keep the page at zero;
    # the PDF-side detector will then fall back to one sequential system.
    return out


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


def analyze_level1(measures: list[Measure], attacks: list[Attack]):
    attack_times = {a.onset for a in attacks}
    points = []
    segments = []
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
            m = next(x for x in measures[a : b + 1] if x.start <= t < x.end)
            is_attack = t in attack_times
            points.append(
                {
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
                }
            )
        segments.append(
            {
                "segment_index": si,
                "start_measure": first.number,
                "end_measure": last.number,
                "meter": f"{profile.numerator}/{profile.denominator}",
                "arity": profile.arity,
                "sequence": seq,
            }
        )
    return {
        "engine_contract": "dissertation-level1-clean-v0.4",
        "only_level_1": True,
        "level": 1,
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


def find_audiveris():
    configured = os.environ.get("AUDIVERIS_CMD", "").strip()
    if configured and Path(configured).exists():
        return configured
    for c in ("Audiveris", "audiveris", "/opt/audiveris/bin/Audiveris"):
        found = shutil.which(c) if not c.startswith("/") else c
        if found and Path(found).exists():
            return str(found)
    raise RuntimeError("Audiveris is not available.")


def pdf_to_musicxml(pdf: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_audiveris(),
        "-batch",
        "-transcribe",
        "-save",
        "-export",
        "-output",
        str(out),
        "--",
        str(pdf),
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


def detect_system_regions(image: Image.Image, expected_count: int) -> list[dict]:
    """Detect printed music systems from horizontal staff-line evidence.

    The expected count comes from Audiveris' new-system/new-page markers. Only
    the original PDF raster is inspected for positions; no old overlay geometry
    is reused.
    """
    expected_count = max(1, int(expected_count or 1))
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    x0, x1 = int(w * 0.05), int(w * 0.96)
    band = gray[:, x0:x1]
    density = (band < 170).mean(axis=1)

    # Horizontal staff rules are much denser than ordinary text/noteheads.
    threshold = 0.16
    rows = np.flatnonzero(density >= threshold)
    if len(rows) < expected_count * 5:
        threshold = 0.10
        rows = np.flatnonzero(density >= threshold)
    if len(rows) < expected_count:
        threshold = 0.055
        rows = np.flatnonzero(density >= threshold)

    # Collapse line thickness into one center per horizontal rule.
    centers: list[int] = []
    if len(rows):
        start = prev = int(rows[0])
        for r0 in rows[1:]:
            r = int(r0)
            if r <= prev + 3:
                prev = r
            else:
                centers.append((start + prev) // 2)
                start = prev = r
        centers.append((start + prev) // 2)

    # Ignore page-edge artifacts.
    centers = [y for y in centers if int(h * 0.04) <= y <= int(h * 0.97)]

    groups: list[list[int]] = []
    if centers:
        if expected_count == 1:
            groups = [centers]
        else:
            gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
            split_after = sorted(i for _, i in sorted(gaps, reverse=True)[: expected_count - 1])
            last = 0
            for idx in split_after:
                groups.append(centers[last : idx + 1])
                last = idx + 1
            groups.append(centers[last:])
            groups = [g for g in groups if g]

    # Fallback: distribute expected systems across the broad music-bearing area.
    if len(groups) != expected_count:
        ink_density = (band < 200).mean(axis=1)
        ink_rows = np.flatnonzero(ink_density > 0.008)
        if len(ink_rows):
            top = max(int(h * 0.08), int(ink_rows.min()))
            bottom = min(int(h * 0.95), int(ink_rows.max()))
        else:
            top, bottom = int(h * 0.10), int(h * 0.92)
        step = max(1, (bottom - top) // expected_count)
        groups = [[top + i * step, top + min((i + 1) * step, bottom - top)] for i in range(expected_count)]

    regions: list[dict] = []
    for gi, g in enumerate(groups):
        staff_top = int(min(g))
        staff_bottom = int(max(g))
        y0 = max(0, staff_top - 18)
        y1 = min(h, staff_bottom + 70)
        sub = gray[y0:y1, :]
        col_density = (sub < 195).mean(axis=0)
        cols = np.flatnonzero(col_density > 0.008)
        if len(cols):
            left = max(int(w * 0.035), int(cols.min()))
            right = min(int(w * 0.975), int(cols.max()))
        else:
            left, right = int(w * 0.07), int(w * 0.94)
        if right - left < w * 0.35:
            left, right = int(w * 0.07), int(w * 0.94)
        regions.append(
            {
                "system_index": gi,
                "staff_top": staff_top,
                "staff_bottom": staff_bottom,
                "left": left,
                "right": right,
            }
        )
    return regions


def _system_measure_plan(layout: list[LayoutMeasure], measures: list[Measure], pdf_page_count: int):
    if not layout:
        return {(0, 0): list(range(len(measures)))}

    # If Audiveris did not emit any system/page break markers, distribute
    # measures over a single system. Otherwise preserve its engraving order.
    has_breaks = any(l.page_index > 0 or l.system_index > 0 for l in layout)
    if not has_breaks:
        return {(0, 0): list(range(len(measures)))}

    out: dict[tuple[int, int], list[int]] = {}
    for l in layout:
        page = min(max(0, l.page_index), max(0, pdf_page_count - 1))
        out.setdefault((page, l.system_index), []).append(l.measure_index)
    return out


def _measure_weights(indices: list[int], layout_by_measure: dict[int, LayoutMeasure], measures: list[Measure]):
    weights = []
    for mi in indices:
        lw = layout_by_measure.get(mi)
        if lw is not None and lw.width > 1.0:
            weights.append(float(lw.width))
        else:
            weights.append(float(measures[mi].end - measures[mi].start))
    if not any(w > 0 for w in weights):
        weights = [1.0] * len(indices)
    return weights


def render_original_pdf_with_level1(
    pdf_path: Path,
    symbolic_path: Path,
    measures: list[Measure],
    result: dict,
) -> tuple[list[str], list[dict]]:
    doc = fitz.open(str(pdf_path))
    layout = extract_layout_plan(symbolic_path, len(measures))
    layout_by_measure = {l.measure_index: l for l in layout}
    plan = _system_measure_plan(layout, measures, len(doc))

    pages_out: list[str] = []
    diagnostics: list[dict] = []
    points_by_measure: dict[int, list[dict]] = {}
    for p in result.get("points", []):
        points_by_measure.setdefault(int(p["measure_index"]), []).append(p)

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_ZOOM, PDF_ZOOM), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image)

        page_system_keys = sorted([k for k in plan if k[0] == page_index], key=lambda x: x[1])
        expected = max(1, len(page_system_keys))
        regions = detect_system_regions(image, expected)

        # If no layout break data exists, divide measures across detected systems
        # in order so every PDF page still displays its complete score.
        if not page_system_keys:
            page_system_keys = [(page_index, i) for i in range(len(regions))]

        page_diag = {
            "page_index": page_index,
            "detected_systems": len(regions),
            "planned_systems": len(page_system_keys),
            "systems": [],
        }

        font_label = _font(max(22, int(image.width / 72)), bold=True)
        font_title = _font(max(18, int(image.width / 90)), bold=True)

        for local_system_i, region in enumerate(regions):
            key = page_system_keys[min(local_system_i, len(page_system_keys) - 1)]
            measure_indices = plan.get(key, [])
            if not measure_indices:
                continue

            left, right = region["left"], region["right"]
            staff_top = region["staff_top"]
            strip_bottom = max(20, staff_top - 10)
            strip_top = max(4, strip_bottom - max(48, int(image.width / 34)))

            # Preserve the PDF raster underneath except for the analysis band in
            # the whitespace immediately above this printed system.
            draw.rounded_rectangle(
                [left, strip_top, right, strip_bottom],
                radius=6,
                fill=(242, 242, 242),
                outline=(180, 180, 180),
                width=1,
            )

            weights = _measure_weights(measure_indices, layout_by_measure, measures)
            total_weight = sum(weights)
            cumulative = 0.0
            measure_slot: dict[int, tuple[float, float]] = {}
            for mi, weight in zip(measure_indices, weights):
                measure_slot[mi] = (cumulative, weight)
                cumulative += weight

            label_x = max(4, left - int(image.width * 0.055))
            title_y = strip_top + max(2, (strip_bottom - strip_top - font_title.size) // 2)
            draw.text((label_x, title_y), "L1", fill=(30, 30, 30), font=font_title)

            point_count = 0
            for mi in measure_indices:
                if mi not in measure_slot:
                    continue
                slot_start, slot_width = measure_slot[mi]
                m = measures[mi]
                m_duration = m.end - m.start
                for p in points_by_measure.get(mi, []):
                    t = parse_fraction(p["time_quarter"])
                    frac = 0.0 if m_duration == 0 else float((t - m.start) / m_duration)
                    frac = max(0.0, min(1.0, frac))
                    relative = (slot_start + frac * slot_width) / total_weight if total_weight else 0.0
                    x = int(left + relative * (right - left))
                    label = str(p["label"])
                    bbox = draw.textbbox((0, 0), label, font=font_label)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    y = strip_top + max(1, (strip_bottom - strip_top - th) // 2 - 1)
                    draw.text((x - tw // 2, y), label, fill=(0, 0, 0), font=font_label)
                    point_count += 1

            page_diag["systems"].append(
                {
                    "system_index": local_system_i,
                    "measure_indices": measure_indices,
                    "measure_numbers": [measures[i].number for i in measure_indices],
                    "level1_labels": point_count,
                    "staff_top": staff_top,
                    "analysis_top": strip_top,
                    "analysis_bottom": strip_bottom,
                }
            )

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        pages_out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        diagnostics.append(page_diag)

    doc.close()
    return pages_out, diagnostics


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Tone-Metric Level 1</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;color:#111;background:#fff}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}#chart{border:1px solid #ddd;overflow-x:auto;padding:12px;margin-top:18px}.note{font-size:12px;color:#444;margin-top:8px}table{border-collapse:collapse;margin-top:18px;font-size:13px}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}#score{margin-top:28px}.score-title{font-size:20px;font-weight:700;margin:0 0 8px}.score-page{display:block;max-width:100%;height:auto;margin:18px auto;border:1px solid #ccc;box-shadow:0 2px 10px rgba(0,0,0,.08);background:white}.score-note{font-size:13px;color:#444;margin-bottom:12px}
</style></head><body>
<h1>Tone-Metric Analyzer — Level 1 only</h1>
<div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Opening meter<select id="meter"><option selected>4/4</option><option>2/2</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze Level 1</button></div>
<div id="status">No analysis yet.</div>
<div class="note">Clean Level-1 runtime only. The analytical engine is unchanged; the PDF view is a separate display layer using the exact uploaded PDF pages.</div>
<div id="chart"></div><div id="table"></div><div id="score"></div>
<script>
const $=id=>document.getElementById(id);
function pf(s){s=String(s);if(s.includes('/')){const[a,b]=s.split('/').map(Number);return a/b}return Number(s)}
function render(d){
 const ms=d.measures||[],pts=d.points||[];if(!ms.length)return;
 const W=Math.max(1100,ms.length*72),H=150,L=70,R=20,inner=W-L-R,total=pf(ms[ms.length-1].end_quarter),x=t=>L+inner*(pf(t)/total);
 let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><text x="4" y="28" font-size="16" font-weight="700">Measures</text><text x="14" y="91" font-size="16" font-weight="700">Level 1</text><rect x="${L}" y="68" width="${inner}" height="30" fill="#ececec"/>`;
 for(const m of ms){const xx=x(m.start_quarter);s+=`<line x1="${xx}" y1="42" x2="${xx}" y2="106" stroke="#bbb"/><text x="${xx+4}" y="30" font-size="14" font-weight="700">${m.number}</text>`}
 for(const p of pts)s+=`<text x="${x(p.time_quarter)}" y="90" text-anchor="middle" font-size="16" font-weight="700">${p.label}</text>`;
 s+='</svg>';$('chart').innerHTML=s;
 let h='<table><thead><tr><th>Sequence pos.</th><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Level 1</th><th>New attack?</th></tr></thead><tbody>';
 for(const p of pts)h+=`<tr><td>${p.sequence_position}</td><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;
 h+='</tbody></table>';$('table').innerHTML=h;
 const pages=d.pdf_pages||[];
 if(pages.length){let q='<div class="score-title">Original uploaded PDF + Level 1 analysis</div><div class="score-note">The score image is rendered from the uploaded PDF itself. Each Level-1 strip is positioned immediately above its corresponding printed system.</div>';pages.forEach((src,i)=>q+=`<img class="score-page" src="${src}" alt="Analyzed score page ${i+1}">`);$('score').innerHTML=q}else{$('score').innerHTML=''}
}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Analyzing Level 1…';$('chart').innerHTML='';$('table').innerHTML='';$('score').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Level 1 complete — ${d.measure_count} measures, ${d.points.length} Level-1 positions.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}};
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
        "mode": "level1-only-clean-room-plus-original-pdf-display",
        "legacy_runtime_imports": False,
        "default_opening_meter": "4/4",
        "pdf_overlay": "original-uploaded-pdf-system-aligned",
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

    with tempfile.TemporaryDirectory(prefix="tm-level1-v4-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            symbolic = pdf_to_musicxml(source, work / "audiveris") if suffix == ".pdf" else source
            measures, attacks = parse_score(symbolic, meter)
            result = analyze_level1(measures, attacks)
            result["version"] = APP_VERSION
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "opening_meter": f"{meter[0]}/{meter[1]}",
            }
            if suffix == ".pdf":
                pages, diagnostics = render_original_pdf_with_level1(source, symbolic, measures, result)
                result["pdf_pages"] = pages
                result["pdf_overlay_diagnostics"] = diagnostics
            else:
                result["pdf_pages"] = []
                result["pdf_overlay_diagnostics"] = []
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Level-1 analysis failed: {exc}") from exc
