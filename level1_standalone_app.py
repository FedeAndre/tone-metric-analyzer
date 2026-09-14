from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from lxml import etree

APP_VERSION = "0.1.0-level1-clean"
MAX_UPLOAD = 80 * 1024 * 1024

app = FastAPI(title="Tone-Metric Level 1", version=APP_VERSION)


# This file is intentionally standalone. It imports no previous tone_metric,
# dissertation_*, overlay, wave, pivot, tree, or legacy analyzer modules.


@dataclass(frozen=True)
class MeterProfile:
    numerator: int
    denominator: int
    beat_unit: Fraction
    top_arity: int
    name: str


METER_PROFILES: dict[tuple[int, int], MeterProfile] = {
    (2, 2): MeterProfile(2, 2, Fraction(2), 2, "pure-binary"),
    (4, 4): MeterProfile(4, 4, Fraction(1), 2, "pure-binary"),
    (3, 4): MeterProfile(3, 4, Fraction(1), 3, "mixed-ternary-binary"),
    (6, 8): MeterProfile(6, 8, Fraction(3, 2), 2, "mixed-binary-ternary"),
    (9, 8): MeterProfile(9, 8, Fraction(3, 2), 3, "mixed-ternary"),
    (12, 8): MeterProfile(12, 8, Fraction(3, 2), 2, "mixed-binary-ternary"),
}


@dataclass
class MeasureRow:
    index: int
    number: str
    start: Fraction
    end: Fraction
    actual_duration: Fraction
    full_duration: Fraction
    pickup_shift: Fraction
    numerator: int
    denominator: int


@dataclass
class AttackRow:
    onset: Fraction
    measure_index: int
    measure_number: str


def frac_text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child(el, name: str):
    for c in el:
        if local_name(c.tag) == name:
            return c
    return None


def children(el, name: str):
    return [c for c in el if local_name(c.tag) == name]


def child_text(el, name: str, default: str | None = None) -> str | None:
    c = child(el, name)
    if c is None or c.text is None:
        return default
    return c.text.strip()


def parse_meter_text(value: str | None) -> tuple[int, int] | None:
    text = (value or "").strip().lower()
    if not text or text == "auto":
        return None
    if "/" not in text:
        raise ValueError("Meter must be written as numerator/denominator, for example 4/4.")
    a, b = text.split("/", 1)
    num, den = int(a), int(b)
    if (num, den) not in METER_PROFILES:
        raise ValueError(f"Meter {num}/{den} is not in the dissertation-validated Level-1 profiles.")
    return num, den


def geometric_sequence(base: int, limit: int) -> list[int]:
    if base not in (2, 3):
        raise ValueError("Level 1 uses only the dissertation binary or ternary sequence.")
    if limit < 1:
        return []
    out = [1]
    if limit == 1:
        return out
    out.append(2)
    while out[-1] < limit:
        out.append(base * out[-1] - (base - 1))
    return out


def _xml_root(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".mxl":
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            rootfile = None
            if "META-INF/container.xml" in names:
                container = etree.fromstring(zf.read("META-INF/container.xml"))
                for el in container.iter():
                    if local_name(el.tag) == "rootfile":
                        rootfile = el.get("full-path")
                        if rootfile:
                            break
            if not rootfile:
                candidates = [n for n in names if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF/")]
                if not candidates:
                    raise ValueError("MXL contains no MusicXML score.")
                rootfile = sorted(candidates)[0]
            return etree.fromstring(zf.read(rootfile))
    return etree.parse(str(path)).getroot()


def _time_signature(attributes) -> tuple[int, int] | None:
    if attributes is None:
        return None
    time_el = child(attributes, "time")
    if time_el is None:
        return None
    beats = child_text(time_el, "beats")
    beat_type = child_text(time_el, "beat-type")
    if not beats or not beat_type:
        return None
    if "+" in beats:
        raise ValueError("Additive meters are not part of this clean Level-1 build.")
    return int(beats), int(beat_type)


def _measure_scan(measure_el, divisions: int) -> tuple[Fraction, list[tuple[Fraction, bool]]]:
    cursor = Fraction(0)
    max_cursor = Fraction(0)
    previous_note_onset = Fraction(0)
    notes: list[tuple[Fraction, bool]] = []

    for item in measure_el:
        tag = local_name(item.tag)
        if tag == "note":
            if child(item, "grace") is not None:
                continue
            chord = child(item, "chord") is not None
            duration_text = child_text(item, "duration", "0") or "0"
            duration = Fraction(int(duration_text), max(1, divisions))
            onset = previous_note_onset if chord else cursor
            previous_note_onset = onset
            if not chord:
                cursor += duration
            max_cursor = max(max_cursor, onset + duration, cursor)

            is_rest = child(item, "rest") is not None
            tie_stop = False
            for tie in children(item, "tie"):
                if (tie.get("type") or "").lower() == "stop":
                    tie_stop = True
            notations = child(item, "notations")
            if notations is not None:
                for tied in notations.iter():
                    if local_name(tied.tag) == "tied" and (tied.get("type") or "").lower() == "stop":
                        tie_stop = True
            attack = (not is_rest) and (not tie_stop)
            notes.append((onset, attack))

        elif tag == "backup":
            d = int(child_text(item, "duration", "0") or "0")
            cursor -= Fraction(d, max(1, divisions))
            if cursor < 0:
                cursor = Fraction(0)
        elif tag == "forward":
            d = int(child_text(item, "duration", "0") or "0")
            cursor += Fraction(d, max(1, divisions))
            max_cursor = max(max_cursor, cursor)

    return max_cursor, notes


def parse_musicxml_level1(path: Path, initial_meter: tuple[int, int] | None = None) -> tuple[list[MeasureRow], list[AttackRow], list[str]]:
    root = _xml_root(path)
    parts = [el for el in root.iter() if local_name(el.tag) == "part" and el.getparent() is root]
    if not parts:
        parts = [el for el in root.iter() if local_name(el.tag) == "part"]
    if not parts:
        raise ValueError("No MusicXML part was found.")

    warnings: list[str] = []
    measures: list[MeasureRow] = []

    # First pass: establish one global measure timeline from the first part only.
    divisions = 1
    inherited_meter = initial_meter
    global_start = Fraction(0)
    first_part_measures = children(parts[0], "measure")
    for mi, measure_el in enumerate(first_part_measures):
        attributes = child(measure_el, "attributes")
        if attributes is not None:
            div_text = child_text(attributes, "divisions")
            if div_text:
                divisions = max(1, int(div_text))
            explicit_meter = _time_signature(attributes)
            if explicit_meter is not None:
                inherited_meter = explicit_meter
        if inherited_meter is None:
            raise ValueError(
                f"Measure {measure_el.get('number') or mi + 1}: no explicit/inherited meter is available. "
                "Choose the opening meter explicitly."
            )
        num, den = inherited_meter
        if (num, den) not in METER_PROFILES:
            raise ValueError(f"Meter {num}/{den} is not supported by this dissertation Level-1 build.")
        full = Fraction(num * 4, den)
        actual, _ = _measure_scan(measure_el, divisions)
        if actual <= 0:
            actual = full
        pickup_shift = Fraction(0)
        if mi == 0 and actual < full:
            pickup_shift = full - actual
        number = measure_el.get("number") or str(mi + 1)
        measures.append(
            MeasureRow(
                index=mi,
                number=number,
                start=global_start,
                end=global_start + full,
                actual_duration=actual,
                full_duration=full,
                pickup_shift=pickup_shift,
                numerator=num,
                denominator=den,
            )
        )
        global_start += full

    # Second pass: collect sounding attacks from every part onto that timeline.
    attack_times: set[tuple[Fraction, int, str]] = set()
    for part in parts:
        divisions = 1
        inherited_meter = initial_meter
        part_measures = children(part, "measure")
        for mi, measure_el in enumerate(part_measures[: len(measures)]):
            attributes = child(measure_el, "attributes")
            if attributes is not None:
                div_text = child_text(attributes, "divisions")
                if div_text:
                    divisions = max(1, int(div_text))
                explicit_meter = _time_signature(attributes)
                if explicit_meter is not None:
                    inherited_meter = explicit_meter
            actual, note_rows = _measure_scan(measure_el, divisions)
            m = measures[mi]
            shift = m.pickup_shift
            for local_onset, attack in note_rows:
                if not attack:
                    continue
                onset = m.start + shift + local_onset
                if onset >= m.end:
                    continue
                attack_times.add((onset, m.index, m.number))

    attacks = [AttackRow(*row) for row in sorted(attack_times, key=lambda r: (r[0], r[1], r[2]))]
    return measures, attacks, warnings


def build_meter_segments(measures: list[MeasureRow]) -> list[tuple[int, int]]:
    if not measures:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    cur = (measures[0].numerator, measures[0].denominator)
    for i in range(1, len(measures) + 1):
        changed = i == len(measures) or (measures[i].numerator, measures[i].denominator) != cur
        if changed:
            out.append((start, i - 1))
            if i < len(measures):
                start = i
                cur = (measures[i].numerator, measures[i].denominator)
    return out


def level1_analysis(measures: list[MeasureRow], attacks: list[AttackRow]) -> dict:
    attack_times = {a.onset for a in attacks}
    points: list[dict] = []
    segment_rows: list[dict] = []

    for seg_index, (a, b) in enumerate(build_meter_segments(measures)):
        first = measures[a]
        last = measures[b]
        meter = (first.numerator, first.denominator)
        profile = METER_PROFILES[meter]
        start = first.start
        end = last.end
        duration = end - start
        if duration % profile.beat_unit != 0:
            raise ValueError(f"Meter segment {seg_index + 1} is not an exact multiple of its tactus.")
        intervals = int(duration / profile.beat_unit)
        max_pos = intervals + 1
        seq = geometric_sequence(profile.top_arity, max_pos)

        seg_points = []
        for n in seq:
            t = start + (n - 1) * profile.beat_unit
            if not (start <= t < end):
                continue
            measure = next((m for m in measures[a : b + 1] if m.start <= t < m.end), None)
            if measure is None:
                continue
            beat_index = int((t - measure.start) / profile.beat_unit) + 1
            attack = t in attack_times
            row = {
                "segment_index": seg_index,
                "time_quarter": frac_text(t),
                "measure_index": measure.index,
                "measure_number": measure.number,
                "beat": beat_index,
                "level": 1,
                "attack": attack,
                "parenthetical": not attack,
                "label": "1" if attack else "(1)",
            }
            points.append(row)
            seg_points.append(row)

        segment_rows.append(
            {
                "segment_index": seg_index,
                "start_measure": first.number,
                "end_measure": last.number,
                "meter": f"{profile.numerator}/{profile.denominator}",
                "meter_profile": profile.name,
                "beat_unit_quarter": frac_text(profile.beat_unit),
                "top_sequence_arity": profile.top_arity,
                "sequence": seq,
                "level1_point_count": len(seg_points),
            }
        )

    return {
        "engine_contract": "dissertation-level1-clean-v0.1",
        "level": 1,
        "only_level_1": True,
        "sequence_rules": {
            "binary": "1,2,3,5,9,17,33,... (y=2x-1)",
            "ternary": "1,2,4,10,28,82,244,... (y=3x-2)",
            "attack_independent_structure": True,
            "parentheses_mean_no_new_attack": True,
        },
        "segments": segment_rows,
        "points": points,
        "measures": [
            {
                "index": m.index,
                "number": m.number,
                "start_quarter": frac_text(m.start),
                "end_quarter": frac_text(m.end),
                "meter": f"{m.numerator}/{m.denominator}",
                "pickup_shift_quarter": frac_text(m.pickup_shift),
            }
            for m in measures
        ],
        "attack_count": len({a.onset for a in attacks}),
        "measure_count": len(measures),
    }


def find_audiveris() -> str:
    configured = os.environ.get("AUDIVERIS_CMD", "").strip()
    if configured and Path(configured).exists():
        return configured
    for candidate in ("Audiveris", "audiveris", "/opt/audiveris/bin/Audiveris"):
        found = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if found and Path(found).exists():
            return str(found)
    raise RuntimeError("Audiveris is not available.")


def pdf_to_musicxml(pdf_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [find_audiveris(), "-batch", "-transcribe", "-save", "-export", "-output", str(out_dir), "--", str(pdf_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-12000:]
        raise RuntimeError(f"Audiveris failed with exit code {proc.returncode}.\n\n{tail}")
    candidates = []
    for pattern in ("*.mxl", "*.musicxml", "*.xml"):
        candidates.extend(out_dir.rglob(pattern))
    candidates = [p for p in candidates if "container.xml" not in str(p)]
    if not candidates:
        raise RuntimeError("Audiveris completed but no MusicXML export was found.")
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".mxl" else 1, len(str(p))))
    return candidates[0]


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Tone-Metric Level 1</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;background:#fff;color:#111} .row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}
label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}
#chart{border:1px solid #ddd;overflow-x:auto;padding:12px;margin-top:18px}.note{font-size:12px;color:#444;margin-top:8px}
table{border-collapse:collapse;margin-top:18px;font-size:13px}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}
</style></head><body>
<h1>Tone-Metric Analyzer — Level 1 only</h1>
<div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label>
<label>Opening meter<select id="meter"><option value="auto">Auto</option><option>2/2</option><option>4/4</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label>
<button id="go">Analyze Level 1</button></div>
<div id="status">No analysis yet.</div><div class="note">Only Level 1 is computed. No previous Levels engine, Waves, Pivots, Trees, score-overlay, or legacy wrapper is loaded.</div>
<div id="chart"></div><div id="table"></div>
<script>
const $=id=>document.getElementById(id);
function render(data){
  const measures=data.measures||[], pts=data.points||[]; if(!measures.length){$('chart').innerHTML='';return}
  const W=Math.max(1100,measures.length*70), H=150, left=55, top=48, inner=W-left-20;
  const total=parseFrac(measures[measures.length-1].end_quarter);
  const x=t=>left+inner*(parseFrac(t)/total);
  let svg=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><text x="6" y="28" font-size="16" font-weight="700">Measures</text><text x="12" y="92" font-size="16" font-weight="700">Level 1</text>`;
  svg+=`<rect x="${left}" y="68" width="${inner}" height="30" fill="#ececec"/>`;
  for(const m of measures){const mx=x(m.start_quarter);svg+=`<line x1="${mx}" y1="42" x2="${mx}" y2="105" stroke="#bbb"/><text x="${mx+4}" y="30" font-size="14" font-weight="700">${m.number}</text>`}
  for(const p of pts){const px=x(p.time_quarter);svg+=`<text x="${px}" y="90" text-anchor="middle" font-size="16" font-weight="700">${p.label}</text>`}
  svg+='</svg>';$('chart').innerHTML=svg;
  let h='<table><thead><tr><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Level 1</th><th>Attack?</th></tr></thead><tbody>';
  for(const p of pts)h+=`<tr><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;
  h+='</tbody></table>';$('table').innerHTML=h;
}
function parseFrac(s){if(String(s).includes('/')){const [a,b]=String(s).split('/').map(Number);return a/b}return Number(s)}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Analyzing Level 1…';$('chart').innerHTML='';$('table').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Level 1 complete — ${d.measure_count} measures, ${d.points.length} Level-1 structural positions.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}}
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "mode": "level1-only-clean-room",
        "legacy_runtime_imports": False,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), initial_meter: str = Form("auto")):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".mxl", ".musicxml", ".xml"}:
        raise HTTPException(400, "Upload PDF, MXL, MusicXML, or XML.")
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, "Upload exceeds the 80 MB limit.")
    try:
        meter = parse_meter_text(initial_meter)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="tm-level1-") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / filename
        source.write_bytes(payload)
        try:
            symbolic = pdf_to_musicxml(source, tmpdir / "audiveris") if suffix == ".pdf" else source
            measures, attacks, warnings = parse_musicxml_level1(symbolic, meter)
            result = level1_analysis(measures, attacks)
            result["source"] = {"filename": filename, "input_type": suffix.lstrip("."), "meter_override": initial_meter}
            result["warnings"] = warnings
            result["version"] = APP_VERSION
            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Level-1 analysis failed: {exc}") from exc
