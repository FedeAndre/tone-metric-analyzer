from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from lxml import etree

APP_VERSION = "0.2.0-level1-clean"
MAX_UPLOAD = 80 * 1024 * 1024

app = FastAPI(title="Tone-Metric Level 1", version=APP_VERSION)

# CLEAN-ROOM RULE:
# This runtime imports no tone_metric package, no dissertation_* module,
# no legacy app, no overlay code, and no Waves/Pivots/Trees implementation.


@dataclass(frozen=True)
class MeterProfile:
    numerator: int
    denominator: int
    beat_unit: Fraction
    top_arity: int
    label: str


METER_PROFILES: dict[tuple[int, int], MeterProfile] = {
    (2, 2): MeterProfile(2, 2, Fraction(2), 2, "pure-binary"),
    (4, 4): MeterProfile(4, 4, Fraction(1), 2, "pure-binary"),
    (3, 4): MeterProfile(3, 4, Fraction(1), 3, "mixed-ternary-binary"),
    (6, 8): MeterProfile(6, 8, Fraction(3, 2), 2, "mixed-binary-ternary"),
    (9, 8): MeterProfile(9, 8, Fraction(3, 2), 3, "mixed-ternary"),
    (12, 8): MeterProfile(12, 8, Fraction(3, 2), 2, "mixed-binary-ternary"),
}


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class AttackRow:
    onset: Fraction
    measure_index: int
    measure_number: str


def frac_text(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def local_name(tag) -> str:
    # lxml comments / processing instructions expose a callable .tag object,
    # not a string. They are ignored rather than passed to string methods.
    if not isinstance(tag, str):
        return ""
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


def parse_meter(value: str | None) -> tuple[int, int] | None:
    text = (value or "").strip().lower()
    if not text or text == "auto":
        return None
    if "/" not in text:
        raise ValueError("Meter must be written as numerator/denominator, for example 4/4.")
    left, right = text.split("/", 1)
    meter = (int(left), int(right))
    if meter not in METER_PROFILES:
        raise ValueError(f"Meter {meter[0]}/{meter[1]} is not in the dissertation-validated Level-1 profiles.")
    return meter


def geometric_sequence(base: int, limit: int) -> list[int]:
    """Dissertation Level-1 sequence positions, 1-based."""
    if base not in (2, 3):
        raise ValueError("Level 1 supports only dissertation binary or ternary sequence arity.")
    if limit < 1:
        return []
    out = [1]
    if limit == 1:
        return out
    out.append(2)
    while out[-1] < limit:
        out.append(base * out[-1] - (base - 1))
    return out


def xml_root(path: Path):
    if path.suffix.lower() == ".mxl":
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


def time_signature(attributes) -> tuple[int, int] | None:
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
        raise ValueError("Additive meters are outside this clean Level-1 build.")
    return int(beats), int(beat_type)


def scan_measure(measure_el, divisions: int) -> tuple[Fraction, list[tuple[Fraction, bool]]]:
    """Return actual notated span and note attack rows in quarter-note units.

    Rules used only to decide whether a Level-1 structural position has a new attack:
    grace notes are ignored; rests are not attacks; tied continuations are not attacks;
    simultaneous chord notes share one onset; backup/forward preserve polyphonic time.
    """
    cursor = Fraction(0)
    max_cursor = Fraction(0)
    previous_note_onset = Fraction(0)
    rows: list[tuple[Fraction, bool]] = []

    for item in measure_el:
        tag = local_name(item.tag)
        if not tag:
            continue

        if tag == "note":
            if child(item, "grace") is not None:
                continue

            chord_member = child(item, "chord") is not None
            raw_duration = child_text(item, "duration", "0") or "0"
            duration = Fraction(int(raw_duration), max(1, divisions))
            onset = previous_note_onset if chord_member else cursor
            previous_note_onset = onset

            if not chord_member:
                cursor += duration
            max_cursor = max(max_cursor, onset + duration, cursor)

            is_rest = child(item, "rest") is not None
            tied_from_previous = False
            for tie in children(item, "tie"):
                if (tie.get("type") or "").lower() == "stop":
                    tied_from_previous = True
            notations = child(item, "notations")
            if notations is not None:
                for node in notations.iter():
                    if local_name(node.tag) == "tied" and (node.get("type") or "").lower() == "stop":
                        tied_from_previous = True

            rows.append((onset, (not is_rest) and (not tied_from_previous)))

        elif tag == "backup":
            raw = child_text(item, "duration", "0") or "0"
            cursor -= Fraction(int(raw), max(1, divisions))
            if cursor < 0:
                cursor = Fraction(0)

        elif tag == "forward":
            raw = child_text(item, "duration", "0") or "0"
            cursor += Fraction(int(raw), max(1, divisions))
            max_cursor = max(max_cursor, cursor)

    return max_cursor, rows


def parse_musicxml(path: Path, meter_override: tuple[int, int] | None) -> tuple[list[MeasureRow], list[AttackRow]]:
    root = xml_root(path)
    parts = [el for el in root.iter() if local_name(el.tag) == "part" and el.getparent() is root]
    if not parts:
        parts = [el for el in root.iter() if local_name(el.tag) == "part"]
    if not parts:
        raise ValueError("No MusicXML part was found.")

    # One shared measure timeline from the first part.
    measures: list[MeasureRow] = []
    divisions = 1
    inherited_meter = meter_override
    global_start = Fraction(0)
    first_part_measures = children(parts[0], "measure")

    for mi, measure_el in enumerate(first_part_measures):
        attrs = child(measure_el, "attributes")
        if attrs is not None:
            div_text = child_text(attrs, "divisions")
            if div_text:
                divisions = max(1, int(div_text))
            explicit_meter = time_signature(attrs)
            if explicit_meter is not None:
                inherited_meter = explicit_meter

        if inherited_meter is None:
            number = measure_el.get("number") or str(mi + 1)
            raise ValueError(
                f"Measure {number}: no explicit/inherited meter is available. Choose the opening meter explicitly."
            )

        num, den = inherited_meter
        if (num, den) not in METER_PROFILES:
            raise ValueError(f"Meter {num}/{den} is not supported by this Level-1 build.")

        full = Fraction(num * 4, den)
        actual, _ = scan_measure(measure_el, divisions)
        if actual <= 0:
            actual = full

        # An incomplete first measure is positioned at the END of its full metric
        # container, matching the project's pickup rule.
        pickup_shift = full - actual if mi == 0 and actual < full else Fraction(0)
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

    # Global attacks from all voices/parts, deduplicated by musical onset.
    attacks: dict[Fraction, AttackRow] = {}
    for part in parts:
        divisions = 1
        inherited_meter = meter_override
        pmeasures = children(part, "measure")
        for mi, measure_el in enumerate(pmeasures[: len(measures)]):
            attrs = child(measure_el, "attributes")
            if attrs is not None:
                div_text = child_text(attrs, "divisions")
                if div_text:
                    divisions = max(1, int(div_text))
                explicit_meter = time_signature(attrs)
                if explicit_meter is not None:
                    inherited_meter = explicit_meter

            _actual, note_rows = scan_measure(measure_el, divisions)
            m = measures[mi]
            for local_onset, is_attack in note_rows:
                if not is_attack:
                    continue
                onset = m.start + m.pickup_shift + local_onset
                if m.start <= onset < m.end:
                    attacks.setdefault(onset, AttackRow(onset, m.index, m.number))

    return measures, [attacks[t] for t in sorted(attacks)]


def meter_segments(measures: list[MeasureRow]) -> list[tuple[int, int]]:
    if not measures:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    current = (measures[0].numerator, measures[0].denominator)
    for i in range(1, len(measures) + 1):
        changed = i == len(measures) or (measures[i].numerator, measures[i].denominator) != current
        if changed:
            out.append((start, i - 1))
            if i < len(measures):
                start = i
                current = (measures[i].numerator, measures[i].denominator)
    return out


def analyze_level1(measures: list[MeasureRow], attacks: list[AttackRow]) -> dict:
    attack_times = {a.onset for a in attacks}
    points: list[dict] = []
    segments: list[dict] = []

    for segment_index, (start_i, end_i) in enumerate(meter_segments(measures)):
        first = measures[start_i]
        last = measures[end_i]
        profile = METER_PROFILES[(first.numerator, first.denominator)]
        segment_start = first.start
        segment_end = last.end
        duration = segment_end - segment_start
        if duration % profile.beat_unit != 0:
            raise ValueError(f"Meter segment {segment_index + 1} is not an exact multiple of its tactus.")

        interval_count = int(duration / profile.beat_unit)
        max_position = interval_count + 1
        sequence = geometric_sequence(profile.top_arity, max_position)
        segment_points = 0

        for seq_position in sequence:
            t = segment_start + (seq_position - 1) * profile.beat_unit
            if not (segment_start <= t < segment_end):
                continue
            measure = next(
                (m for m in measures[start_i : end_i + 1] if m.start <= t < m.end),
                None,
            )
            if measure is None:
                continue
            beat = int((t - measure.start) / profile.beat_unit) + 1
            is_attack = t in attack_times
            points.append(
                {
                    "segment_index": segment_index,
                    "sequence_position": seq_position,
                    "time_quarter": frac_text(t),
                    "measure_index": measure.index,
                    "measure_number": measure.number,
                    "beat": beat,
                    "level": 1,
                    "attack": is_attack,
                    "parenthetical": not is_attack,
                    "label": "1" if is_attack else "(1)",
                }
            )
            segment_points += 1

        segments.append(
            {
                "segment_index": segment_index,
                "start_measure": first.number,
                "end_measure": last.number,
                "meter": f"{profile.numerator}/{profile.denominator}",
                "meter_profile": profile.label,
                "beat_unit_quarter": frac_text(profile.beat_unit),
                "top_sequence_arity": profile.top_arity,
                "sequence": sequence,
                "level1_point_count": segment_points,
            }
        )

    return {
        "engine_contract": "dissertation-level1-clean-v0.2",
        "only_level_1": True,
        "level": 1,
        "sequence_rules": {
            "binary": "1,2,3,5,9,17,33,... (y=2x-1)",
            "ternary": "1,2,4,10,28,82,244,... (y=3x-2)",
            "attack_independent_structure": True,
            "parentheses_mean_structural_position_without_new_attack": True,
        },
        "segments": segments,
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
        "measure_count": len(measures),
        "attack_count": len(attack_times),
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


def pdf_to_musicxml(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_audiveris(),
        "-batch",
        "-transcribe",
        "-save",
        "-export",
        "-output",
        str(output_dir),
        "--",
        str(pdf_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=240,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Audiveris failed with exit code {proc.returncode}.\n\n{(proc.stdout or '')[-12000:]}"
        )

    candidates: list[Path] = []
    for pattern in ("*.mxl", "*.musicxml", "*.xml"):
        candidates.extend(output_dir.rglob(pattern))
    candidates = [p for p in candidates if "container.xml" not in str(p)]
    if not candidates:
        raise RuntimeError("Audiveris completed but no MusicXML export was found.")
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".mxl" else 1, len(str(p))))
    return candidates[0]


HTML = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tone-Metric Level 1</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;background:#fff;color:#111}
.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}
label{display:flex;flex-direction:column;gap:5px;font-size:13px}
button{padding:8px 16px}
#status{margin:14px 0;font-weight:600}
#chart{border:1px solid #ddd;overflow-x:auto;padding:12px;margin-top:18px}
.note{font-size:12px;color:#444;margin-top:8px}
table{border-collapse:collapse;margin-top:18px;font-size:13px}
td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}
</style>
</head>
<body>
<h1>Tone-Metric Analyzer — Level 1 only</h1>
<div class="row">
<label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label>
<label>Opening meter
<select id="meter">
<option value="auto">Auto</option><option>2/2</option><option>4/4</option><option>3/4</option>
<option>6/8</option><option>9/8</option><option>12/8</option>
</select></label>
<button id="go">Analyze Level 1</button>
</div>
<div id="status">No analysis yet.</div>
<div class="note">Clean Level-1 runtime only. No previous Levels engine, Waves, Pivots, Trees, overlay, or legacy wrapper is loaded.</div>
<div id="chart"></div><div id="table"></div>
<script>
const $=id=>document.getElementById(id);
function pf(s){s=String(s);if(s.includes('/')){const [a,b]=s.split('/').map(Number);return a/b}return Number(s)}
function render(d){
 const ms=d.measures||[],pts=d.points||[]; if(!ms.length)return;
 const W=Math.max(1100,ms.length*72),H=150,L=70,R=20,inner=W-L-R,total=pf(ms[ms.length-1].end_quarter);
 const x=t=>L+inner*(pf(t)/total);
 let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><text x="4" y="28" font-size="16" font-weight="700">Measures</text><text x="14" y="91" font-size="16" font-weight="700">Level 1</text><rect x="${L}" y="68" width="${inner}" height="30" fill="#ececec"/>`;
 for(const m of ms){const xx=x(m.start_quarter);s+=`<line x1="${xx}" y1="42" x2="${xx}" y2="106" stroke="#bbb"/><text x="${xx+4}" y="30" font-size="14" font-weight="700">${m.number}</text>`}
 for(const p of pts){s+=`<text x="${x(p.time_quarter)}" y="90" text-anchor="middle" font-size="16" font-weight="700">${p.label}</text>`}
 s+='</svg>';$('chart').innerHTML=s;
 let h='<table><thead><tr><th>Sequence pos.</th><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Level 1</th><th>New attack?</th></tr></thead><tbody>';
 for(const p of pts)h+=`<tr><td>${p.sequence_position}</td><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;
 h+='</tbody></table>';$('table').innerHTML=h;
}
$('go').onclick=async()=>{
 const f=$('file').files[0]; if(!f){$('status').textContent='Choose a score.';return}
 const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);
 $('status').textContent='Analyzing Level 1…';$('chart').innerHTML='';$('table').innerHTML='';
 try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const text=await r.text();if(!r.ok)throw new Error(text);const d=JSON.parse(text);$('status').textContent=`Level 1 complete — ${d.measure_count} measures, ${d.points.length} Level-1 structural positions.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}
};
</script>
</body></html>'''


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
        "mode": "level1-only-clean-room",
        "legacy_runtime_imports": False,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), initial_meter: str = Form("auto")):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".mxl", ".musicxml", ".xml"}:
        raise HTTPException(400, "Upload PDF, MXL, MusicXML, or XML.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Upload exceeds the 80 MB limit.")

    try:
        meter_override = parse_meter(initial_meter)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="tm-level1-clean-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            symbolic = pdf_to_musicxml(source, work / "audiveris") if suffix == ".pdf" else source
            measures, attacks = parse_musicxml(symbolic, meter_override)
            result = analyze_level1(measures, attacks)
            result["version"] = APP_VERSION
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "meter_override": initial_meter,
            }
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Level-1 analysis failed: {exc}") from exc
