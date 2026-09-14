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

APP_VERSION = "0.3.0-level1-clean"
MAX_UPLOAD = 80 * 1024 * 1024

# Standalone clean-room app: deliberately imports no prior analyzer code.
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


def parse_meter(value: str | None) -> tuple[int, int]:
    raw = (value or "").strip().lower()
    # The score currently used as the Level-1 regression target is common time.
    # Accept stale clients that still send "auto" as the explicit 4/4 target.
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
                candidates = [n for n in names if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF/")]
                if not candidates:
                    raise ValueError("MXL contains no MusicXML score.")
                rootfile = sorted(candidates)[0]
            return etree.fromstring(zf.read(rootfile))
    return etree.parse(str(path)).getroot()


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
    parts = [n for n in root.iter() if lname(n.tag) == "part" and n.getparent() is root]
    if not parts:
        parts = [n for n in root.iter() if lname(n.tag) == "part"]
    if not parts:
        raise ValueError("No MusicXML part found.")

    measures: list[Measure] = []
    divisions = 1
    inherited = opening_meter
    global_start = Fraction(0)

    for mi, m in enumerate(children(parts[0], "measure")):
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
        measures.append(Measure(mi, m.get("number") or str(mi + 1), global_start, global_start + full, pickup, inherited))
        global_start += full

    attacks: set[Fraction] = set()
    for part in parts:
        divisions = 1
        inherited = opening_meter
        for mi, m in enumerate(children(part, "measure")[:len(measures)]):
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
            m = next(x for x in measures[a:b+1] if x.start <= t < x.end)
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
        "engine_contract": "dissertation-level1-clean-v0.3",
        "only_level_1": True,
        "level": 1,
        "points": points,
        "segments": segments,
        "measures": [{"index":m.index,"number":m.number,"start_quarter":ftxt(m.start),"end_quarter":ftxt(m.end),"meter":f"{m.meter[0]}/{m.meter[1]}"} for m in measures],
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
    cmd = [find_audiveris(), "-batch", "-transcribe", "-save", "-export", "-output", str(out), "--", str(pdf)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    if p.returncode != 0:
        raise RuntimeError(f"Audiveris failed with exit code {p.returncode}.\n\n{(p.stdout or '')[-12000:]}")
    candidates = []
    for pattern in ("*.mxl", "*.musicxml", "*.xml"):
        candidates.extend(out.rglob(pattern))
    candidates = [x for x in candidates if "container.xml" not in str(x)]
    if not candidates:
        raise RuntimeError("Audiveris completed but no MusicXML export was found.")
    candidates.sort(key=lambda x: (0 if x.suffix.lower() == ".mxl" else 1, len(str(x))))
    return candidates[0]


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Tone-Metric Level 1</title><style>
body{font-family:Arial,sans-serif;margin:28px;color:#111}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}#chart{border:1px solid #ddd;overflow-x:auto;padding:12px;margin-top:18px}.note{font-size:12px;color:#444;margin-top:8px}table{border-collapse:collapse;margin-top:18px;font-size:13px}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}
</style></head><body><h1>Tone-Metric Analyzer — Level 1 only</h1><div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Opening meter<select id="meter"><option selected>4/4</option><option>2/2</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze Level 1</button></div><div id="status">No analysis yet.</div><div class="note">Clean Level-1 runtime only. No previous analyzer, higher levels, Waves, Pivots, Trees, or overlay code is loaded.</div><div id="chart"></div><div id="table"></div><script>
const $=id=>document.getElementById(id);function pf(s){s=String(s);if(s.includes('/')){const[a,b]=s.split('/').map(Number);return a/b}return Number(s)}function render(d){const ms=d.measures||[],pts=d.points||[];if(!ms.length)return;const W=Math.max(1100,ms.length*72),H=150,L=70,R=20,inner=W-L-R,total=pf(ms[ms.length-1].end_quarter),x=t=>L+inner*(pf(t)/total);let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><text x="4" y="28" font-size="16" font-weight="700">Measures</text><text x="14" y="91" font-size="16" font-weight="700">Level 1</text><rect x="${L}" y="68" width="${inner}" height="30" fill="#ececec"/>`;for(const m of ms){const xx=x(m.start_quarter);s+=`<line x1="${xx}" y1="42" x2="${xx}" y2="106" stroke="#bbb"/><text x="${xx+4}" y="30" font-size="14" font-weight="700">${m.number}</text>`}for(const p of pts)s+=`<text x="${x(p.time_quarter)}" y="90" text-anchor="middle" font-size="16" font-weight="700">${p.label}</text>`;s+='</svg>';$('chart').innerHTML=s;let h='<table><thead><tr><th>Sequence pos.</th><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Level 1</th><th>New attack?</th></tr></thead><tbody>';for(const p of pts)h+=`<tr><td>${p.sequence_position}</td><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;h+='</tbody></table>';$('table').innerHTML=h}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Analyzing Level 1…';$('chart').innerHTML='';$('table').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Level 1 complete — ${d.measure_count} measures, ${d.points.length} Level-1 positions.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}};
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
    return {"ok":True,"version":APP_VERSION,"mode":"level1-only-clean-room","legacy_runtime_imports":False,"default_opening_meter":"4/4"}


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
    with tempfile.TemporaryDirectory(prefix="tm-level1-v3-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            symbolic = pdf_to_musicxml(source, work / "audiveris") if suffix == ".pdf" else source
            measures, attacks = parse_score(symbolic, meter)
            result = analyze_level1(measures, attacks)
            result["version"] = APP_VERSION
            result["source"] = {"filename":filename,"input_type":suffix.lstrip("."),"opening_meter":f"{meter[0]}/{meter[1]}"}
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Level-1 analysis failed: {exc}") from exc
