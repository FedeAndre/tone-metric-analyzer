from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

VERSION = "homr-independent-reader-probe-v1"
SESSIONS = Path(tempfile.gettempdir()) / "homr_reader_probe"
SESSIONS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Independent HOMR score reader probe")

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Independent HOMR reader probe</title>
<style>body{font-family:Arial,sans-serif;margin:0;background:#f4f4f4;color:#111}main{max-width:900px;margin:auto;padding:24px}
form{display:flex;gap:10px;align-items:center;margin:18px 0}button{padding:8px 14px}pre{white-space:pre-wrap;background:white;padding:14px;border:1px solid #ddd}</style>
</head><body><main><h1>Independent HOMR reader probe</h1>
<p>This service is isolated from the Tone-Metric/Audiveris hit engine. It only reads notation with HOMR and reports MusicXML-derived attack timing.</p>
<form id="f"><input id="file" type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" required><button>Read score</button></form>
<pre id="out"></pre>
<script>
const f=document.getElementById('f'),o=document.getElementById('out');
f.addEventListener('submit',async e=>{e.preventDefault();o.textContent='Reading…';const fd=new FormData();fd.append('file',document.getElementById('file').files[0]);
try{const r=await fetch('/api/read',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Read failed');o.textContent=JSON.stringify(d,null,2);}
catch(err){o.textContent=err.message;}});</script></main></body></html>"""


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _tag(c) == name:
            return c
    return None


def _child_text(el: ET.Element, name: str, default: str = "") -> str:
    c = _child(el, name)
    return (c.text or default).strip() if c is not None else default


def analyze_musicxml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    parts = [e for e in root.iter() if _tag(e) == "part" and _tag(e.getparent()) == "x"] if False else [
        e for e in root if _tag(e) == "part"
    ]
    total_notes = total_rests = total_graces = total_chord_notes = total_tie_stops = 0
    type_counts: dict[str, int] = {}
    all_events: list[dict] = []
    global_events: set[tuple[int, Fraction]] = set()
    max_measures = 0

    for part_index, part in enumerate(parts):
        divisions = 1
        measures = [m for m in part if _tag(m) == "measure"]
        max_measures = max(max_measures, len(measures))
        for measure_index, measure in enumerate(measures, start=1):
            cursor = Fraction(0)
            previous_note_start = Fraction(0)
            for item in measure:
                kind = _tag(item)
                if kind == "attributes":
                    div_text = _child_text(item, "divisions")
                    if div_text:
                        divisions = max(1, int(div_text))
                    continue
                if kind == "backup":
                    d = int(_child_text(item, "duration", "0") or 0)
                    cursor -= Fraction(d, divisions)
                    continue
                if kind == "forward":
                    d = int(_child_text(item, "duration", "0") or 0)
                    cursor += Fraction(d, divisions)
                    continue
                if kind != "note":
                    continue

                total_notes += 1
                is_chord = _child(item, "chord") is not None
                is_rest = _child(item, "rest") is not None
                is_grace = _child(item, "grace") is not None
                if is_chord:
                    total_chord_notes += 1
                if is_rest:
                    total_rests += 1
                if is_grace:
                    total_graces += 1

                note_type = _child_text(item, "type")
                if note_type:
                    type_counts[note_type] = type_counts.get(note_type, 0) + 1

                duration_raw = int(_child_text(item, "duration", "0") or 0)
                duration = Fraction(duration_raw, divisions) if duration_raw else Fraction(0)
                start = previous_note_start if is_chord else cursor
                if not is_chord:
                    previous_note_start = start

                tie_types = {
                    (c.attrib.get("type") or "").strip().lower()
                    for c in item
                    if _tag(c) == "tie"
                }
                is_tie_continuation = "stop" in tie_types
                if is_tie_continuation:
                    total_tie_stops += 1

                if not is_rest and not is_grace and not is_tie_continuation:
                    key = (measure_index, start)
                    global_events.add(key)
                    all_events.append({
                        "part": part_index + 1,
                        "measure": measure_index,
                        "offset_quarter_units": str(start),
                        "voice": _child_text(item, "voice", "1"),
                        "type": note_type or None,
                        "chord_note": is_chord,
                    })

                if not is_chord and not is_grace:
                    cursor += duration

    events_sorted = sorted(global_events, key=lambda x: (x[0], x[1]))
    return {
        "part_count": len(parts),
        "measure_count": max_measures,
        "note_elements": total_notes,
        "rests": total_rests,
        "grace_notes": total_graces,
        "chord_continuation_notes": total_chord_notes,
        "tie_continuations_excluded": total_tie_stops,
        "note_type_counts": dict(sorted(type_counts.items())),
        "global_attack_count": len(events_sorted),
        "global_attacks": [
            {"measure": m, "offset_quarter_units": str(o)} for m, o in events_sorted
        ],
    }


def run_homr(input_path: Path, cwd: Path) -> Path:
    before = set(cwd.glob("*.musicxml"))
    cmd = ["homr", str(input_path)]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
        env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-120:])
        raise RuntimeError(f"HOMR failed with exit code {proc.returncode}.\n{tail}")
    candidates = [p for p in cwd.glob("*.musicxml") if p not in before]
    if not candidates:
        candidates = list(cwd.glob("*.musicxml"))
    if not candidates:
        raise RuntimeError("HOMR completed but produced no MusicXML file.")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return HTML


@app.get("/api/status")
def status() -> dict:
    return {
        "ok": True,
        "version": VERSION,
        "reader": "HOMR",
        "audiveris_present": shutil.which("Audiveris") is not None,
        "audiveris_used": False,
        "tma_code_present": False,
        "purpose": "independent notation-reader validation",
    }


@app.post("/api/read")
async def read_score(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(400, "Upload a PDF, PNG, or JPEG score.")

    session_id = uuid.uuid4().hex
    session = SESSIONS / session_id
    session.mkdir(parents=True, exist_ok=True)
    input_path = session / ("score" + suffix)
    with input_path.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    if input_path.stat().st_size == 0:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        xml_path = run_homr(input_path, session)
        metrics = analyze_musicxml(xml_path)
        stable_xml = session / "result.musicxml"
        if xml_path != stable_xml:
            shutil.copy2(xml_path, stable_xml)
    except Exception as exc:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "version": VERSION,
        "reader": "HOMR",
        "audiveris_used": False,
        "musicxml": f"/session/{session_id}/result.musicxml",
        "metrics": metrics,
    })


@app.get("/session/{session_id}/result.musicxml")
def get_result(session_id: str):
    if not session_id.isalnum():
        raise HTTPException(404)
    path = SESSIONS / session_id / "result.musicxml"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/vnd.recordare.musicxml+xml", filename="homr-result.musicxml")
