from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from core import extract_hit_strikes
from render import render_pdf_with_strikes

SESSIONS = Path(tempfile.gettempdir()) / "tone_metric_hit_only"
SESSIONS.mkdir(parents=True, exist_ok=True)
AUDIVERIS = os.environ.get("AUDIVERIS_CMD", "/opt/audiveris/bin/Audiveris")
VERSION = "hit-only-semantic-recovery-v5"

app = FastAPI(title="Hit-only score marker")

HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hit-only score marker</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#eee;color:#111}main{max-width:1200px;margin:auto;padding:20px}
form{display:flex;gap:10px;align-items:center;margin-bottom:16px}button{padding:8px 14px}#status{margin:8px 0 16px;white-space:pre-wrap}.page{display:block;width:100%;height:auto;margin:0 0 20px;background:white}
</style></head><body><main>
<form id="f"><input id="file" type="file" accept="application/pdf,.pdf" required><button>Analyze</button></form>
<div id="status"></div><div id="pages"></div>
<script>
const f=document.getElementById('f'),status=document.getElementById('status'),pages=document.getElementById('pages');
f.addEventListener('submit',async e=>{e.preventDefault();pages.innerHTML='';status.textContent='Analyzing…';
 const fd=new FormData();fd.append('file',document.getElementById('file').files[0]);
 try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Analysis failed');
 status.textContent='';pages.innerHTML=d.pages.map(p=>`<img class="page" src="${p}">`).join('');
 }catch(err){status.textContent=err.message;}
});
</script></main></body></html>'''


def _find_one(root: Path, suffix: str) -> Path:
    matches = sorted(root.rglob(f"*{suffix}"))
    if not matches:
        raise RuntimeError(f"Audiveris did not produce {suffix}")
    return matches[0]


def _run_audiveris(pdf_path: Path, out_dir: Path) -> Path:
    # -export forces Audiveris through the complete transcription pipeline.
    # The exported MusicXML is not used by the hit extractor.
    cmd = [
        AUDIVERIS,
        "-batch",
        "-save",
        "-export",
        "-output",
        str(out_dir),
        str(pdf_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(f"Audiveris failed with exit code {proc.returncode}.\n{tail}")
    omr_path = _find_one(out_dir, ".omr")
    if omr_path.stat().st_size <= 0:
        raise RuntimeError("Audiveris produced an empty .omr project")
    return omr_path


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


@app.get("/api/status")
def status():
    return {
        "ok": True,
        "version": VERSION,
        "audiveris_found": Path(AUDIVERIS).exists(),
        "hit_source": "audiveris-semantic-slots-plus-unvoiced-head-chord-recovery",
        "musicxml_used_for_hits": False,
        "one_strike_per_hit": True,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    name = Path(file.filename or "score.pdf").name
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(400, "Upload a PDF score.")

    session_id = uuid.uuid4().hex
    session = SESSIONS / session_id
    session.mkdir(parents=True, exist_ok=True)
    pdf_path = session / "score.pdf"
    with pdf_path.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)

    if pdf_path.stat().st_size <= 0:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(400, "The uploaded PDF is empty.")

    omr_out = session / "audiveris"
    omr_out.mkdir()
    try:
        omr_path = _run_audiveris(pdf_path, omr_out)
        logical_hit_count, strikes = extract_hit_strikes(omr_path)
        if len(strikes) != logical_hit_count:
            raise RuntimeError(
                f"Hit invariant failed: {logical_hit_count} hits but {len(strikes)} strikes"
            )
        pages = render_pdf_with_strikes(pdf_path, strikes, session / "pages")
    except Exception as exc:
        print(f"ANALYZE_ERROR [{VERSION}] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc

    return JSONResponse({
        "version": VERSION,
        "hit_count": logical_hit_count,
        "strike_count": len(strikes),
        "pages": [f"/session/{session_id}/{p.name}" for p in pages],
    })


@app.get("/session/{session_id}/{filename}")
def session_page(session_id: str, filename: str):
    if not session_id.isalnum() or not filename.startswith("page-") or not filename.endswith(".png"):
        raise HTTPException(404)
    path = SESSIONS / session_id / "pages" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")
