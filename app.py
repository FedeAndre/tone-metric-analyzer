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

VERSION = "hit-reconciled-v1"
AUDIVERIS = os.environ.get("AUDIVERIS_CMD", "/opt/audiveris/bin/Audiveris")
SESSIONS = Path(tempfile.gettempdir()) / "hit_reconciled_v1"
SESSIONS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Hit-only score marker")

HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hit-only score marker</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#eee;color:#111}
main{max-width:1200px;margin:auto;padding:20px}
form{display:flex;gap:10px;align-items:center;margin-bottom:16px}
button{padding:8px 14px}
#status{margin:8px 0 16px;white-space:pre-wrap}
.page{display:block;width:100%;height:auto;margin:0 0 20px;background:white}
</style>
</head>
<body>
<main>
<form id="f">
<input id="file" type="file" accept="application/pdf,.pdf" required>
<button>Analyze</button>
</form>
<div id="status"></div>
<div id="pages"></div>
<script>
const form=document.getElementById('f');
const status=document.getElementById('status');
const pages=document.getElementById('pages');
form.addEventListener('submit',async(event)=>{
  event.preventDefault();
  pages.innerHTML='';
  status.textContent='Analyzing…';
  const data=new FormData();
  data.append('file',document.getElementById('file').files[0]);
  try{
    const response=await fetch('/api/analyze',{method:'POST',body:data});
    const body=await response.json();
    if(!response.ok) throw new Error(body.detail||'Analysis failed');
    status.textContent='';
    pages.innerHTML=body.pages.map(src=>`<img class="page" src="${src}">`).join('');
  }catch(error){
    status.textContent=error.message;
  }
});
</script>
</main>
</body>
</html>"""


def _find_one(root: Path, suffix: str) -> Path:
    matches = sorted(root.rglob(f"*{suffix}"))
    if not matches:
        raise RuntimeError(f"Audiveris did not produce {suffix}")
    return matches[0]


def _run_audiveris(pdf_path: Path, output_dir: Path) -> Path:
    command = [
        AUDIVERIS,
        "-batch",
        "-save",
        "-export",
        "-output",
        str(output_dir),
        str(pdf_path),
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    if process.returncode != 0:
        tail = "\n".join(process.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"Audiveris failed with exit code {process.returncode}.\n{tail}"
        )

    omr_path = _find_one(output_dir, ".omr")
    if omr_path.stat().st_size <= 0:
        raise RuntimeError("Audiveris produced an empty .omr project")
    return omr_path


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return HTML


@app.get("/api/status")
def status() -> dict:
    return {
        "ok": True,
        "version": VERSION,
        "audiveris_found": Path(AUDIVERIS).exists(),
        "architecture": "symbolic-onset-plus-semantic-chord-reconciliation",
        "geometry_can_change_existing_hit_identity": False,
        "ambiguous_transcription_policy": "fail-closed",
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    filename = Path(file.filename or "score.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "Upload a PDF score.")

    session_id = uuid.uuid4().hex
    session = SESSIONS / session_id
    session.mkdir(parents=True, exist_ok=True)
    pdf_path = session / "score.pdf"

    with pdf_path.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    if pdf_path.stat().st_size <= 0:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(400, "The uploaded PDF is empty.")

    audiveris_dir = session / "audiveris"
    audiveris_dir.mkdir()

    try:
        omr_path = _run_audiveris(pdf_path, audiveris_dir)
        hit_count, strikes, diagnostics = extract_hit_strikes(omr_path)

        if hit_count != len(strikes):
            raise RuntimeError(
                f"Hit invariant failed: {hit_count} hits but {len(strikes)} strikes"
            )
        if diagnostics.assigned_chords != diagnostics.sounding_chords:
            raise RuntimeError("Semantic-chord coverage invariant failed")

        rendered = render_pdf_with_strikes(
            pdf_path,
            strikes,
            session / "pages",
        )
    except Exception as exc:
        print(
            f"ANALYZE_ERROR [{VERSION}] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(
            422,
            f"{type(exc).__name__}: {exc}",
        ) from exc

    return JSONResponse(
        {
            "version": VERSION,
            "hit_count": hit_count,
            "strike_count": len(strikes),
            "pages": [
                f"/session/{session_id}/{path.name}"
                for path in rendered
            ],
        }
    )


@app.get("/session/{session_id}/{filename}")
def session_page(session_id: str, filename: str):
    if (
        not session_id.isalnum()
        or not filename.startswith("page-")
        or not filename.endswith(".png")
    ):
        raise HTTPException(404)
    path = SESSIONS / session_id / "pages" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")
