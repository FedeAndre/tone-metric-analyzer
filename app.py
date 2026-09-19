from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


APP_VERSION = "tma-clean-product-v1"
JOB_ROOT = Path(tempfile.gettempdir()) / "tma-optical-jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tone-Metric Analyzer", version=APP_VERSION)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tma-optical")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _safe_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return dict(job)


def _set_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _run_job(job_id: str, src: Path, out_dir: Path, meter: str) -> None:
    """Run each score analysis in a disposable child process.

    The API process deliberately does not import ONNX Runtime or the optical
    model stack.  This keeps the web server's resident set small and lets the
    OS reclaim every native allocator page when a score finishes, instead of
    accumulating model/session memory inside a long-lived worker thread.
    """
    _set_job(job_id, status="running", started_at=time.time())
    try:
        app_root = Path(__file__).resolve().parent
        command = [
            sys.executable,
            str(app_root / "analyzer.py"),
            str(src),
            "--out",
            str(out_dir),
            "--meter",
            meter,
            "--dpi",
            "300",
        ]
        completed = subprocess.run(
            command,
            cwd=str(app_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60 * 60,
            check=False,
        )
        log_text = completed.stdout or ""
        if completed.returncode != 0:
            tail = log_text[-12000:]
            raise RuntimeError(
                f"reader subprocess exited {completed.returncode}.\n{tail}"
            )

        result_path = out_dir / "analysis.json"
        if not result_path.exists():
            raise RuntimeError(
                "analyzer subprocess completed without analysis.json"
            )
        result = json.loads(result_path.read_text())
        _set_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            result=result,
            result_path=str(result_path),
            log_tail=log_text[-4000:],
        )
    except subprocess.TimeoutExpired as exc:
        _set_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            error=f"Optical notation extraction timed out: {exc}",
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            error=f"Optical notation extraction failed: {exc}",
        )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tone-Metric Analyzer</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#0b0f14;color:#e8eef5;font-family:Arial,Helvetica,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:48px 24px}
  h1{font-size:30px;margin:0 0 8px}
  .sub{color:#9fb0c3;margin:0 0 28px;line-height:1.5}
  .card{border:1px solid #263241;background:#111821;border-radius:14px;padding:22px;margin-bottom:18px}
  input[type=file]{display:block;width:100%;margin:12px 0 18px}
  button{background:#e8eef5;color:#0b0f14;border:0;border-radius:9px;padding:11px 18px;font-weight:700;cursor:pointer}
  button:disabled{opacity:.45;cursor:not-allowed}
  .status{font-size:14px;color:#9fb0c3;margin-top:14px}
  pre{white-space:pre-wrap;word-break:break-word;max-height:520px;overflow:auto;background:#080c11;border-radius:10px;padding:16px;border:1px solid #202a36}
  .ok{color:#79d6a3}.err{color:#ff8f8f}
</style>
</head>
<body>
<div class="wrap">
  <h1>Tone-Metric Analyzer</h1>
  <p class="sub">Upload a score and provide its notated meter. The clean pipeline reads the score optically, proves the exact rhythmic attack timeline, then computes Tone-Metric Levels, wave, pivots, and trees. If exact timing cannot be proved, analysis stops instead of inventing timing.</p>
  <div class="card">
    <form id="form">
      <label for="file"><strong>Score file</strong></label>
      <input id="file" name="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" required>
      <label for="meter"><strong>Notated meter</strong></label>
      <input id="meter" name="meter" type="text" value="4/4" pattern="[0-9]+/[0-9]+" required style="display:block;margin:12px 0 18px;padding:9px;border-radius:7px;border:1px solid #334155;background:#0b0f14;color:#e8eef5">
      <button id="submit" type="submit">Analyze score</button>
      <div id="status" class="status">Ready.</div>
    </form>
  </div>
  <div class="card">
    <strong>Result</strong>
    <pre id="result">No analysis yet.</pre>
  </div>
</div>
<script>
const form=document.getElementById('form');
const button=document.getElementById('submit');
const status=document.getElementById('status');
const result=document.getElementById('result');

async function readJson(response){
  const text=await response.text();
  let data;
  try{ data=JSON.parse(text); }
  catch(_){ throw new Error('HTTP '+response.status+': '+(text||'non-JSON server response')); }
  if(!response.ok) throw new Error(data.detail||data.error||('HTTP '+response.status));
  return data;
}
function sleep(ms){ return new Promise(resolve=>setTimeout(resolve,ms)); }

form.addEventListener('submit',async(e)=>{
  e.preventDefault();
  const file=document.getElementById('file').files[0];
  if(!file)return;
  button.disabled=true;
  status.className='status';
  status.textContent='Uploading score…';
  result.textContent='Queued…';
  const body=new FormData(); body.append('file',file); body.append('meter',document.getElementById('meter').value);
  try{
    const submitResponse=await fetch('/api/analyze',{method:'POST',body});
    const submitted=await readJson(submitResponse);
    const jobId=submitted.job_id;
    status.textContent='Analysis queued. The page will keep checking until it finishes.';

    while(true){
      await sleep(2000);
      const pollResponse=await fetch('/api/optical/jobs/'+encodeURIComponent(jobId),{cache:'no-store'});
      const job=await readJson(pollResponse);
      if(job.status==='queued'){
        status.textContent='Waiting for the optical reader…';
        continue;
      }
      if(job.status==='running'){
        status.textContent='Analyzing score… this can take several minutes.';
        continue;
      }
      if(job.status==='failed'){
        throw new Error(job.error||'Analysis failed.');
      }
      if(job.status==='completed'){
        status.className='status ok';
        status.textContent='Analysis complete.';
        result.textContent=JSON.stringify(job.result,null,2);
        break;
      }
      throw new Error('Unknown job status: '+job.status);
    }
  }catch(err){
    status.className='status err';
    status.textContent='Analysis failed.';
    result.textContent=String(err);
  }finally{button.disabled=false;}
});
</script>
</body>
</html>"""


@app.get("/api/status")
def status() -> dict:
    return {
        "ok": True,
        "version": APP_VERSION,
        "reader": "independent-optical",
        "stage": "optical->exact-rhythm->tone-metric",
        "semantic_timing_used": False,
        "x_position_used_as_time": False,
        "job_mode": "asynchronous",
    }


@app.post("/api/analyze", status_code=202)
async def analyze_score_endpoint(
    file: UploadFile = File(...),
    meter: str = Form(...),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(
            status_code=415,
            detail="Supported inputs: PDF, PNG, JPG/JPEG, TIFF.",
        )

    try:
        from rhythm_reconstructor import parse_meter
        parse_meter(meter)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job_id = uuid.uuid4().hex
    root = JOB_ROOT / job_id
    root.mkdir(parents=True, exist_ok=False)
    src = root / f"input{suffix}"
    with src.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    out_dir = root / "audit"
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "filename": file.filename,
            "meter": meter,
        }

    _executor.submit(_run_job, job_id, src, out_dir, meter)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/api/optical/jobs/{job_id}",
        },
    )


@app.get("/api/optical/jobs/{job_id}")
def optical_job(job_id: str) -> dict:
    job = _safe_job(job_id)
    payload = {
        "job_id": job["job_id"],
        "status": job["status"],
    }
    if job.get("error"):
        payload["error"] = job["error"]
    if job.get("result") is not None:
        payload["result"] = job["result"]
    return payload
