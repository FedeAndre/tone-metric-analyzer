from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from optical_reader import analyze_input

APP_VERSION = "tma-optical-reader-clean-v1"

app = FastAPI(title="TMA Optical Attack Reader", version=APP_VERSION)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TMA Optical Attack Reader</title>
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
  <h1>TMA Optical Attack Reader</h1>
  <p class="sub">Independent optical score reader. Upload a PDF or score image to generate the current notation graph. This build does not yet produce final TMA rhythmic attacks.</p>
  <div class="card">
    <form id="form">
      <label for="file"><strong>Score file</strong></label>
      <input id="file" name="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" required>
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
form.addEventListener('submit',async(e)=>{
  e.preventDefault();
  const file=document.getElementById('file').files[0];
  if(!file)return;
  button.disabled=true;
  status.className='status';
  status.textContent='Analyzing… this can take a few minutes for a PDF.';
  result.textContent='Working…';
  const body=new FormData(); body.append('file',file);
  try{
    const response=await fetch('/api/optical/read',{method:'POST',body});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||('HTTP '+response.status));
    status.className='status ok';
    status.textContent='Analysis complete.';
    result.textContent=JSON.stringify(data,null,2);
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
        "stage": "optical_notation_graph",
        "semantic_timing_used": False,
        "rhythmic_attacks_ready": False,
    }


@app.post("/api/optical/read")
async def optical_read(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(
            status_code=415,
            detail="Supported inputs: PDF, PNG, JPG/JPEG, TIFF.",
        )

    with tempfile.TemporaryDirectory(prefix="tma-optical-") as tmp:
        root = Path(tmp)
        src = root / f"input{suffix}"
        with src.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        out_dir = root / "audit"
        try:
            result = analyze_input(src, out_dir, dpi=300, cache_dir=None)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Optical notation extraction failed: {exc}",
            ) from exc

        return result
