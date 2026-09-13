from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from dissertation_app import app
from tone_metric.omr import find_audiveris


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    # Keep the API response unchanged, but surface the actual 4xx reason in Railway logs.
    print(
        f"HTTPException {exc.status_code} {request.method} {request.url.path}: {exc.detail}",
        flush=True,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/omr-status")
def omr_status():
    cmd = find_audiveris()
    tess = os.environ.get("TESSDATA_PREFIX")
    payload = {
        "audiveris_command": cmd,
        "audiveris_exists": bool(cmd and Path(cmd).exists()),
        "audiveris_executable": bool(cmd and Path(cmd).exists() and os.access(cmd, os.X_OK)),
        "tessdata_prefix": tess,
        "tessdata_exists": bool(tess and Path(tess).exists()),
    }
    if not cmd:
        payload["launch_ok"] = False
        payload["launch_error"] = "Audiveris command was not found."
        return payload
    try:
        proc = subprocess.run(
            [cmd, "-help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        payload["launch_returncode"] = proc.returncode
        payload["launch_ok"] = proc.returncode == 0
        payload["launch_output_tail"] = (proc.stdout or "")[-4000:]
    except Exception as exc:
        payload["launch_ok"] = False
        payload["launch_error"] = repr(exc)
    return payload
