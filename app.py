from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from optical_reader import analyze_input

APP_VERSION = "tma-optical-reader-clean-v1"

app = FastAPI(title="TMA Optical Attack Reader", version=APP_VERSION)


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
