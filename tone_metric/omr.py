from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path


def find_audiveris() -> str | None:
    env = os.environ.get("AUDIVERIS_CMD")
    if env:
        return env
    return shutil.which("audiveris") or shutil.which("Audiveris")


def _run(cmd: str, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([cmd] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def _find_export(output_dir: Path) -> Path | None:
    for pattern in ("*.mxl", "*.musicxml", "*.xml"):
        candidates = sorted(output_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


def _find_omr(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("*.omr"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_annotations(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("*annotations*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def pdf_to_musicxml(pdf_path: str | Path, output_dir: str | Path) -> tuple[Path, Path | None]:
    """Transcribe a PDF once and return the MusicXML export plus finalized OMR.

    Do not use Audiveris -save or -annotate during transcription. On larger
    multi-sheet scores, save-every-step can force Audiveris to reopen
    intermediate sheet state while the same book ZIP filesystem is active.
    The finalized OMR is annotated separately by pdf_to_annotations().
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = find_audiveris()
    if not cmd:
        raise RuntimeError("Audiveris was not found. PDF upload requires Audiveris. Set AUDIVERIS_CMD or add it to PATH.")

    proc = _run(
        cmd,
        ["-batch", "-transcribe", "-export", "-output", str(output_dir), "--", str(pdf_path)],
    )
    try:
        (output_dir / "audiveris-transcribe.log").write_text(
            proc.stdout or "", encoding="utf-8", errors="replace"
        )
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"Audiveris failed with exit code {proc.returncode}.\n\n"
            f"Audiveris output:\n{(proc.stdout or '')[-7000:]}"
        )

    symbolic = _find_export(output_dir)
    omr = _find_omr(output_dir)
    if symbolic is None:
        outputs = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
        raise RuntimeError(
            "Audiveris completed but no MusicXML export was found. "
            + f"Outputs found: {outputs[:50]}\n\nAudiveris output:\n{(proc.stdout or '')[-5000:]}"
        )
    if omr is None:
        outputs = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
        raise RuntimeError(
            "Audiveris completed but no finalized OMR project was found. "
            + f"Outputs found: {outputs[:50]}"
        )
    return symbolic, omr


def pdf_to_annotations(pdf_path: str | Path, output_dir: str | Path) -> tuple[Path | None, str]:
    """Generate physical annotations from the already-finalized OMR project.

    This is annotation-only: it never retranscribes the PDF and never uses
    save-every-step. Audiveris writes book annotations beside the finalized
    OMR project, so the annotation pass is deliberately run in that same
    directory. Failure remains non-fatal to tone-metric analysis because
    annotations are used only for the physical score overlay.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_dir = output_dir.parent / "omr"
    omr = _find_omr(primary_dir)
    if omr is None:
        return None, "Finalized Audiveris OMR project was not found; physical overlay is unavailable."

    existing = _find_annotations(primary_dir)
    if existing is not None:
        return existing, ""

    cmd = find_audiveris()
    if not cmd:
        return None, "Audiveris was not found for annotation generation; physical overlay is unavailable."

    proc = _run(
        cmd,
        ["-batch", "-annotate", "-output", str(primary_dir), "--", str(omr)],
    )
    try:
        (output_dir / "audiveris-annotate.log").write_text(
            proc.stdout or "", encoding="utf-8", errors="replace"
        )
    except Exception:
        pass

    archive = _find_annotations(primary_dir)
    if proc.returncode != 0:
        return None, (
            f"Audiveris annotation-only pass failed with exit code {proc.returncode}; "
            "tone-metric analysis is available but the physical overlay is unavailable. "
            f"Audiveris output: {(proc.stdout or '')[-2500:]}"
        )
    if archive is None:
        return None, (
            "Audiveris annotation-only pass completed without an annotation archive; "
            "physical overlay is unavailable."
        )
    return archive, ""
