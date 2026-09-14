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
    pdf_path = Path(pdf_path); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cmd = find_audiveris()
    if not cmd:
        raise RuntimeError("Audiveris was not found. PDF upload requires Audiveris. Set AUDIVERIS_CMD or add it to PATH.")
    proc = _run(cmd, ["-batch", "-transcribe", "-export", "-annotate", "-output", str(output_dir), "--", str(pdf_path)])
    try:
        (output_dir / "audiveris-transcribe.log").write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(f"Audiveris failed with exit code {proc.returncode}.\n\nAudiveris output:\n{(proc.stdout or '')[-7000:]}")
    symbolic = _find_export(output_dir)
    if symbolic is None:
        outputs = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
        raise RuntimeError("Audiveris completed but no MusicXML export was found. " + f"Outputs found: {outputs[:50]}\n\nAudiveris output:\n{(proc.stdout or '')[-5000:]}")
    return symbolic, _find_omr(output_dir)


def pdf_to_annotations(pdf_path: str | Path, output_dir: str | Path) -> tuple[Path | None, str]:
    """Return annotations produced by the primary Audiveris pass.

    Audiveris can export MusicXML, save the .omr project, and annotate symbols in
    the same transcription. Re-running transcription here would duplicate the
    expensive OMR pass and can exceed hosted HTTP request limits.
    """
    output_dir = Path(output_dir)
    primary_dir = output_dir.parent / "omr"
    archive = _find_annotations(primary_dir)
    if archive is not None:
        return archive, ""
    return None, "Audiveris primary pass completed without an annotation archive; physical overlay is unavailable."
