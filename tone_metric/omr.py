from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True)
class AudiverisOmrResult:
    path: Path
    returncode: int
    salvaged_after_nonzero: bool
    command_args: tuple[str, ...]
    log_excerpt: str


def find_audiveris() -> str | None:
    env = os.environ.get("AUDIVERIS_CMD")
    if env:
        return env
    return shutil.which("audiveris") or shutil.which("Audiveris")


def _run(cmd: str, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [cmd] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _find_omr(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.rglob("*.omr"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def _find_annotations(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.rglob("*annotations*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _log_excerpt(text: str, head: int = 2600, tail: int = 7000) -> str:
    text = text or ""
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n\n... [middle of Audiveris log omitted] ...\n\n" + text[-tail:]


def _validate_saved_omr(path: Path) -> tuple[bool, str]:
    """Validate only persistence/structure here; score-time validity is checked downstream.

    A valid native Audiveris project is a ZIP archive containing book.xml and at
    least one sheet XML.  This deliberately does not invent or repair musical data.
    """
    try:
        with ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, f"corrupt ZIP member: {bad}"
            names = set(zf.namelist())
            if "book.xml" not in names:
                return False, "book.xml missing"
            sheet_xml = [
                name
                for name in names
                if name.startswith("sheet#") and name.endswith(".xml") and "/sheet#" in name
            ]
            if not sheet_xml:
                return False, "no sheet XML found"
    except (BadZipFile, OSError) as exc:
        return False, str(exc)
    return True, "ok"


def pdf_to_omr(pdf_path: str | Path, output_dir: str | Path) -> AudiverisOmrResult:
    """Run the OMR pipeline through PAGE and save the native .omr project.

    IMPORTANT: this required PDF pass intentionally uses ``-step PAGE -save``.
    It does *not* request ``-transcribe`` and does *not* request MusicXML export.
    Tone-Metric reads score time from the native .omr project itself.

    Audiveris can occasionally return a non-zero code after it has already
    persisted a usable .omr project because a later/secondary operation failed.
    We therefore validate the persisted project before deciding whether a
    non-zero return code is fatal.  Musical completeness is still checked by the
    canonical score-time parser; no alternate event source is substituted.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = find_audiveris()
    if not cmd:
        raise RuntimeError(
            "Audiveris was not found. PDF upload requires Audiveris. "
            "Set AUDIVERIS_CMD or add it to PATH."
        )

    args = [
        "-batch",
        "-step",
        "PAGE",
        "-save",
        "-output",
        str(output_dir),
        "--",
        str(pdf_path),
    ]
    proc = _run(cmd, args)
    full_log = proc.stdout or ""
    excerpt = _log_excerpt(full_log)
    try:
        (output_dir / "audiveris-page-save.log").write_text(
            full_log, encoding="utf-8", errors="replace"
        )
        (output_dir / "audiveris-command.txt").write_text(
            " ".join([cmd] + args), encoding="utf-8", errors="replace"
        )
    except Exception:
        pass

    omr = _find_omr(output_dir)
    if omr is None:
        outputs = [
            str(p.relative_to(output_dir))
            for p in output_dir.rglob("*")
            if p.is_file()
        ]
        raise RuntimeError(
            f"Audiveris PAGE/save pass ended with exit code {proc.returncode}, "
            "and no saved .omr project was found. "
            f"Outputs found: {outputs[:50]}\n\nAudiveris output:\n{excerpt}"
        )

    valid, reason = _validate_saved_omr(omr)
    if not valid:
        raise RuntimeError(
            f"Audiveris produced an unusable .omr project ({reason}); "
            f"exit code {proc.returncode}.\n\nAudiveris output:\n{excerpt}"
        )

    return AudiverisOmrResult(
        path=omr,
        returncode=int(proc.returncode),
        salvaged_after_nonzero=bool(proc.returncode != 0),
        command_args=tuple(args),
        log_excerpt=excerpt,
    )


def omr_to_annotations(omr_path: str | Path, output_dir: str | Path) -> tuple[Path | None, str]:
    """Optional display-only annotation pass from the already-saved .omr project.

    This pass never retranscribes the source PDF and never requests MusicXML
    export.  Failure cannot alter score time or Levels.
    """
    omr_path = Path(omr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = find_audiveris()
    if not cmd:
        return None, "Audiveris not found for annotation pass."

    args = ["-batch", "-annotate", "-output", str(output_dir), "--", str(omr_path)]
    proc = _run(cmd, args)
    full_log = proc.stdout or ""
    excerpt = _log_excerpt(full_log, head=1600, tail=4200)
    try:
        (output_dir / "audiveris-annotate.log").write_text(
            full_log, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass
    archive = _find_annotations(output_dir)
    if proc.returncode != 0:
        return None, f"Audiveris annotation pass failed with exit code {proc.returncode}. {excerpt}"
    if archive is None:
        return None, (
            "Audiveris annotation pass completed but no *annotations*.zip file was found. "
            + excerpt
        )
    return archive, excerpt
