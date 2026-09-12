from __future__ import annotations

import ast
import collections
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "0.16.3"


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def _project_python_files() -> list[Path]:
    """Return only first-party runtime source files.

    Deliberately excludes .venv/site-packages, build outputs, caches, and tests so
    third-party code can never create a false legacy/duplicate audit result.
    """
    files = [ROOT / "app.py"]
    files.extend(sorted((ROOT / "tone_metric").glob("*.py")))
    return [p for p in files if p.is_file()]


def source_audit() -> None:
    # No duplicate top-level definitions in first-party runtime source:
    # one current implementation per symbol.
    duplicates = []
    for path in _project_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        repeated = [
            name for name, count in collections.Counter(names).items() if count > 1
        ]
        if repeated:
            duplicates.append((str(path.relative_to(ROOT)), repeated))
    if duplicates:
        fail(f"duplicate top-level definitions: {duplicates}")

    engine = (ROOT / "tone_metric" / "engine.py").read_text(encoding="utf-8")
    for forbidden in (
        "_refine_span(",
        "_ternary_spans_from_hits(",
        "_denominator_reachable_by_local_arity(",
        "explicit_local_ternary_span_count",
    ):
        if forbidden in engine:
            fail(f"legacy Levels path still present: {forbidden}")

    app = (ROOT / "app.py").read_text(encoding="utf-8")
    if app.count("from tone_metric.engine import analyze") != 1:
        fail("app does not import exactly one Levels analyze entry point")
    if 'result["silent_event_fallback_enabled"] = False' not in app:
        fail("strict no-fallback marker missing")
    if (
        "Preserve the previous symbolic fallback" in app
        or "symbolic MusicXML hits were used" in app
    ):
        fail("old symbolic fallback path remains")

    if "pdf_to_musicxml" in app or "reconcile_measure_framework_from_omr" in app:
        fail("export-dependent PDF path remains in app")
    if "pdf_to_omr" not in app or "build_measure_framework_from_omr" not in app:
        fail("OMR-only PDF path is not wired into app")

    omr = (ROOT / "tone_metric" / "omr.py").read_text(encoding="utf-8")
    if "def pdf_to_musicxml" in omr or '"-export"' in omr:
        fail("MusicXML export remains in the required Audiveris PDF driver")
    if '"-step",' not in omr or '"PAGE",' not in omr or '"-save",' not in omr:
        fail("required Audiveris driver is not explicitly step PAGE + save")
    if 'args = ["-batch", "-transcribe"' in omr:
        fail("required PDF pass still invokes transcribe")
    if "def pdf_to_annotations" in omr:
        fail("old PDF retranscription annotation path remains")
    if "def omr_to_annotations" not in omr:
        fail("saved-OMR annotation path missing")

    canonical = (ROOT / "tone_metric" / "canonical_score.py").read_text(encoding="utf-8")
    if "def reconcile_measure_framework_from_omr" in canonical:
        fail("superseded MusicXML/OMR reconciliation path remains")


def main() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
    )
    if proc.returncode:
        fail("unit tests failed")
    source_audit()
    import app

    if app.app.version != VERSION:
        fail(f"app version is {app.app.version}, expected {VERSION}")
    print(f"VALIDATION PASSED: v{VERSION} tests, first-party source audit, and app import")


if __name__ == "__main__":
    main()
