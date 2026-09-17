from __future__ import annotations

import json
import shutil
import time
import uuid
from fractions import Fraction
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tone_metric.engine import analyze
from tone_metric.musicxml import extract_visual_groups, parse_musicxml
from tone_metric.omr import find_audiveris, pdf_to_annotations, pdf_to_musicxml
from tone_metric.omr_project import omr_slots_debug_rows, read_omr_slots
from tone_metric.pdfview import render_pdf_pages
from tone_metric.physical import build_normalized_overlay
from tone_metric.score_registration import build_symbolic_registration_meta

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".tone_metric_cache"
CACHE.mkdir(exist_ok=True)

app = FastAPI(title="Tone-Metric Analyzer", version="0.18.0-rc2-symbolic-notehead-registration")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def _meter_override_tuple(value: str | None):
    text = (value or "auto").strip().lower()
    if text == "auto" or not text:
        return None
    try:
        n, d = [int(x.strip()) for x in text.split("/", 1)]
    except Exception:
        raise HTTPException(400, "Meter override must be Auto or a value such as 2/2, 4/4, 3/4, 6/8, or 12/8.")
    if n <= 0 or d <= 0:
        raise HTTPException(400, "Invalid meter override.")
    return n, d


def _level1_anchor_labels(segment: dict, measures: list[dict]) -> list[str]:
    beat = Fraction(segment.get("beat_unit_quarter", "1"))
    out = []
    if beat <= 0:
        return out
    for raw in segment.get("level1_positions_quarter", []):
        t = Fraction(raw)
        for m in measures:
            start = Fraction(m["start_quarter"])
            end = Fraction(m["end_quarter"])
            if start <= t < end:
                q = (t - start) / beat
                if q.denominator == 1:
                    out.append(f"m.{m['number']} b{int(q) + 1}")
                else:
                    out.append(f"m.{m['number']} +{t - start}")
                break
    return out


def _cleanup_cache(max_age_hours: float = 24.0):
    cutoff = time.time() - max_age_hours * 3600
    for p in CACHE.iterdir():
        try:
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return {
        "ok": True,
        "version": app.version,
        "timing_authority": "musicxml-symbolic-global-onset",
        "omr_recovery": "disabled-fail-closed",
        "pdf_registration": "exact-symbolic-time-key-to-semantic-omr-notehead",
        "legacy_canonical_score_path": "absent",
        "audiveris_found": bool(find_audiveris()),
        "audiveris_command": find_audiveris(),
    }


@app.get("/api/session/{session_id}/page/{page_index}")
def score_page(session_id: str, page_index: int):
    if not session_id.replace("-", "").isalnum():
        raise HTTPException(404)
    path = CACHE / session_id / "pages" / f"page_{page_index + 1}.png"
    if not path.exists():
        raise HTTPException(404, "Score page not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/session/{session_id}/debug")
def debug_bundle(session_id: str):
    if not session_id.replace("-", "").isalnum():
        raise HTTPException(404)
    session = CACHE / session_id
    if not session.exists():
        raise HTTPException(404, "Analysis session not found")
    bundle = session / "tone-metric-debug.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as zf:
        for path in sorted(session.rglob("*")):
            if not path.is_file() or path == bundle:
                continue
            rel = path.relative_to(session)
            if rel.parts and rel.parts[0] == "pages":
                continue
            zf.write(path, str(rel))
    return FileResponse(bundle, media_type="application/zip", filename=f"tone-metric-debug-{session_id[:8]}.zip")


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), meter_override: str = Form("auto")):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in {".pdf", ".xml", ".musicxml", ".mxl"}:
        raise HTTPException(400, "Upload a PDF score. MusicXML/MXL are also accepted for validation/testing.")

    _cleanup_cache()
    session_id = uuid.uuid4().hex
    session = CACHE / session_id
    session.mkdir(parents=True, exist_ok=True)
    inp = session / (file.filename or f"upload{suffix}")
    inp.write_bytes(await file.read())

    try:
        page_info = []
        omr_path = None
        annotation_archive = None
        annotation_warning = ""

        if suffix == ".pdf":
            page_info = render_pdf_pages(inp, session / "pages")
            symbolic, omr_path = pdf_to_musicxml(inp, session / "omr")
            annotation_archive, annotation_warning = pdf_to_annotations(inp, session / "annotations")
        else:
            symbolic = inp

        meter_override_tuple = _meter_override_tuple(meter_override)
        hits, measures, parse_warnings = parse_musicxml(
            symbolic,
            initial_meter_override=meter_override_tuple,
        )

        # The parser's globally merged symbolic hits are the sole musical-time
        # authority. No OMR/PDF geometry can add, remove, split, merge, reorder,
        # or retime attacks before or after this point.
        result = analyze(hits, measures)
        result["analysis_hit_source"] = "musicxml-symbolic-global-onset"
        result["symbolic_hit_count"] = len(hits)
        result["omr_recovery_policy"] = "disabled-fail-closed"
        result["omr_recovered_hit_count"] = 0
        result["canonical_score_meta"] = {}
        result["canonical_hit_count"] = 0
        result["levels_enabled"] = sorted({
            int(level)
            for seg in result.get("segments", [])
            for level in seg.get("levels_enabled", [])
        })
        result["max_level"] = max(result["levels_enabled"], default=0)
        result["meter_override"] = meter_override if meter_override_tuple else "auto"
        for seg in result.get("segments", []):
            seg["level1_anchor_labels"] = _level1_anchor_labels(seg, result.get("measures", []))

        result["source_filename"] = file.filename
        result["symbolic_source"] = symbolic.name
        result["parse_warnings"] = list(parse_warnings)
        if suffix == ".pdf":
            result["parse_warnings"].append(
                "OMR-only attack recovery remains disabled. OMR is used only to locate already-established symbolic events on the PDF by exact measure/time key and semantic BEGIN-chord/head relations."
            )
        result["session_id"] = session_id
        result["debug_bundle_url"] = f"/api/session/{session_id}/debug"
        result["original_score_pages"] = [
            {**p, "url": f"/api/session/{session_id}/page/{p['index']}"}
            for p in page_info
        ]
        result["omr_saved"] = bool(omr_path)
        result["physical_overlay"] = {
            "available": False,
            "pages": [],
            "warnings": [],
            "matching": {
                "registration_policy": "symbolic-time-authoritative; semantic-omr-geometry-only; no-x-timing"
            },
        }

        registration_meta = None
        registration_ready = False
        if omr_path is not None:
            # Keep the raw Audiveris slot dump as an independent audit artifact.
            try:
                omr_slots, omr_meta = read_omr_slots(omr_path)
                (session / "omr-slots.json").write_text(
                    json.dumps({"meta": omr_meta, "slots": omr_slots_debug_rows(omr_slots)}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                (session / "omr-slots-error.txt").write_text(str(exc), encoding="utf-8")

            # Build a separate semantic geometry graph. It contains no reconstructed
            # attacks: it can only be joined later to a symbolic event with exactly
            # the same measure index and score-time offset.
            try:
                registration_meta = build_symbolic_registration_meta(omr_path)
                registration_ready = bool(registration_meta.get("available"))
                result["_symbolic_registration_meta"] = registration_meta
                (session / "symbolic-registration.json").write_text(
                    json.dumps(registration_meta, indent=2), encoding="utf-8"
                )
                result["symbolic_registration"] = {
                    k: registration_meta.get(k)
                    for k in (
                        "available",
                        "architecture",
                        "timing_authority",
                        "geometry_source",
                        "slot_count",
                        "slots_with_begin_chords",
                        "slots_with_exact_heads",
                        "begin_chord_count",
                        "exact_head_count",
                        "duplicate_time_keys",
                        "warnings",
                    )
                }
            except Exception as exc:
                result["symbolic_registration"] = {
                    "available": False,
                    "warnings": [f"Semantic OMR registration graph could not be built: {exc}"],
                }
                (session / "symbolic-registration-error.txt").write_text(str(exc), encoding="utf-8")

        if suffix == ".pdf" and annotation_archive is not None:
            try:
                visual_groups, layout_known = extract_visual_groups(
                    symbolic,
                    initial_meter_override=meter_override_tuple,
                )
                result["physical_overlay"] = build_normalized_overlay(
                    annotation_archive,
                    visual_groups,
                    layout_known,
                    result,
                    session / "physical",
                    # Non-null merely enables the registration branch in physical.py.
                    # The actual registration data come exclusively from
                    # _symbolic_registration_meta; canonical_score.py is absent.
                    omr_path=omr_path if registration_ready else None,
                )
                matching = result["physical_overlay"].setdefault("matching", {})
                matching.pop("canonical_score_meta", None)
                matching["registration_policy"] = (
                    "symbolic-time-authoritative; exact-measure/time-key; "
                    "voice-BEGIN-chord-contained-notehead; no-x-timing"
                )
                matching["active_registration_pipeline"] = (
                    "musicxml-symbolic-time->recursive-levels->exact-omr-slot-time-key->"
                    "voice-BEGIN->contained-notehead->pdf"
                )
                matching["legacy_canonical_score_path_enabled"] = False
                matching["omr_only_attack_recovery_enabled"] = False
                if not registration_ready:
                    result["physical_overlay"].setdefault("warnings", []).insert(
                        0,
                        "Semantic OMR notehead registration was unavailable; PDF event labels were left unmapped rather than guessed.",
                    )
            except Exception as exc:
                result["physical_overlay"] = {
                    "available": False,
                    "pages": [],
                    "matching": {
                        "registration_policy": "symbolic-time-authoritative; semantic-omr-geometry-only; no-x-timing",
                        "legacy_canonical_score_path_enabled": False,
                        "omr_only_attack_recovery_enabled": False,
                    },
                    "warnings": [f"Physical annotation extraction could not be built: {exc}"],
                }
        elif suffix == ".pdf" and annotation_warning:
            result["physical_overlay"]["warnings"].append(annotation_warning)

        # Private geometry graph is never returned as application state; the full
        # audit copy remains in symbolic-registration.json inside the debug bundle.
        result.pop("_symbolic_registration_meta", None)

        try:
            debug_summary = {
                "app_version": app.version,
                "source_filename": result.get("source_filename"),
                "symbolic_source": result.get("symbolic_source"),
                "analysis_hit_source": result.get("analysis_hit_source"),
                "symbolic_hit_count": result.get("symbolic_hit_count"),
                "omr_recovery_policy": result.get("omr_recovery_policy"),
                "omr_saved": result.get("omr_saved"),
                "meter_override": result.get("meter_override"),
                "symbolic_registration": result.get("symbolic_registration", {}),
                "measures": result.get("measures", []),
                "segments": [
                    {
                        "meter": seg.get("meter"),
                        "beat_unit_quarter": seg.get("beat_unit_quarter"),
                        "level1_anchor_labels": seg.get("level1_anchor_labels", []),
                        "structural_points": seg.get("structural_points", []),
                        "events": [
                            {k: e.get(k) for k in (
                                "event_index",
                                "measure_index",
                                "measure_number",
                                "offset_in_measure_quarter",
                                "onset_quarter",
                                "duration_quarter",
                                "tone_metric_levels",
                                "tone_metric_height",
                                "tone_metric_density",
                                "lowest_tone_metric_level",
                                "attack_key",
                                "pitches",
                            )}
                            for e in seg.get("events", [])
                            if e.get("tone_metric_levels")
                        ],
                    }
                    for seg in result.get("segments", [])
                ],
                "physical_matching": result.get("physical_overlay", {}).get("matching", {}),
                "physical_warnings": result.get("physical_overlay", {}).get("warnings", []),
            }
            (session / "debug-summary.json").write_text(
                json.dumps(debug_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        return JSONResponse(result)
    except RuntimeError as exc:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(503, str(exc))
    except HTTPException:
        shutil.rmtree(session, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(422, f"Could not analyze score: {exc}")
