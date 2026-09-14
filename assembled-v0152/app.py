from __future__ import annotations
import shutil
import time
import uuid
import json
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from fractions import Fraction

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from tone_metric.musicxml import parse_musicxml, extract_visual_groups
from tone_metric.engine import analyze
from tone_metric.omr import pdf_to_musicxml, pdf_to_annotations, find_audiveris
from tone_metric.pdfview import render_pdf_pages
from tone_metric.physical import build_normalized_overlay
from tone_metric.omr_project import read_omr_slots, omr_slots_debug_rows
from tone_metric.canonical_score import build_hits_from_canonical_score

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".tone_metric_cache"
CACHE.mkdir(exist_ok=True)
app = FastAPI(title="Tone-Metric Analyzer", version="0.15.2")
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
            start = Fraction(m["start_quarter"]); end = Fraction(m["end_quarter"])
            if start <= t < end:
                q = (t - start) / beat
                if q.denominator == 1:
                    out.append(f"m.{m['number']} b{int(q)+1}")
                else:
                    out.append(f"m.{m['number']} +{t-start}")
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
    return {"ok": True, "audiveris_found": bool(find_audiveris()), "audiveris_command": find_audiveris()}


@app.get("/api/session/{session_id}/page/{page_index}")
def score_page(session_id: str, page_index: int):
    if not session_id.replace('-', '').isalnum():
        raise HTTPException(404)
    path = CACHE / session_id / "pages" / f"page_{page_index+1}.png"
    if not path.exists():
        raise HTTPException(404, "Score page not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/session/{session_id}/debug")
def debug_bundle(session_id: str):
    """Download all Audiveris intermediates for one analysis session.

    The archive is intentionally kept for debugging alignment: input PDF, saved
    .omr project, MusicXML/MXL export, annotation archive, logs and a compact
    mapping summary. Sessions are retained by the normal 24-hour cache policy.
    """
    if not session_id.replace('-', '').isalnum():
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
            # Score-page PNGs are reproducible from the PDF and make debug bundles huge.
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
            # Separate optional pass: physical notehead boxes are much more reliable for
            # overlays than MusicXML layout coordinates. Failure here must not kill analysis.
            annotation_archive, annotation_warning = pdf_to_annotations(inp, session / "annotations")
        else:
            symbolic = inp

        meter_override_tuple = _meter_override_tuple(meter_override)
        hits, measures, parse_warnings = parse_musicxml(symbolic, initial_meter_override=meter_override_tuple)
        analysis_hits = hits
        canonical_score_meta = {}
        if suffix == ".pdf" and omr_path is not None:
            try:
                recovered_hits, canonical_warnings, canonical_score_meta = build_hits_from_canonical_score(
                    omr_path, measures, symbolic_hits=hits
                )
                if recovered_hits:
                    analysis_hits = recovered_hits
                    parse_warnings = list(parse_warnings) + list(canonical_warnings)
            except Exception as exc:
                parse_warnings = list(parse_warnings) + [
                    f"Canonical score-time recovery failed; symbolic MusicXML hits were used: {exc}"
                ]
        result = analyze(analysis_hits, measures)
        result["analysis_hit_source"] = "canonical-score-time" if analysis_hits is not hits else "musicxml-symbolic"
        result["symbolic_hit_count"] = len(hits)
        result["canonical_score_meta"] = canonical_score_meta
        result["canonical_hit_count"] = int(canonical_score_meta.get("canonical_hit_count", len(analysis_hits)) or 0)
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
        result["parse_warnings"] = parse_warnings
        result["session_id"] = session_id
        result["debug_bundle_url"] = f"/api/session/{session_id}/debug"
        result["original_score_pages"] = [
            {**p, "url": f"/api/session/{session_id}/page/{p['index']}"}
            for p in page_info
        ]
        result["omr_saved"] = bool(omr_path)
        result["physical_overlay"] = {"available": False, "pages": [], "warnings": [], "matching": {}}

        # Persist a human-readable slot table alongside the raw .omr.  This is not
        # used to calculate the result; it exists so a returned debug bundle can be
        # inspected without reverse-engineering the Audiveris archive first.
        if omr_path is not None:
            try:
                omr_slots, omr_meta = read_omr_slots(omr_path)
                (session / "omr-slots.json").write_text(
                    json.dumps({"meta": omr_meta, "slots": omr_slots_debug_rows(omr_slots)}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                (session / "omr-slots-error.txt").write_text(str(exc), encoding="utf-8")

        if suffix == ".pdf" and annotation_archive is not None:
            try:
                visual_groups, layout_known = extract_visual_groups(symbolic, initial_meter_override=meter_override_tuple)
                result["physical_overlay"] = build_normalized_overlay(
                    annotation_archive, visual_groups, layout_known, result, session / "physical", omr_path=omr_path
                )
            except Exception as exc:
                result["physical_overlay"] = {
                    "available": False, "pages": [], "matching": {},
                    "warnings": [f"Physical notehead overlay could not be built: {exc}"],
                }
        elif suffix == ".pdf" and annotation_warning:
            result["physical_overlay"]["warnings"].append(annotation_warning)

        # v0.15.2 aligns the executable output with the mathematical paper by
        # exposing H(t), D(t), and lambda(t), using Definition 8 pivot moments as single structural turning points,
        # and retaining the separate executable theory-reference module.
        # Canonical score timing and exact attack registration remain unchanged.

        try:
            debug_summary = {
                "app_version": app.version,
                "source_filename": result.get("source_filename"),
                "symbolic_source": result.get("symbolic_source"),
                "omr_saved": result.get("omr_saved"),
                "meter_override": result.get("meter_override"),
                "measures": result.get("measures", []),
                "segments": [
                    {
                        "meter": seg.get("meter"),
                        "beat_unit_quarter": seg.get("beat_unit_quarter"),
                        "level1_anchor_labels": seg.get("level1_anchor_labels", []),
                        "events": [
                            {k: e.get(k) for k in (
                                "event_index","measure_index","measure_number","offset_in_measure_quarter",
                                "onset_quarter","duration_quarter","tone_metric_levels","tone_metric_height","tone_metric_density","lowest_tone_metric_level","attack_key","pitches"
                            )}
                            for e in seg.get("events", []) if e.get("tone_metric_levels")
                        ],
                    } for seg in result.get("segments", [])
                ],
                "physical_matching": result.get("physical_overlay", {}).get("matching", {}),
                "physical_warnings": result.get("physical_overlay", {}).get("warnings", []),
                "layer_anchors": {
                    str(p.get("page_index")): p.get("layer_anchors", [])
                    for p in result.get("physical_overlay", {}).get("pages", [])
                },
                "structural_anchors": {
                    str(p.get("page_index")): p.get("structural_anchors", [])
                    for p in result.get("physical_overlay", {}).get("pages", [])
                },
                "wave_anchors": {
                    str(p.get("page_index")): p.get("wave_anchors", [])
                    for p in result.get("physical_overlay", {}).get("pages", [])
                },
                "pivot_anchors": {
                    str(p.get("page_index")): p.get("pivot_anchors", [])
                    for p in result.get("physical_overlay", {}).get("pages", [])
                },
                "tree_nodes": {
                    str(p.get("page_index")): p.get("tree_nodes", [])
                    for p in result.get("physical_overlay", {}).get("pages", [])
                },
                "tree_branches": {
                    str(p.get("page_index")): p.get("tree_branches", [])
                    for p in result.get("physical_overlay", {}).get("pages", [])
                },
            }
            (session / "debug-summary.json").write_text(json.dumps(debug_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return JSONResponse(result)
    except RuntimeError as e:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(503, str(e))
    except Exception as e:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(422, f"Could not analyze score: {e}")
