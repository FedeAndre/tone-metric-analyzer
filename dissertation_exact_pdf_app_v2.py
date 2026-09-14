from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import dissertation_app as base
import dissertation_exact_pdf_app as old
from tone_metric.canonical_score import build_hits_from_canonical_score
from tone_metric.pivots import build_pivot_profile, register_pivot_profile
from tone_metric.score_registration import build_layer_anchors_from_canonical_score
from tone_metric.trees import build_tree_profile, register_tree_profile
from tone_metric.waves import build_wave_profile, register_wave_profile

# PDF integration wrapper. The validated dissertation Levels / waves / pivots / trees
# implementation remains unchanged. Audiveris is executed once. The saved OMR
# project supplies the canonical physical attack times used by the analysis and the
# exact same canonical geometry is then reused to register that result to the
# uploaded PDF. MusicXML supplies notation metadata and measures, but it is not
# allowed to move or replace a physical PDF attack.
APP_VERSION = "1.2.1-dissertation-canonical-score-time"
app = old.app
app.version = APP_VERSION

CACHE = old.CACHE


def _remove_post_analyze_route() -> None:
    for route in list(app.router.routes):
        if getattr(route, "path", None) == "/api/analyze" and "POST" in (getattr(route, "methods", set()) or set()):
            app.router.routes.remove(route)


def _page_dimensions(meta: dict) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    raw = meta.get("page_dimensions") or {}
    for key, value in raw.items():
        try:
            page = int(key)
            width = float(value[0])
            height = float(value[1])
        except Exception:
            continue
        if width > 0 and height > 0:
            out[page] = (width, height)
    return out


def _system_bounds(meta: dict, dims: dict[int, tuple[float, float]]) -> dict[int, list[dict]]:
    groups: dict[tuple[int, int], dict] = {}
    for measure in meta.get("measures", []) or []:
        try:
            page = int(measure.get("page_index", 0))
            system = int(measure.get("system_index", 0))
        except Exception:
            continue
        width, height = dims.get(page, (1.0, 1.0))
        bucket = groups.setdefault((page, system), {
            "xs": [],
            "ys": [],
            "stack_left": [],
            "stack_right": [],
            "width": width,
            "height": height,
        })
        try:
            bucket["stack_left"].append(float(measure.get("stack_left")))
            bucket["stack_right"].append(float(measure.get("stack_right")))
        except Exception:
            pass
        for col in measure.get("columns", []) or []:
            try:
                bucket["xs"].append(float(col.get("visual_x_abs", col.get("x_abs"))))
                bucket["ys"].append(float(col.get("y_abs")))
            except Exception:
                continue

    by_page: dict[int, list[dict]] = {}
    for (page, system), bucket in groups.items():
        width = float(bucket["width"] or 1.0)
        height = float(bucket["height"] or 1.0)
        ys = bucket["ys"]
        xs = bucket["xs"]
        if not ys:
            continue
        ypad = max(18.0, height * 0.018)
        xpad = max(12.0, width * 0.010)
        left_candidates = list(bucket["stack_left"]) + xs
        right_candidates = list(bucket["stack_right"]) + xs
        top = max(0.0, min(ys) - ypad)
        bottom = min(height, max(ys) + ypad)
        left = max(0.0, min(left_candidates) - xpad) if left_candidates else 0.0
        right = min(width, max(right_candidates) + xpad) if right_candidates else width
        by_page.setdefault(page, []).append({
            "system_index": system,
            "top_norm": top / height,
            "bottom_norm": bottom / height,
            "guard_bottom_norm": bottom / height,
            "left_norm": left / width,
            "right_norm": right / width,
        })

    for page, rows in by_page.items():
        rows.sort(key=lambda row: row["top_norm"])
        for i, row in enumerate(rows):
            if i + 1 < len(rows):
                row["guard_bottom_norm"] = (row["bottom_norm"] + rows[i + 1]["top_norm"]) / 2.0
    return by_page


def _canonical_overlay(levels_payload: dict, canonical_meta: dict) -> dict:
    dims = _page_dimensions(canonical_meta)
    visual_levels = deepcopy(levels_payload)
    visual_levels["canonical_score_meta"] = canonical_meta

    anchors_by_page, structural_by_page, stats, warnings = build_layer_anchors_from_canonical_score(
        visual_levels, dims
    )
    wave_profile = build_wave_profile(visual_levels)
    wave_by_page, wave_stats, wave_warnings = register_wave_profile(
        wave_profile, anchors_by_page, structural_by_page
    )
    pivot_profile = build_pivot_profile(wave_profile)
    pivot_by_page, pivot_stats, pivot_warnings = register_pivot_profile(
        pivot_profile, wave_by_page
    )
    tree_profile = build_tree_profile(wave_profile)
    tree_nodes_by_page, tree_branches_by_page, tree_stats, tree_warnings = register_tree_profile(
        tree_profile, wave_by_page
    )

    stats = dict(stats)
    stats.update(wave_stats)
    stats.update(pivot_stats)
    stats.update(tree_stats)
    warnings = list(warnings) + list(wave_warnings) + list(pivot_warnings) + list(tree_warnings)

    bounds = _system_bounds(canonical_meta, dims)
    pages = []
    for page in sorted(dims):
        pages.append({
            "page_index": page,
            "overlays": [],
            "layer_anchors": list(anchors_by_page.get(page, [])),
            "structural_anchors": list(structural_by_page.get(page, [])),
            "wave_anchors": list(wave_by_page.get(page, [])),
            "pivot_anchors": list(pivot_by_page.get(page, [])),
            "tree_nodes": list(tree_nodes_by_page.get(page, [])),
            "tree_branches": list(tree_branches_by_page.get(page, [])),
            "system_bounds": list(bounds.get(page, [])),
        })

    layer_anchor_count = sum(len(row["layer_anchors"]) for row in pages)
    stats["active_registration_pipeline"] = (
        "saved-omr-canonical-score->recursive-levels->existing-level-anchors->"
        "derived-wave-envelope->derived-pivots->derived-trees->original-pdf"
    )
    stats["visual_geometry_source"] = "saved-audiveris-omr"
    stats["musicxml_layout_required"] = False
    return {
        "available": layer_anchor_count > 0,
        "pages": pages,
        "max_level": int(stats.get("max_layer_level", 0) or 0),
        "matching": stats,
        "warnings": list(dict.fromkeys(str(x) for x in warnings if str(x).strip())),
    }


def _registration_report(canonical_meta: dict, pdf_pages: list[dict]) -> list[dict]:
    dims = _page_dimensions(canonical_meta)
    out = []
    for page in pdf_pages:
        idx = int(page["index"])
        pdf_w = float(page.get("width") or 1.0)
        pdf_h = float(page.get("height") or 1.0)
        omr_w, omr_h = dims.get(idx, (0.0, 0.0))
        pdf_ratio = pdf_w / pdf_h if pdf_h else 0.0
        omr_ratio = omr_w / omr_h if omr_h else 0.0
        rel_error = abs(pdf_ratio - omr_ratio) / max(abs(pdf_ratio), 1e-9) if omr_ratio else None
        out.append({
            "page_index": idx,
            "pdf_width": int(pdf_w),
            "pdf_height": int(pdf_h),
            "omr_width": int(omr_w) if omr_w else 0,
            "omr_height": int(omr_h) if omr_h else 0,
            "normalized_page_registration": bool(rel_error is not None and rel_error <= 0.02),
            "aspect_ratio_relative_error": rel_error,
        })
    return out


def _build_visual_job(
    session_id: str,
    pdf_path: Path,
    canonical_meta: dict,
    canonical_warnings: list[str],
    analysis_payload: dict,
) -> None:
    """Build only the visual product from the already-authoritative analysis state."""
    session = CACHE / session_id
    try:
        old._set_job(session_id, status="processing", stage="Rendering the exact uploaded PDF")
        pages = old.render_pdf_pages(pdf_path, session / "pages")
        public_pages = [
            {**p, "url": f"/api/visual-session/{session_id}/page/{p['index']}"}
            for p in pages
        ]
        old._set_job(session_id, pages=public_pages, stage="Registering analysis to the original PDF")

        overlay = _canonical_overlay(analysis_payload.get("levels") or {}, canonical_meta)
        registration = _registration_report(canonical_meta, pages)
        warnings = list(overlay.get("warnings", []) or [])
        warnings.extend(str(w) for w in canonical_warnings if str(w).strip())

        annotated_pdf = session / "score-with-tone-metric-analysis.pdf"
        old._write_annotated_pdf(pdf_path, annotated_pdf, overlay)

        manifest = {
            "status": "ready",
            "session_id": session_id,
            "pages": public_pages,
            "overlay": overlay,
            "page_registration": registration,
            "warnings": list(dict.fromkeys(warnings)),
            "analysis_unchanged": True,
            "analysis_engine_unchanged": True,
            "analysis_hit_source": "canonical-score-time",
            "exact_pdf_background": True,
            "background_source": "uploaded-pdf",
            "geometry_source": "saved-audiveris-omr",
            "musicxml_layout_required": False,
            "annotated_pdf_url": f"/api/visual-session/{session_id}/annotated.pdf",
            "original_pdf_url": f"/api/visual-session/{session_id}/original.pdf",
        }
        (session / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        old._set_job(session_id, **manifest)
    except Exception as exc:
        print(f"VISUAL_JOB_FAILED session={session_id} detail={exc}", flush=True)
        old._set_job(
            session_id,
            status="failed",
            stage="Original-PDF score overlay stopped",
            detail=str(exc),
            analysis_unchanged=True,
            analysis_engine_unchanged=True,
        )


_remove_post_analyze_route()


@app.post("/api/analyze")
async def analyze_with_single_omr_pass(
    file: UploadFile = File(...),
    initial_meter: str = Form("auto"),
):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix != ".pdf":
        return await base.analyze_endpoint(file=file, initial_meter=initial_meter)

    old._cleanup_cache()
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(payload) > 80 * 1024 * 1024:
        raise HTTPException(413, "Upload is larger than the 80 MB online limit.")

    try:
        override = base._meter_override(initial_meter)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    session_id = uuid.uuid4().hex
    session = CACHE / session_id
    session.mkdir(parents=True, exist_ok=True)
    saved_pdf = session / filename
    saved_pdf.write_bytes(payload)

    # One and only one PDF->Audiveris pass. MusicXML establishes measure/notation
    # metadata; the saved OMR project establishes canonical physical attack times.
    try:
        symbolic_path, omr_path = old.pdf_to_musicxml(saved_pdf, session / "audiveris")
    except Exception as exc:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(422, f"PDF optical-music recognition failed: {exc}") from exc

    try:
        symbolic_hits, measures, parser_warnings = old.parse_musicxml(
            symbolic_path, initial_meter_override=override
        )
        if not measures:
            raise ValueError("No measures were recovered from the score.")

        canonical_hits, canonical_warnings, canonical_meta = build_hits_from_canonical_score(
            omr_path, measures, symbolic_hits=symbolic_hits
        )
        if not canonical_hits:
            raise ValueError(
                "Audiveris did not provide canonical physical attack columns; "
                "the dissertation PDF pipeline will not silently substitute MusicXML timing."
            )

        # The analytical algorithm itself is unchanged. The difference from the
        # previous PDF wrapper is that it now receives the physical attack sequence
        # recovered from the score rather than lossy MusicXML-only attack timing.
        analysis_payload = base.analyze_full(canonical_hits, measures)
    except Exception as exc:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(422, f"Tone-metric analysis failed: {exc}") from exc

    meter_value = "auto" if override is None else f"{override[0]}/{override[1]}"
    analysis_payload["source"] = {
        "filename": filename,
        "input_type": "pdf",
        "meter_override": meter_value,
        "attack_timing_source": "canonical-score-time",
    }
    analysis_payload["analysis_hit_source"] = "canonical-score-time"
    analysis_payload["symbolic_hit_count"] = len(symbolic_hits)
    analysis_payload["canonical_hit_count"] = len(canonical_hits)
    analysis_payload["warnings"] = base._warnings(
        analysis_payload,
        list(parser_warnings) + list(canonical_warnings),
    )
    response = JSONResponse(analysis_payload)

    old._set_job(
        session_id,
        status="queued",
        stage="Waiting to build original-PDF visualization",
        created_at=time.time(),
        pages=[],
        analysis_unchanged=True,
        analysis_engine_unchanged=True,
        analysis_hit_source="canonical-score-time",
    )
    threading.Thread(
        target=_build_visual_job,
        args=(
            session_id,
            saved_pdf,
            canonical_meta,
            list(canonical_warnings),
            analysis_payload,
        ),
        daemon=True,
        name=f"tone-metric-exact-pdf-v2-{session_id[:8]}",
    ).start()

    response.headers["X-Tone-Metric-Visual-Session"] = session_id
    return response
