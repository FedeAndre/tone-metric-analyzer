from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path

import fitz
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import dissertation_app as base
from tone_metric.canonical_score import build_hits_from_canonical_score
from tone_metric.musicxml import extract_visual_groups, parse_musicxml
from tone_metric.omr import pdf_to_annotations, pdf_to_musicxml
from tone_metric.pdfview import render_pdf_pages
from tone_metric.physical import build_normalized_overlay, read_annotation_pages

# Presentation-only build. The validated analytical implementation remains
# dissertation_app.analyze_endpoint and is invoked unchanged below.
APP_VERSION = "1.1.0-dissertation-exact-pdf"
app = base.app
app.version = APP_VERSION

CACHE = Path("/tmp/tone-metric-dissertation-exact-pdf")
CACHE.mkdir(parents=True, exist_ok=True)
VISUAL_JOBS: dict[str, dict] = {}
VISUAL_LOCK = threading.Lock()


def _set_job(job_key: str, **values) -> None:
    with VISUAL_LOCK:
        row = VISUAL_JOBS.setdefault(job_key, {})
        row.update(values)


def _get_job(session_id: str) -> dict | None:
    with VISUAL_LOCK:
        row = VISUAL_JOBS.get(session_id)
        return dict(row) if row is not None else None


def _cleanup_cache(max_age_hours: float = 24.0) -> None:
    cutoff = time.time() - max_age_hours * 3600.0
    for p in CACHE.iterdir():
        try:
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _registration_report(annotation_archive: Path, pdf_pages: list[dict]) -> list[dict]:
    """Audit whether normalized Audiveris page coordinates map to the PDF page.

    The score displayed to the user is always rendered from the uploaded PDF. The
    Audiveris raster is never used as the visible background. Registration uses
    normalized full-page coordinates, so matching page aspect ratios are the
    invariant that must hold for the transform to be exact up to raster resolution.
    """
    annotations = read_annotation_pages(annotation_archive)
    by_index = {int(p.page_index): p for p in annotations}
    out: list[dict] = []
    for page in pdf_pages:
        idx = int(page["index"])
        ann = by_index.get(idx)
        pw = float(page.get("width") or 1)
        ph = float(page.get("height") or 1)
        aw = float(getattr(ann, "annotation_width", 0) or 0)
        ah = float(getattr(ann, "annotation_height", 0) or 0)
        pdf_ratio = pw / ph if ph else 0.0
        ann_ratio = aw / ah if ah else 0.0
        rel_error = abs(pdf_ratio - ann_ratio) / max(abs(pdf_ratio), 1e-9) if ann_ratio else None
        out.append({
            "page_index": idx,
            "pdf_width": int(pw),
            "pdf_height": int(ph),
            "annotation_width": aw,
            "annotation_height": ah,
            "normalized_page_registration": bool(rel_error is not None and rel_error <= 0.02),
            "aspect_ratio_relative_error": rel_error,
        })
    return out


def _system_models(overlay_page: dict, page_rect: fitz.Rect) -> dict[int, dict]:
    width, height = float(page_rect.width), float(page_rect.height)
    bounds = {int(b.get("system_index", 0)): b for b in overlay_page.get("system_bounds", []) or []}
    models: dict[int, dict] = {}
    for sys_idx, b in bounds.items():
        top = float(b.get("top_norm", 0.0)) * height
        bottom = float(b.get("bottom_norm", 0.0)) * height
        left = float(b.get("left_norm", 0.0)) * width
        right = float(b.get("right_norm", 1.0)) * width
        models[sys_idx] = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "max_level": 1,
        }
    for key in ("layer_anchors", "structural_anchors", "wave_anchors", "tree_nodes"):
        for row in overlay_page.get(key, []) or []:
            k = int(row.get("system_index", 0))
            m = models.get(k)
            if not m:
                continue
            vals = [int(x) for x in (row.get("levels") or []) if str(x).isdigit()]
            if row.get("height") is not None:
                try:
                    vals.append(int(row.get("height")))
                except Exception:
                    pass
            if row.get("lowest_level") is not None:
                try:
                    vals.append(int(row.get("lowest_level")))
                except Exception:
                    pass
            if vals:
                m["max_level"] = max(m["max_level"], max(vals))
    ordered = sorted(models.items(), key=lambda kv: kv[1]["top"])
    prev_bottom = 0.0
    for _, m in ordered:
        available = max(12.0, m["top"] - prev_bottom - 8.0)
        row_h = min(9.0, max(4.2, available / max(2, m["max_level"] + 1)))
        m["row_h"] = row_h
        m["grid_bottom"] = max(prev_bottom + 6.0, m["top"] - 5.0)
        m["grid_top"] = max(prev_bottom + 4.0, m["grid_bottom"] - row_h * max(1, m["max_level"] - 1))
        prev_bottom = m["bottom"]
    return models


def _pdf_level_y(model: dict, level: int) -> float:
    return float(model["grid_bottom"]) - (max(1, int(level)) - 1) * float(model["row_h"])


def _write_annotated_pdf(source_pdf: Path, out_pdf: Path, overlay: dict) -> None:
    """Write a new PDF whose base pages are the original uploaded PDF.

    No score reconstruction occurs: the source PDF pages remain the document pages,
    and only vector analysis marks are added on top.
    """
    doc = fitz.open(source_pdf)
    by_page = {int(p.get("page_index", 0)): p for p in overlay.get("pages", []) or []}
    for page_index, page in enumerate(doc):
        op = by_page.get(page_index)
        if not op:
            continue
        rect = page.rect
        width = float(rect.width)
        models = _system_models(op, rect)

        for row in op.get("layer_anchors", []) or []:
            m = models.get(int(row.get("system_index", 0)))
            if not m:
                continue
            x = float(row.get("cx_norm", 0.0)) * width
            for raw in row.get("levels", []) or []:
                try:
                    level = int(raw)
                except Exception:
                    continue
                y = _pdf_level_y(m, level)
                page.insert_text((x - 2.0, y + 2.0), str(level), fontsize=5.8, color=(0, 0, 0), overlay=True)
        for row in op.get("structural_anchors", []) or []:
            m = models.get(int(row.get("system_index", 0)))
            if not m:
                continue
            x = float(row.get("cx_norm", 0.0)) * width
            for raw in row.get("levels", []) or []:
                try:
                    level = int(raw)
                except Exception:
                    continue
                y = _pdf_level_y(m, level)
                page.insert_text((x - 3.0, y + 2.0), f"({level})", fontsize=5.2, color=(0.25, 0.25, 0.25), overlay=True)

        wave_by_system: dict[int, list[dict]] = {}
        for row in op.get("wave_anchors", []) or []:
            wave_by_system.setdefault(int(row.get("system_index", 0)), []).append(row)
        for sys_idx, rows in wave_by_system.items():
            m = models.get(sys_idx)
            if not m:
                continue
            rows.sort(key=lambda r: float(r.get("cx_norm", 0.0)))
            pts: list[fitz.Point] = []
            for row in rows:
                try:
                    x = float(row.get("cx_norm", 0.0)) * width
                    y = _pdf_level_y(m, int(row.get("height", 1) or 1))
                except Exception:
                    continue
                pts.append(fitz.Point(x, y))
            if len(pts) > 1:
                shape = page.new_shape()
                for a, b in zip(pts, pts[1:]):
                    shape.draw_line(a, b)
                shape.finish(color=(0.15, 0.15, 0.15), width=0.65)
                shape.commit(overlay=True)

        for row in op.get("pivot_anchors", []) or []:
            m = models.get(int(row.get("system_index", 0)))
            if not m:
                continue
            try:
                x = float(row.get("cx_norm", row.get("pivot_cx_norm", 0.0))) * width
                level = int(row.get("crest_height", row.get("height", 1)) or 1)
            except Exception:
                continue
            y = _pdf_level_y(m, level)
            page.draw_circle(fitz.Point(x, y), 1.7, color=(0, 0, 0), fill=(0.45, 0.45, 0.45), width=0.45, overlay=True)

        for row in op.get("tree_branches", []) or []:
            if row.get("span_role") not in (None, "", "complete"):
                continue
            m = models.get(int(row.get("system_index", 0)))
            if not m:
                continue
            try:
                x1 = float(row.get("source_cx_norm")) * width
                x2 = float(row.get("target_cx_norm")) * width
                y1 = _pdf_level_y(m, int(row.get("source_level", 1)))
                y2 = _pdf_level_y(m, int(row.get("target_level", 1)))
            except Exception:
                continue
            page.draw_line(fitz.Point(x1, y1), fitz.Point(x2, y2), color=(0.35, 0.35, 0.35), width=0.45, overlay=True)

    doc.save(out_pdf, garbage=3, deflate=True)
    doc.close()


def _build_visual_job(session_id: str, pdf_path: Path, initial_meter: str, analysis_payload: dict) -> None:
    session = CACHE / session_id
    try:
        _set_job(session_id, status="processing", stage="Rendering the exact uploaded PDF")
        pages = render_pdf_pages(pdf_path, session / "pages")
        public_pages = [
            {
                **p,
                "url": f"/api/visual-session/{session_id}/page/{p['index']}",
            }
            for p in pages
        ]
        _set_job(session_id, pages=public_pages, stage="Recovering score geometry with Audiveris")

        symbolic, omr_path = pdf_to_musicxml(pdf_path, session / "audiveris")
        override = base._meter_override(initial_meter)
        hits, measures, _parser_warnings = parse_musicxml(symbolic, initial_meter_override=override)
        _unused_hits, canonical_warnings, canonical_meta = build_hits_from_canonical_score(
            omr_path, measures, symbolic_hits=hits
        )

        _set_job(session_id, stage="Registering analysis to the original PDF pages")
        annotation_archive, annotation_warning = pdf_to_annotations(pdf_path, session / "annotations")
        if annotation_archive is None:
            raise RuntimeError(annotation_warning or "Audiveris did not produce a physical annotation archive.")

        visual_groups, layout_known = extract_visual_groups(symbolic, initial_meter_override=override)
        visual_levels = deepcopy(analysis_payload.get("levels") or {})
        visual_levels["canonical_score_meta"] = canonical_meta
        overlay = build_normalized_overlay(
            annotation_archive,
            visual_groups,
            layout_known,
            visual_levels,
            session / "physical",
            omr_path=omr_path,
        )

        registration = _registration_report(Path(annotation_archive), pages)
        warnings = list(overlay.get("warnings", []) or [])
        warnings.extend(str(w) for w in canonical_warnings if str(w).strip())
        if annotation_warning:
            warnings.append(str(annotation_warning))

        annotated_pdf = session / "score-with-tone-metric-analysis.pdf"
        _write_annotated_pdf(pdf_path, annotated_pdf, overlay)

        manifest = {
            "status": "ready",
            "session_id": session_id,
            "pages": public_pages,
            "overlay": overlay,
            "page_registration": registration,
            "warnings": list(dict.fromkeys(warnings)),
            "analysis_unchanged": True,
            "exact_pdf_background": True,
            "background_source": "uploaded-pdf",
            "annotated_pdf_url": f"/api/visual-session/{session_id}/annotated.pdf",
            "original_pdf_url": f"/api/visual-session/{session_id}/original.pdf",
        }
        (session / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _set_job(session_id, **manifest)
    except Exception as exc:
        _set_job(
            session_id,
            status="failed",
            stage="Original-PDF score overlay stopped",
            detail=str(exc),
            analysis_unchanged=True,
        )


for route in list(app.router.routes):
    if getattr(route, "path", None) == "/api/analyze" and "POST" in (getattr(route, "methods", set()) or set()):
        app.router.routes.remove(route)


@app.post("/api/analyze")
async def analyze_with_exact_pdf_layer(
    file: UploadFile = File(...),
    initial_meter: str = Form("auto"),
):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix != ".pdf":
        return await base.analyze_endpoint(file=file, initial_meter=initial_meter)

    _cleanup_cache()
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(payload) > 80 * 1024 * 1024:
        raise HTTPException(413, "Upload is larger than the 80 MB online limit.")

    session_id = uuid.uuid4().hex
    session = CACHE / session_id
    session.mkdir(parents=True, exist_ok=True)
    saved_pdf = session / filename
    saved_pdf.write_bytes(payload)

    await file.seek(0)
    try:
        response = await base.analyze_endpoint(file=file, initial_meter=initial_meter)
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        raise

    try:
        analysis_payload = json.loads(response.body.decode("utf-8"))
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        return response

    _set_job(
        session_id,
        status="queued",
        stage="Waiting to build original-PDF visualization",
        created_at=time.time(),
        pages=[],
        analysis_unchanged=True,
    )
    threading.Thread(
        target=_build_visual_job,
        args=(session_id, saved_pdf, initial_meter, analysis_payload),
        daemon=True,
        name=f"tone-metric-exact-pdf-{session_id[:8]}",
    ).start()

    response.headers["X-Tone-Metric-Visual-Session"] = session_id
    return response


@app.get("/api/visual-session/{session_id}/manifest")
def visual_manifest(session_id: str):
    if not session_id.isalnum():
        raise HTTPException(404, "Visual score session not found.")
    job = _get_job(session_id)
    if job is None:
        raise HTTPException(404, "Visual score session not found.")
    return job


@app.get("/api/visual-session/{session_id}/page/{page_index}")
def visual_page(session_id: str, page_index: int):
    if not session_id.isalnum() or page_index < 0:
        raise HTTPException(404, "Score page not found.")
    path = CACHE / session_id / "pages" / f"page_{page_index + 1}.png"
    if not path.exists():
        raise HTTPException(404, "Score page not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/visual-session/{session_id}/original.pdf")
def original_pdf(session_id: str):
    if not session_id.isalnum():
        raise HTTPException(404, "Score session not found.")
    folder = CACHE / session_id
    candidates = [p for p in folder.glob("*.pdf") if p.name != "score-with-tone-metric-analysis.pdf"]
    if not candidates:
        raise HTTPException(404, "Original PDF not found.")
    return FileResponse(candidates[0], media_type="application/pdf", filename=candidates[0].name)


@app.get("/api/visual-session/{session_id}/annotated.pdf")
def annotated_pdf(session_id: str):
    if not session_id.isalnum():
        raise HTTPException(404, "Score session not found.")
    path = CACHE / session_id / "score-with-tone-metric-analysis.pdf"
    if not path.exists():
        raise HTTPException(404, "Annotated PDF is not ready.")
    return FileResponse(path, media_type="application/pdf", filename="score-with-tone-metric-analysis.pdf")


_EXTRA_STYLE = r'''<style id="tm-exact-pdf-style">
.tm-exact-score{margin-top:14px}.tm-score-status{margin:8px 0 10px;color:#555;white-space:pre-wrap}.tm-score-toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:8px 0}.tm-score-toolbar button,.tm-score-toolbar a{font:inherit;padding:7px 9px;border-radius:7px;border:1px solid #aaa;text-decoration:none}.tm-score-toolbar button{background:#fff;color:#111}.tm-score-toolbar button.active{background:#111;color:#fff;border-color:#111}.tm-score-toolbar a{background:#555;color:#fff;border-color:#555;font-weight:700}.tm-score-stage{overflow:auto;border:1px solid #ddd;border-radius:10px;padding:8px;background:#ddd;max-height:86vh}.tm-score-surface{position:relative;width:max-content;transform-origin:top left;background:#fff}.tm-score-surface>img{display:block;max-width:none}.tm-score-surface>svg{position:absolute;inset:0;overflow:visible;pointer-events:none;background:transparent;border:0;border-radius:0}.tm-level{font:700 9px Arial,sans-serif;fill:#111;text-anchor:middle;dominant-baseline:middle;paint-order:stroke;stroke:#fff;stroke-width:2px;stroke-linejoin:round}.tm-struct{font:italic 8.5px Arial,sans-serif;fill:#333;text-anchor:middle;dominant-baseline:middle;paint-order:stroke;stroke:#fff;stroke-width:2px}.tm-level-guide{stroke:#111;stroke-width:.55;opacity:.18;stroke-dasharray:2 2}.tm-wave{fill:none;stroke:#111;stroke-width:1.15;opacity:.78;stroke-linejoin:miter}.tm-pivot-simple{fill:#888;stroke:#222;stroke-width:.4;opacity:.68}.tm-pivot-compound{fill:#222;stroke:#111;stroke-width:.45;opacity:.88}.tm-tree{fill:none;stroke:#555;stroke-width:.85;opacity:.66}.tm-tree-cont{fill:none;stroke:#777;stroke-width:.75;opacity:.58;stroke-dasharray:3 2}.tm-row-label{font:7.5px Arial,sans-serif;fill:#555;text-anchor:end}.tm-band-title{font:700 8px Arial,sans-serif;fill:#222}.tm-score-note{font-size:11px;color:#666}.tm-score-links{display:flex;gap:8px;flex-wrap:wrap}.tm-score-legend{font-size:11px;color:#555;min-height:18px;margin:5px 0}.tm-score-page-label{font-size:12px;font-weight:700;min-width:86px;text-align:center}.tm-score-zoom{display:flex;gap:5px;align-items:center;font-size:11px}.tm-score-zoom input{padding:0;width:120px}
</style>'''

_EXTRA_SCRIPT = r'''<script id="tm-exact-pdf-script">
(function(){
 const oldRender=render;
 render=function(d){oldRender(d);tmStartExactPdf(window.__tmVisualSession||'');};
 function ensurePanel(){
   let p=document.getElementById('tm-exact-score');if(p)return p;
   p=document.createElement('section');p.id='tm-exact-score';p.className='panel tm-exact-score';
   p.innerHTML='<h2>Original uploaded PDF + tone-metric analysis</h2><div id="tm-score-status" class="tm-score-status">Waiting for a PDF analysis.</div><div id="tm-score-toolbar" class="tm-score-toolbar" style="display:none"><button data-mode="attacks" class="active">Attacks</button><button data-mode="full">Full grid</button><button data-mode="waves">Waves</button><button data-mode="fullwave">Grid + waves</button><button data-mode="wavepivots">Waves + pivots</button><button data-mode="fullanalysis">Full analysis</button><button data-mode="trees">Trees</button><button data-mode="fulltree">Full + trees</button><button data-mode="none">Original PDF</button><button id="tm-page-prev">◀ Page</button><span id="tm-page-label" class="tm-score-page-label"></span><button id="tm-page-next">Page ▶</button><span class="tm-score-zoom">Zoom <input id="tm-score-zoom" type="range" min="50" max="220" step="10" value="100"><span id="tm-score-zoom-value">100%</span></span><span class="tm-score-links"><a id="tm-annotated-pdf" target="_blank">Open annotated PDF</a><a id="tm-original-pdf" target="_blank">Open original PDF</a></span></div><div id="tm-score-legend" class="tm-score-legend"></div><div id="tm-score-stage" class="tm-score-stage"><div id="tm-score-surface" class="tm-score-surface"></div></div>';
   document.querySelector('.wrap').appendChild(p);return p;
 }
 function svgEl(tag,attrs){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,String(v));return e}
 function groupsBySystem(rows){const m=new Map();for(const r of rows||[]){const k=Number(r.system_index||0);if(!m.has(k))m.set(k,[]);m.get(k).push(r)}return m}
 function buildModels(pp,w,h){const bounds=groupsBySystem((pp?.system_bounds||[]).map(b=>({...b,system_index:Number(b.system_index||0)})));const allKeys=new Set();for(const b of pp?.system_bounds||[])allKeys.add(Number(b.system_index||0));for(const key of ['layer_anchors','structural_anchors','wave_anchors','pivot_anchors','tree_nodes','tree_branches'])for(const r of pp?.[key]||[])allKeys.add(Number(r.system_index||0));const models=new Map();for(const k of allKeys){const b=(bounds.get(k)||[])[0]||{};const top=Number(b.top_norm??0)*h,bottom=Number(b.bottom_norm??b.top_norm??0)*h,left=Number(b.left_norm??0)*w,right=Number(b.right_norm??1)*w;let maxLevel=1;for(const key of ['layer_anchors','structural_anchors','wave_anchors','tree_nodes'])for(const r of pp?.[key]||[]){if(Number(r.system_index||0)!==k)continue;for(const x of r.levels||[]){const n=Number(x);if(Number.isFinite(n))maxLevel=Math.max(maxLevel,n)}for(const q of [r.height,r.lowest_level]){const n=Number(q);if(Number.isFinite(n))maxLevel=Math.max(maxLevel,n)}}models.set(k,{k,top,bottom,left,right,maxLevel})}const ordered=[...models.values()].sort((a,b)=>a.top-b.top);let prevBottom=0;for(const m of ordered){const avail=Math.max(12,m.top-prevBottom-8),rowH=Math.min(14,Math.max(5,avail/Math.max(2,m.maxLevel+1)));m.rowH=rowH;m.gridBottom=Math.max(prevBottom+7,m.top-8);m.gridTop=Math.max(prevBottom+5,m.gridBottom-rowH*Math.max(1,m.maxLevel-1));prevBottom=m.bottom}return models}
 function yLevel(m,l){return m.gridBottom-(Math.max(1,Number(l)||1)-1)*m.rowH}
 function addText(svg,x,y,value,cls,title){const e=svgEl('text',{x,y,class:cls});e.textContent=value;if(title){const t=svgEl('title');t.textContent=title;e.appendChild(t)}svg.appendChild(e)}
 function drawLayers(svg,pp,models,w,structural){const attacks=groupsBySystem(pp?.layer_anchors||[]),structs=groupsBySystem(pp?.structural_anchors||[]);const keys=new Set([...attacks.keys(),...(structural?[...structs.keys()]:[])]);for(const k of keys){const m=models.get(k);if(!m)continue;addText(svg,Math.max(8,m.left-5),Math.max(9,m.gridTop-4),'Levels','tm-band-title');for(let l=1;l<=m.maxLevel;l++)addText(svg,Math.max(7,m.left-7),yLevel(m,l),String(l),'tm-row-label');for(const r of attacks.get(k)||[]){const x=Number(r.cx_norm||0)*w;svg.appendChild(svgEl('line',{x1:x,y1:m.gridTop,x2:x,y2:m.top,class:'tm-level-guide'}));for(const l of r.levels||[])addText(svg,x,yLevel(m,l),String(l),'tm-level','m. '+(r.measure_number??'')+' · '+(r.offset_in_measure_quarter??'')+' · Level '+l)}if(structural)for(const r of structs.get(k)||[]){const x=Number(r.cx_norm||0)*w;svg.appendChild(svgEl('line',{x1:x,y1:m.gridTop,x2:x,y2:m.top,class:'tm-level-guide'}));for(const l of r.levels||[])addText(svg,x,yLevel(m,l),'('+l+')','tm-struct','m. '+(r.measure_number??'')+' · structural Level '+l+' · '+(r.structural_reason??''))}}}
 function drawWave(svg,pp,models,w){const by=groupsBySystem(pp?.wave_anchors||[]);for(const[k,rows0]of by){const m=models.get(k);if(!m)continue;const rows=[...rows0].sort((a,b)=>Number(a.cx_norm)-Number(b.cx_norm));const pts=[];for(const r of rows){const x=Number(r.cx_norm)*w,y=yLevel(m,Number(r.height||1));if(Number.isFinite(x)&&Number.isFinite(y))pts.push(x.toFixed(2)+','+y.toFixed(2))}if(pts.length>1)svg.appendChild(svgEl('polyline',{points:pts.join(' '),class:'tm-wave'}));addText(svg,Math.max(8,m.left-5),Math.max(9,m.gridTop-4),'Wave','tm-band-title')}}
 function drawPivots(svg,pp,models,w){for(const r of pp?.pivot_anchors||[]){const m=models.get(Number(r.system_index||0));if(!m)continue;const x=Number(r.cx_norm??r.pivot_cx_norm)*w,level=Number(r.crest_height||r.height||1),y=yLevel(m,level);if(!Number.isFinite(x)||!Number.isFinite(y))continue;const cls=String(r.kind||'').toLowerCase()==='compound'?'tm-pivot-compound':'tm-pivot-simple';const c=svgEl('circle',{cx:x,cy:y,r:4,class:cls});const t=svgEl('title');t.textContent=(r.kind||'pivot')+' pivot · m. '+(r.pivot_measure_number??'')+' · depth '+(r.drop_depth??'');c.appendChild(t);svg.appendChild(c)}}
 function drawTrees(svg,pp,models,w){for(const r of pp?.tree_branches||[]){const m=models.get(Number(r.system_index||0));if(!m)continue;const role=r.span_role||'complete';if(role==='complete'){const x1=Number(r.source_cx_norm)*w,x2=Number(r.target_cx_norm)*w,y1=yLevel(m,Number(r.source_level||1)),y2=yLevel(m,Number(r.target_level||1));if([x1,x2,y1,y2].every(Number.isFinite))svg.appendChild(svgEl('line',{x1,y1,x2,y2,class:'tm-tree'}))}else{const x=Number(r.cx_norm)*w,l=Number(r.endpoint_level||1),y=yLevel(m,l);if(!Number.isFinite(x)||!Number.isFinite(y))continue;const right=role==='source-endpoint',len=Math.max(10,m.rowH*1.7);svg.appendChild(svgEl('line',{x1:right?x:Math.max(m.left,x-len),y1:y,x2:right?Math.min(m.right,x+len):x,y2:y,class:'tm-tree-cont'}))}}}
 function renderPage(state){const page=state.pages[state.pageIndex],pp=(state.overlay.pages||[]).find(p=>Number(p.page_index)===Number(page.index))||{},w=Number(page.width),h=Number(page.height),surface=document.getElementById('tm-score-surface');const svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`,width:w,height:h,preserveAspectRatio:'none'});const models=buildModels(pp,w,h);const m=state.mode;const showLayers=['attacks','full','fullwave','fullanalysis','fulltree'].includes(m),structural=['full','fullwave','fullanalysis','fulltree'].includes(m),showWave=['waves','fullwave','wavepivots','fullanalysis','fulltree'].includes(m),showPivots=['wavepivots','fullanalysis','fulltree'].includes(m),showTrees=['trees','fulltree'].includes(m);if(showWave)drawWave(svg,pp,models,w);if(showLayers)drawLayers(svg,pp,models,w,structural);if(showPivots)drawPivots(svg,pp,models,w);if(showTrees)drawTrees(svg,pp,models,w);const img=document.createElement('img');img.src=page.url;img.alt='Exact uploaded PDF page '+(state.pageIndex+1);img.width=w;img.height=h;surface.innerHTML='';surface.style.width=w+'px';surface.style.height=h+'px';surface.appendChild(img);if(m!=='none')surface.appendChild(svg);surface.style.transform=`scale(${state.zoom/100})`;surface.style.marginRight=Math.max(0,(state.zoom/100-1)*w)+'px';surface.style.marginBottom=Math.max(0,(state.zoom/100-1)*h)+'px';document.getElementById('tm-page-label').textContent=`Page ${state.pageIndex+1} / ${state.pages.length}`;document.getElementById('tm-page-prev').disabled=state.pageIndex===0;document.getElementById('tm-page-next').disabled=state.pageIndex===state.pages.length-1;document.querySelectorAll('#tm-score-toolbar button[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));const legend={attacks:'Exact uploaded PDF + recursive Levels at actual attacks.',full:'Exact uploaded PDF + complete recursive grid; parenthetical labels are structural positions without new attacks.',waves:'Exact uploaded PDF + tone-metric wave.',fullwave:'Exact uploaded PDF + recursive grid + wave.',wavepivots:'Exact uploaded PDF + wave + simple/compound pivots.',fullanalysis:'Exact uploaded PDF + Levels + structural grid + wave + pivots.',trees:'Exact uploaded PDF + tone-metric tree.',fulltree:'Exact uploaded PDF + complete Levels / wave / pivots / tree analysis.',none:'Exact uploaded PDF with no analysis layer.'};document.getElementById('tm-score-legend').textContent=legend[m]||''}
 function wire(state){document.getElementById('tm-page-prev').onclick=()=>{if(state.pageIndex>0){state.pageIndex--;renderPage(state);document.getElementById('tm-score-stage').scrollTo({left:0,top:0})}};document.getElementById('tm-page-next').onclick=()=>{if(state.pageIndex<state.pages.length-1){state.pageIndex++;renderPage(state);document.getElementById('tm-score-stage').scrollTo({left:0,top:0})}};document.querySelectorAll('#tm-score-toolbar button[data-mode]').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;renderPage(state)});const z=document.getElementById('tm-score-zoom');z.oninput=()=>{state.zoom=Number(z.value);document.getElementById('tm-score-zoom-value').textContent=state.zoom+'%';renderPage(state)}}
 async function tmStartExactPdf(session){ensurePanel();const st=document.getElementById('tm-score-status'),toolbar=document.getElementById('tm-score-toolbar'),surface=document.getElementById('tm-score-surface');surface.innerHTML='';toolbar.style.display='none';if(!session){st.textContent='No PDF visualization session was created for this analysis.';return}for(let i=0;i<240;i++){try{const r=await fetch('/api/visual-session/'+session+'/manifest',{cache:'no-store'}),m=await r.json();if(!r.ok)throw Error(m.detail||'Could not read score visualization status.');if(m.status==='failed'){st.textContent='Analysis is complete. Original-PDF visualization stopped: '+(m.detail||'unknown error');return}if(m.status!=='ready'){st.textContent=(m.stage||'Building original-PDF visualization…')+'\nThe analytical result above is already complete and unchanged.';await new Promise(x=>setTimeout(x,1800));continue}toolbar.style.display='flex';document.getElementById('tm-annotated-pdf').href=m.annotated_pdf_url;document.getElementById('tm-original-pdf').href=m.original_pdf_url;const reg=m.page_registration||[],ok=reg.filter(x=>x.normalized_page_registration).length;st.textContent=`Ready. Displaying the exact uploaded PDF pages; analysis is a separate transparent layer. Page registration ${ok}/${reg.length}.`;const state={pages:m.pages||[],overlay:m.overlay||{},pageIndex:0,zoom:100,mode:'fullanalysis'};wire(state);renderPage(state);return}catch(e){st.textContent='Waiting for original-PDF visualization: '+e.message;await new Promise(x=>setTimeout(x,1800))}}st.textContent='Analysis is complete. The original-PDF visualization did not finish within the waiting window.'}
 window.tmStartExactPdf=tmStartExactPdf;ensurePanel();
})();
</script>'''

base.HTML = base.HTML.replace(
    "last=JSON.parse(t);render(last);",
    "window.__tmVisualSession=r.headers.get('X-Tone-Metric-Visual-Session')||'';last=JSON.parse(t);render(last);",
)
base.HTML = base.HTML.replace("</head>", _EXTRA_STYLE + "</head>")
base.HTML = base.HTML.replace("</body>", _EXTRA_SCRIPT + "</body>")
