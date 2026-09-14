from __future__ import annotations

import base64
import io
import tempfile
from fractions import Fraction
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw

import level1_clean_v5 as v5

core = v5.core
APP_VERSION = "0.6.0-level1-level2-omr-slot-aligned"
MAX_UPLOAD = v5.MAX_UPLOAD
PDF_ZOOM = v5.PDF_ZOOM

# ANALYTICAL CONTRACT
# -------------------
# Level 1 is not reimplemented here. This module calls the exact v0.4 Level-1
# function used by v5 and leaves result["points"] untouched.
#
# Level 2 is added according to the dissertation rule: it uses the same meter
# sequence as Level 1, but recursively restarts at each Level-1 boundary and is
# only instantiated for a Level-1 span that contains at least one still-unmapped
# tactus position. The terminal open span after the last Level-1 point is treated
# the same way under metric continuity.
#
# PDF registration is the exact v5 OMR-slot method. Level-1 x coordinates are
# computed by the same helpers as v5; Level 2 uses the same registration rule.
app = FastAPI(title="Tone-Metric Levels 1-2", version=APP_VERSION)


def _measure_for_time(measures: list[core.Measure], start_i: int, end_i: int, t: Fraction):
    return next(m for m in measures[start_i : end_i + 1] if m.start <= t < m.end)


def analyze_level2(
    measures: list[core.Measure],
    attacks: list[core.Attack],
    level1_result: dict,
) -> list[dict]:
    """Add only dissertation Level 2; do not alter Level 1.

    For each meter segment, consider adjacent Level-1 points as inclusive
    boundaries. If there is at least one tactus position between them, restart
    the meter's geometric sequence at the left boundary and place Level 2 at
    those sequence positions through the right boundary. After the final Level-1
    point, continue the same recursive procedure to the end of the segment.
    """
    attack_times = {a.onset for a in attacks}
    l1_by_segment: dict[int, list[Fraction]] = {}
    for p in level1_result.get("points", []):
        l1_by_segment.setdefault(int(p["segment_index"]), []).append(
            core.parse_fraction(p["time_quarter"])
        )

    level2_by_time: dict[tuple[int, Fraction], dict] = {}

    def add_point(si: int, a: int, b: int, profile, t: Fraction, seq_pos: int, span_start: Fraction, span_end: Fraction | None):
        if not (measures[a].start <= t < measures[b].end):
            return
        m = _measure_for_time(measures, a, b, t)
        is_attack = t in attack_times
        key = (si, t)
        if key in level2_by_time:
            return
        level2_by_time[key] = {
            "segment_index": si,
            "sequence_position": seq_pos,
            "time_quarter": core.ftxt(t),
            "measure_index": m.index,
            "measure_number": m.number,
            "beat": int((t - m.start) / profile.beat_unit) + 1,
            "level": 2,
            "attack": is_attack,
            "parenthetical": not is_attack,
            "label": "2" if is_attack else "(2)",
            "recursive_span_start_quarter": core.ftxt(span_start),
            "recursive_span_end_quarter": None if span_end is None else core.ftxt(span_end),
        }

    for si, (a, b) in enumerate(core.meter_segments(measures)):
        first, last = measures[a], measures[b]
        profile = core.PROFILES[first.meter]
        beat = profile.beat_unit
        seg_end = last.end
        boundaries = sorted(set(l1_by_segment.get(si, [])))
        if not boundaries:
            continue

        # Closed Level-1 spans. Adjacent boundaries need no Level 2 because
        # Level 1 has already mapped every tactus position in that span.
        for t0, t1 in zip(boundaries, boundaries[1:]):
            steps = int((t1 - t0) / beat)
            if steps <= 1:
                continue
            position_count = steps + 1  # inclusive of both Level-1 limits
            for pos in core.sequence(profile.arity, position_count):
                if pos > position_count:
                    break
                t = t0 + (pos - 1) * beat
                if t > t1:
                    break
                add_point(si, a, b, profile, t, pos, t0, t1)

        # Open final span under metric continuity. The segment end itself is
        # exclusive, so the number of available tactus positions is the number
        # of beat units from the final Level-1 point to seg_end.
        t0 = boundaries[-1]
        position_count = int((seg_end - t0) / beat)
        if position_count > 1:
            for pos in core.sequence(profile.arity, position_count):
                if pos > position_count:
                    break
                t = t0 + (pos - 1) * beat
                if not (t0 <= t < seg_end):
                    continue
                add_point(si, a, b, profile, t, pos, t0, None)

    return [level2_by_time[k] for k in sorted(level2_by_time, key=lambda x: (x[0], x[1]))]


def analyze_levels12(measures: list[core.Measure], attacks: list[core.Attack]) -> dict:
    # This exact call is the frozen Level-1 path from v4/v5.
    result = core.analyze_level1(measures, attacks)
    frozen_level1_points = result.get("points", [])
    level2_points = analyze_level2(measures, attacks, result)

    # Preserve the v5 backward contract: result["points"] remains Level 1 only.
    result["level1_points"] = frozen_level1_points
    result["level2_points"] = level2_points
    result["levels"] = {"1": frozen_level1_points, "2": level2_points}
    result["highest_level"] = 2
    result["engine_contract_levels12"] = "dissertation-level1-frozen-v0.4-plus-level2-recursive-v0.6"
    if "measures" not in result:
        result["measures"] = [
            {
                "index": m.index,
                "number": m.number,
                "start_quarter": core.ftxt(m.start),
                "end_quarter": core.ftxt(m.end),
                "meter": f"{m.meter[0]}/{m.meter[1]}",
            }
            for m in measures
        ]
    return result


def _draw_point(draw, x: int, row_top: int, row_bottom: int, label: str, font):
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    y = row_top + max(1, (row_bottom - row_top - th) // 2 - 1)
    draw.text((x - tw // 2, y), label, fill=(0, 0, 0), font=font)


def render_original_pdf_with_levels12_exact(
    pdf_path: Path,
    omr_path: Path,
    measures: list[core.Measure],
    result: dict,
) -> tuple[list[str], list[dict]]:
    """Keep v5 Level-1 registration and add a Level-2 row above it."""
    doc = fitz.open(str(pdf_path))
    geometry = v5.parse_omr_geometry(omr_path, len(measures))
    if len(geometry) != len(doc):
        doc.close()
        raise RuntimeError(
            f"OMR/PDF registration page mismatch: OMR has {len(geometry)} pages, PDF has {len(doc)}."
        )

    level_points = {
        1: result.get("level1_points", result.get("points", [])),
        2: result.get("level2_points", []),
    }
    points_by_level_measure: dict[int, dict[int, list[dict]]] = {1: {}, 2: {}}
    for level, pts in level_points.items():
        for point in pts:
            points_by_level_measure[level].setdefault(int(point["measure_index"]), []).append(point)

    pages_out: list[str] = []
    diagnostics: list[dict] = []

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_ZOOM, PDF_ZOOM), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image)
        geom = geometry[page_index]
        scale_x = image.width / float(geom["source_width"])
        scale_y = image.height / float(geom["source_height"])

        font_label = core._font(max(22, int(image.width / 72)), bold=True)
        font_title = core._font(max(18, int(image.width / 90)), bold=True)
        row_height = max(48, int(image.width / 34))
        page_diag = {
            "page_index": page_index,
            "registration": "audiveris-omr-source-pixel-slots",
            "level1_registration_unchanged_from_v5": True,
            "source_width": geom["source_width"],
            "source_height": geom["source_height"],
            "render_width": image.width,
            "render_height": image.height,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "systems": [],
        }

        for system in geom["systems"]:
            left = int(round(float(system["staff_left"]) * scale_x))
            right = int(round(float(system["staff_right"]) * scale_x))
            staff_top = int(round(float(system["staff_top"]) * scale_y))

            # Preserve the v5 Level-1 vertical band exactly; add L2 above it.
            l1_bottom = max(20, staff_top - 10)
            l1_top = max(4, l1_bottom - row_height)
            l2_bottom = l1_top
            l2_top = max(4, l2_bottom - row_height)

            draw.rounded_rectangle(
                [left, l1_top, right, l1_bottom],
                radius=6,
                fill=(242, 242, 242),
                outline=(180, 180, 180),
                width=1,
            )
            draw.rounded_rectangle(
                [left, l2_top, right, l2_bottom],
                radius=6,
                fill=(225, 225, 225),
                outline=(170, 170, 170),
                width=1,
            )

            label_x = max(4, left - int(image.width * 0.055))
            for level, top, bottom in ((1, l1_top, l1_bottom), (2, l2_top, l2_bottom)):
                title_y = top + max(2, (bottom - top - font_title.size) // 2)
                draw.text((label_x, title_y), f"L{level}", fill=(30, 30, 30), font=font_title)

            system_diag = {
                "system_index": system["system_index"],
                "measure_numbers": [],
                "levels": {
                    "1": {"attack_labels_omr_slot_aligned": 0, "attack_labels_without_exact_slot": 0, "structural_nonattack_labels": 0},
                    "2": {"attack_labels_omr_slot_aligned": 0, "attack_labels_without_exact_slot": 0, "structural_nonattack_labels": 0},
                },
                "staff_top": staff_top,
                "level1_top": l1_top,
                "level1_bottom": l1_bottom,
                "level2_top": l2_top,
                "level2_bottom": l2_bottom,
            }

            for stack in system["stacks"]:
                mi = int(stack["measure_index"])
                if not (0 <= mi < len(measures)):
                    continue
                measure = measures[mi]
                system_diag["measure_numbers"].append(measure.number)
                duration_whole = Fraction(measure.end - measure.start, 4)

                for level, row_top, row_bottom in ((1, l1_top, l1_bottom), (2, l2_top, l2_bottom)):
                    for point in points_by_level_measure[level].get(mi, []):
                        t = core.parse_fraction(point["time_quarter"])
                        local_quarter = t - measure.start - measure.pickup_shift
                        local_whole = Fraction(local_quarter, 4)

                        x_omr = None
                        if bool(point.get("attack")):
                            x_omr = v5._nearest_slot_x(stack, local_whole)
                            if x_omr is not None:
                                system_diag["levels"][str(level)]["attack_labels_omr_slot_aligned"] += 1
                            else:
                                system_diag["levels"][str(level)]["attack_labels_without_exact_slot"] += 1
                        else:
                            system_diag["levels"][str(level)]["structural_nonattack_labels"] += 1

                        if x_omr is None:
                            x_omr = v5._interpolated_x(stack, local_whole, duration_whole)
                        x = int(round(float(x_omr) * scale_x))
                        _draw_point(draw, x, row_top, row_bottom, str(point["label"]), font_label)

            page_diag["systems"].append(system_diag)

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        pages_out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        diagnostics.append(page_diag)

    doc.close()
    return pages_out, diagnostics


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Tone-Metric Levels 1–2</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;color:#111;background:#fff}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}.note{font-size:12px;color:#444;margin-top:8px}#score{margin-top:28px}.score-title{font-size:20px;font-weight:700;margin:0 0 8px}.score-page{display:block;max-width:100%;height:auto;margin:18px auto;border:1px solid #ccc;box-shadow:0 2px 10px rgba(0,0,0,.08);background:white}.score-note{font-size:13px;color:#444;margin-bottom:12px}table{border-collapse:collapse;margin-top:18px;font-size:13px}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}
</style></head><body>
<h1>Tone-Metric Analyzer — Levels 1 and 2</h1>
<div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Opening meter<select id="meter"><option selected>4/4</option><option>2/2</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze Levels 1–2</button></div>
<div id="status">No analysis yet.</div>
<div class="note">Level 1 is the frozen clean v0.4 analysis with the exact v5 OMR alignment. Level 2 is added recursively above it; no prior Level-1 behavior is replaced.</div>
<div id="table"></div><div id="score"></div>
<script>
const $=id=>document.getElementById(id);
function render(d){
 const l1=d.level1_points||d.points||[], l2=d.level2_points||[];
 let h='<table><thead><tr><th>Level</th><th>Sequence pos.</th><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Label</th><th>New attack?</th></tr></thead><tbody>';
 for(const [lev,pts] of [[1,l1],[2,l2]])for(const p of pts)h+=`<tr><td>${lev}</td><td>${p.sequence_position}</td><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;
 h+='</tbody></table>';$('table').innerHTML=h;
 const pages=d.pdf_pages||[];if(pages.length){let q='<div class="score-title">Original uploaded PDF + Levels 1–2</div><div class="score-note">L1 retains the previous exact OMR note/chord registration. L2 uses the same OMR slot registration and is drawn immediately above L1.</div>';pages.forEach((src,i)=>q+=`<img class="score-page" src="${src}" alt="Analyzed score page ${i+1}">`);$('score').innerHTML=q}else{$('score').innerHTML=''}
}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Analyzing Levels 1–2…';$('table').innerHTML='';$('score').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Complete — ${d.measure_count} measures, ${(d.level1_points||[]).length} L1 positions, ${(d.level2_points||[]).length} L2 positions.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}};
</script></body></html>'''


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "mode": "frozen-level1-plus-dissertation-level2-exact-omr-registration",
        "level1_analysis_runtime": core.APP_VERSION,
        "level1_registration_runtime": v5.APP_VERSION,
        "level1_unchanged": True,
        "pdf_overlay": "audiveris-omr-source-pixel-slot-aligned-levels-1-2",
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), initial_meter: str = Form("4/4")):
    filename = Path(file.filename or "score").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".mxl", ".musicxml", ".xml"}:
        raise HTTPException(400, "Upload PDF, MXL, MusicXML, or XML.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Upload exceeds 80 MB.")
    try:
        meter = core.parse_meter(initial_meter)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="tm-level12-v6-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            omr_path = None
            if suffix == ".pdf":
                audiveris_out = work / "audiveris"
                symbolic = core.pdf_to_musicxml(source, audiveris_out)
                omr_path = v5.find_omr(audiveris_out, symbolic)
            else:
                symbolic = source

            measures, attacks = core.parse_score(symbolic, meter)
            result = analyze_levels12(measures, attacks)
            result["version"] = APP_VERSION
            result["level1_analysis_version"] = core.APP_VERSION
            result["level1_registration_version"] = v5.APP_VERSION
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "opening_meter": f"{meter[0]}/{meter[1]}",
            }

            if suffix == ".pdf" and omr_path is not None:
                pages, diagnostics = render_original_pdf_with_levels12_exact(
                    source, omr_path, measures, result
                )
                result["pdf_pages"] = pages
                result["pdf_overlay_diagnostics"] = diagnostics
            else:
                result["pdf_pages"] = []
                result["pdf_overlay_diagnostics"] = []
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Levels 1–2 analysis failed: {exc}") from exc
