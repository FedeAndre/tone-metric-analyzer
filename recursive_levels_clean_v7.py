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

import level1_level2_clean_v6 as v6

core = v6.core
v5 = v6.v5
APP_VERSION = "0.7.0-recursive-tactus-levels-omr-slot-aligned"
MAX_UPLOAD = v6.MAX_UPLOAD
PDF_ZOOM = v6.PDF_ZOOM

# ANALYTICAL CONTRACT
# -------------------
# 1. Level 1 is the exact frozen v0.4 implementation used by v5/v6.
# 2. Level 2 is the exact v0.6 implementation used by v6.
# 3. Levels 3+ repeat the same dissertation recursion used for Level 2:
#    the previous level supplies the boundaries; within each previous-level
#    span that still contains an unmapped tactus position, the meter's same
#    geometric sequence restarts at the left boundary. The operation repeats
#    until every tactus position in the meter segment is mapped.
# 4. This v7 stage generalizes the recursive LEVEL architecture at the current
#    tactus denomination only. It does not yet introduce finer rhythmic
#    denominations (eighths, sixteenths, etc.); those are a separate subsequent
#    dissertation step and must not be silently mixed into this regression.
# 5. OMR/PDF registration is unchanged from v5/v6.
app = FastAPI(title="Tone-Metric Recursive Levels", version=APP_VERSION)


def _measure_for_time(measures: list[core.Measure], a: int, b: int, t: Fraction):
    return next(m for m in measures[a : b + 1] if m.start <= t < m.end)


def _group_times_by_segment(points: list[dict]) -> dict[int, list[Fraction]]:
    out: dict[int, list[Fraction]] = {}
    for p in points:
        out.setdefault(int(p["segment_index"]), []).append(core.parse_fraction(p["time_quarter"]))
    return {si: sorted(set(times)) for si, times in out.items()}


def _all_tactus_times(measures: list[core.Measure]) -> dict[int, set[Fraction]]:
    out: dict[int, set[Fraction]] = {}
    for si, (a, b) in enumerate(core.meter_segments(measures)):
        profile = core.PROFILES[measures[a].meter]
        beat = profile.beat_unit
        start = measures[a].start
        end = measures[b].end
        count = int((end - start) / beat)
        out[si] = {start + k * beat for k in range(count)}
    return out


def analyze_next_recursive_level(
    measures: list[core.Measure],
    attacks: list[core.Attack],
    previous_points: list[dict],
    covered_before: dict[int, set[Fraction]],
    level_number: int,
) -> list[dict]:
    """Generate one dissertation-recursive level from the immediately prior one.

    The previous level defines the recursive boundaries. A new level is created
    only in a previous-level span containing at least one tactus point not yet
    mapped by any lower level. The meter-specific geometric sequence restarts at
    the left boundary of that span. Boundaries themselves are repeated on the new
    level exactly as in the dissertation's Levels 2 and 3 examples.
    """
    if level_number < 3:
        raise ValueError("This helper is for Levels 3 and above; Level 2 stays frozen in v6.")

    attack_times = {a.onset for a in attacks}
    previous_by_segment = _group_times_by_segment(previous_points)
    next_by_time: dict[tuple[int, Fraction], dict] = {}

    def add_point(
        si: int,
        a: int,
        b: int,
        profile,
        t: Fraction,
        seq_pos: int,
        span_start: Fraction,
        span_end: Fraction | None,
    ) -> None:
        if not (measures[a].start <= t < measures[b].end):
            return
        key = (si, t)
        if key in next_by_time:
            return
        m = _measure_for_time(measures, a, b, t)
        is_attack = t in attack_times
        next_by_time[key] = {
            "segment_index": si,
            "sequence_position": seq_pos,
            "time_quarter": core.ftxt(t),
            "measure_index": m.index,
            "measure_number": m.number,
            "beat": int((t - m.start) / profile.beat_unit) + 1,
            "level": level_number,
            "attack": is_attack,
            "parenthetical": not is_attack,
            "label": str(level_number) if is_attack else f"({level_number})",
            "recursive_span_start_quarter": core.ftxt(span_start),
            "recursive_span_end_quarter": None if span_end is None else core.ftxt(span_end),
        }

    for si, (a, b) in enumerate(core.meter_segments(measures)):
        profile = core.PROFILES[measures[a].meter]
        beat = profile.beat_unit
        seg_end = measures[b].end
        boundaries = previous_by_segment.get(si, [])
        if not boundaries:
            continue
        covered = covered_before.get(si, set())

        # Closed previous-level spans. The new level is needed only when a
        # tactus position strictly inside the span is still uncovered.
        for t0, t1 in zip(boundaries, boundaries[1:]):
            steps = int((t1 - t0) / beat)
            if steps <= 1:
                continue
            interior = [t0 + k * beat for k in range(1, steps)]
            if not any(t not in covered for t in interior):
                continue
            position_count = steps + 1  # both previous-level limits included
            for pos in core.sequence(profile.arity, position_count):
                if pos > position_count:
                    break
                t = t0 + (pos - 1) * beat
                if t > t1:
                    break
                add_point(si, a, b, profile, t, pos, t0, t1)

        # Open terminal span under metric continuity, matching the frozen v6 L2
        # treatment. The segment end is exclusive.
        t0 = boundaries[-1]
        position_count = int((seg_end - t0) / beat)
        if position_count > 1:
            interior = [t0 + k * beat for k in range(1, position_count)]
            if any(t not in covered for t in interior):
                for pos in core.sequence(profile.arity, position_count):
                    if pos > position_count:
                        break
                    t = t0 + (pos - 1) * beat
                    if t0 <= t < seg_end:
                        add_point(si, a, b, profile, t, pos, t0, None)

    return [next_by_time[k] for k in sorted(next_by_time, key=lambda x: (x[0], x[1]))]


def analyze_recursive_levels(measures: list[core.Measure], attacks: list[core.Attack]) -> dict:
    """Preserve frozen L1/L2 and recursively add as many higher tactus levels as needed."""
    result = v6.analyze_levels12(measures, attacks)

    # Non-negotiable frozen outputs from the validated prior stages.
    l1 = result["level1_points"]
    l2 = result["level2_points"]
    levels: dict[str, list[dict]] = {"1": l1, "2": l2}

    all_tactus = _all_tactus_times(measures)
    covered: dict[int, set[Fraction]] = {si: set() for si in all_tactus}
    for pts in (l1, l2):
        for p in pts:
            covered.setdefault(int(p["segment_index"]), set()).add(core.parse_fraction(p["time_quarter"]))

    def complete() -> bool:
        return all(all_tactus.get(si, set()) <= covered.get(si, set()) for si in all_tactus)

    previous = l2
    level_number = 3
    while not complete():
        if not previous:
            missing = sum(len(all_tactus[si] - covered.get(si, set())) for si in all_tactus)
            raise RuntimeError(
                f"Recursive level construction stalled with {missing} unmapped tactus positions."
            )
        next_points = analyze_next_recursive_level(
            measures, attacks, previous, covered, level_number
        )
        if not next_points:
            missing = sum(len(all_tactus[si] - covered.get(si, set())) for si in all_tactus)
            raise RuntimeError(
                f"Recursive level construction made no progress at Level {level_number}; "
                f"{missing} tactus positions remain unmapped."
            )

        before_count = sum(len(v) for v in covered.values())
        for p in next_points:
            covered.setdefault(int(p["segment_index"]), set()).add(core.parse_fraction(p["time_quarter"]))
        after_count = sum(len(v) for v in covered.values())
        if after_count <= before_count:
            raise RuntimeError(f"Recursive Level {level_number} added no newly mapped tactus position.")

        levels[str(level_number)] = next_points
        previous = next_points
        level_number += 1

    highest = max(int(k) for k, pts in levels.items() if pts) if any(levels.values()) else 1
    result["levels"] = levels
    result["highest_level"] = highest
    result["recursive_levels_complete"] = True
    result["recursive_scope"] = "tactus-denomination-only"
    result["engine_contract_recursive"] = (
        "frozen-level1-v0.4+frozen-level2-v0.6+recursive-higher-levels-v0.7"
    )
    # Convenience aliases without changing the frozen v6 keys.
    for k, pts in levels.items():
        result[f"level{k}_points"] = pts
    return result


def _draw_point(draw, x: int, row_top: int, row_bottom: int, label: str, font):
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    y = row_top + max(1, (row_bottom - row_top - th) // 2 - 1)
    draw.text((x - tw // 2, y), label, fill=(0, 0, 0), font=font)


def render_original_pdf_with_recursive_levels_exact(
    pdf_path: Path,
    omr_path: Path,
    measures: list[core.Measure],
    result: dict,
) -> tuple[list[str], list[dict]]:
    """Render arbitrary recursive levels while preserving v5/v6 registration rules."""
    doc = fitz.open(str(pdf_path))
    geometry = v5.parse_omr_geometry(omr_path, len(measures))
    if len(geometry) != len(doc):
        doc.close()
        raise RuntimeError(
            f"OMR/PDF registration page mismatch: OMR has {len(geometry)} pages, PDF has {len(doc)}."
        )

    levels: dict[int, list[dict]] = {
        int(k): v for k, v in result.get("levels", {}).items() if v
    }
    highest = max(levels) if levels else 1
    points_by_level_measure: dict[int, dict[int, list[dict]]] = {
        level: {} for level in range(1, highest + 1)
    }
    for level, pts in levels.items():
        for p in pts:
            points_by_level_measure[level].setdefault(int(p["measure_index"]), []).append(p)

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
            "level2_registration_unchanged_from_v6": True,
            "highest_level": highest,
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

            # L1 is kept at the exact v5 vertical location. L2 stays directly
            # above it as in v6. Higher levels simply continue the same stack.
            l1_bottom = max(20, staff_top - 10)
            l1_top = max(4, l1_bottom - row_height)
            rows: dict[int, tuple[int, int]] = {1: (l1_top, l1_bottom)}
            for level in range(2, highest + 1):
                lower_top = rows[level - 1][0]
                rows[level] = (max(4, lower_top - row_height), lower_top)

            fills = [(242, 242, 242), (225, 225, 225), (208, 208, 208), (191, 191, 191),
                     (174, 174, 174), (157, 157, 157), (140, 140, 140), (123, 123, 123),
                     (106, 106, 106), (89, 89, 89)]
            label_x = max(4, left - int(image.width * 0.055))
            for level in range(1, highest + 1):
                top, bottom = rows[level]
                fill = fills[min(level - 1, len(fills) - 1)]
                draw.rounded_rectangle(
                    [left, top, right, bottom], radius=6, fill=fill,
                    outline=(170, 170, 170), width=1
                )
                title_y = top + max(2, (bottom - top - font_title.size) // 2)
                draw.text((label_x, title_y), f"L{level}", fill=(30, 30, 30), font=font_title)

            system_diag = {
                "system_index": system["system_index"],
                "measure_numbers": [],
                "levels": {
                    str(level): {
                        "attack_labels_omr_slot_aligned": 0,
                        "attack_labels_without_exact_slot": 0,
                        "structural_nonattack_labels": 0,
                    }
                    for level in range(1, highest + 1)
                },
                "staff_top": staff_top,
                "rows": {str(level): {"top": rows[level][0], "bottom": rows[level][1]} for level in rows},
            }

            for stack in system["stacks"]:
                mi = int(stack["measure_index"])
                if not (0 <= mi < len(measures)):
                    continue
                measure = measures[mi]
                system_diag["measure_numbers"].append(measure.number)
                duration_whole = Fraction(measure.end - measure.start, 4)

                for level in range(1, highest + 1):
                    row_top, row_bottom = rows[level]
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
<html><head><meta charset="utf-8"><title>Tone-Metric Recursive Levels</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;color:#111;background:#fff}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px}button{padding:8px 16px}#status{margin:14px 0;font-weight:600}.note{font-size:12px;color:#444;margin-top:8px}#score{margin-top:28px}.score-title{font-size:20px;font-weight:700;margin:0 0 8px}.score-page{display:block;max-width:100%;height:auto;margin:18px auto;border:1px solid #ccc;box-shadow:0 2px 10px rgba(0,0,0,.08);background:white}.score-note{font-size:13px;color:#444;margin-bottom:12px}table{border-collapse:collapse;margin-top:18px;font-size:13px}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left}
</style></head><body>
<h1>Tone-Metric Analyzer — Recursive Levels</h1>
<div class="row"><label>Score<input id="file" type="file" accept=".pdf,.mxl,.musicxml,.xml"></label><label>Opening meter<select id="meter"><option selected>4/4</option><option>2/2</option><option>3/4</option><option>6/8</option><option>9/8</option><option>12/8</option></select></label><button id="go">Analyze Recursive Levels</button></div>
<div id="status">No analysis yet.</div>
<div class="note">Levels 1 and 2 are frozen from the validated previous builds. Higher levels repeat the same dissertation recursion until every tactus position is mapped. Finer rhythmic denominations are not introduced in this stage.</div>
<div id="table"></div><div id="score"></div>
<script>
const $=id=>document.getElementById(id);
function render(d){
 const levels=d.levels||{};
 let h='<table><thead><tr><th>Level</th><th>Sequence pos.</th><th>Measure</th><th>Beat</th><th>Quarter-time</th><th>Label</th><th>New attack?</th></tr></thead><tbody>';
 for(let lev=1;lev<=d.highest_level;lev++)for(const p of (levels[String(lev)]||[]))h+=`<tr><td>${lev}</td><td>${p.sequence_position??''}</td><td>${p.measure_number}</td><td>${p.beat}</td><td>${p.time_quarter}</td><td>${p.label}</td><td>${p.attack?'yes':'no'}</td></tr>`;
 h+='</tbody></table>';$('table').innerHTML=h;
 const pages=d.pdf_pages||[];if(pages.length){let q=`<div class="score-title">Original uploaded PDF + recursive Levels 1–${d.highest_level}</div><div class="score-note">L1/L2 are unchanged. Every attack label uses the same exact OMR note/chord-slot registration.</div>`;pages.forEach((src,i)=>q+=`<img class="score-page" src="${src}" alt="Analyzed score page ${i+1}">`);$('score').innerHTML=q}else{$('score').innerHTML=''}
}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a score.';return}const fd=new FormData();fd.append('file',f);fd.append('initial_meter',$('meter').value);$('status').textContent='Building recursive Levels…';$('table').innerHTML='';$('score').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const t=await r.text();if(!r.ok)throw new Error(t);const d=JSON.parse(t);$('status').textContent=`Complete — ${d.measure_count} measures, Levels 1–${d.highest_level}, tactus mapping complete.`;render(d)}catch(e){$('status').textContent='Analysis stopped: '+e.message}};
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
        "mode": "frozen-levels-1-2-plus-generalized-recursive-tactus-levels",
        "level1_analysis_runtime": core.APP_VERSION,
        "level1_registration_runtime": v5.APP_VERSION,
        "level2_runtime": v6.APP_VERSION,
        "level1_unchanged": True,
        "level2_unchanged": True,
        "recursive_scope": "tactus-denomination-only",
        "pdf_overlay": "audiveris-omr-source-pixel-slot-aligned-recursive-levels",
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

    with tempfile.TemporaryDirectory(prefix="tm-recursive-v7-") as tmp:
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
            result = analyze_recursive_levels(measures, attacks)
            result["version"] = APP_VERSION
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "opening_meter": f"{meter[0]}/{meter[1]}",
            }

            if suffix == ".pdf" and omr_path is not None:
                pages, diagnostics = render_original_pdf_with_recursive_levels_exact(
                    source, omr_path, measures, result
                )
                result["pdf_pages"] = pages
                result["pdf_overlay_diagnostics"] = diagnostics
            else:
                result["pdf_pages"] = []
                result["pdf_overlay_diagnostics"] = []
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Recursive-level analysis failed: {exc}") from exc
