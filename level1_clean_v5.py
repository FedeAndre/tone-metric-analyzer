from __future__ import annotations

import base64
import io
import re
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from lxml import etree
from PIL import Image, ImageDraw

import level1_clean_v4 as core

APP_VERSION = "0.5.0-level1-omr-slot-aligned"
MAX_UPLOAD = core.MAX_UPLOAD
PDF_ZOOM = core.PDF_ZOOM

# ANALYTICAL CONTRACT:
# The Level-1 algorithm remains exactly the v0.4 clean-room implementation.
# This runtime only replaces the PDF registration/display layer. PDF attack
# labels are registered to Audiveris OMR time slots, which are expressed in
# the source page's own pixel coordinate system.
app = FastAPI(title="Tone-Metric Level 1", version=APP_VERSION)


def _sheet_number(path: str) -> int:
    m = re.search(r"sheet#(\d+)/sheet#\1\.xml$", path)
    return int(m.group(1)) if m else 10**9


def find_omr(out_dir: Path, symbolic_path: Path | None = None) -> Path:
    candidates = sorted(out_dir.rglob("*.omr"))
    if not candidates:
        raise RuntimeError("Audiveris completed but no OMR save file was found for exact PDF registration.")
    if symbolic_path is not None:
        stem = symbolic_path.stem.lower()
        matching = [p for p in candidates if p.stem.lower() == stem]
        if matching:
            candidates = matching
    candidates.sort(key=lambda p: (-p.stat().st_size, len(str(p))))
    return candidates[0]


def parse_omr_geometry(omr_path: Path, measure_count: int) -> list[dict]:
    """Read exact source-page/system/measure/slot coordinates from Audiveris OMR.

    The OMR stack coordinates are the registration authority for the PDF overlay.
    They are not used by the Level-1 analytical algorithm.
    """
    pages: list[dict] = []
    measure_index = 0

    with zipfile.ZipFile(omr_path) as zf:
        sheet_xmls = sorted(
            [n for n in zf.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n)],
            key=_sheet_number,
        )
        if not sheet_xmls:
            raise RuntimeError("OMR save file contains no sheet XML geometry.")

        for page_index, name in enumerate(sheet_xmls):
            root = etree.fromstring(zf.read(name))
            picture = next((n for n in root if core.lname(n.tag) == "picture"), None)
            page = next((n for n in root if core.lname(n.tag) == "page"), None)
            if picture is None or page is None:
                raise RuntimeError(f"OMR sheet {page_index + 1} is missing picture/page geometry.")

            picture_width = float(picture.get("width") or 0)
            picture_height = float(picture.get("height") or 0)
            if picture_width <= 0 or picture_height <= 0:
                raise RuntimeError(f"OMR sheet {page_index + 1} has invalid source dimensions.")

            systems: list[dict] = []
            for system_index, system in enumerate(
                [n for n in page if core.lname(n.tag) == "system"]
            ):
                staff_x: list[float] = []
                staff_y: list[float] = []
                for node in system.iter():
                    if core.lname(node.tag) == "staff":
                        try:
                            staff_x.extend([float(node.get("left")), float(node.get("right"))])
                        except Exception:
                            pass
                        for point in node.iter():
                            if core.lname(point.tag) != "point":
                                continue
                            try:
                                staff_y.append(float(point.get("y")))
                            except Exception:
                                pass

                stacks: list[dict] = []
                for stack in [n for n in system if core.lname(n.tag) == "stack"]:
                    if measure_index >= measure_count:
                        break
                    left = float(stack.get("left") or 0.0)
                    right = float(stack.get("right") or left)
                    slots: dict[Fraction, float] = {}
                    for slot in [n for n in stack if core.lname(n.tag) == "slot"]:
                        raw_time = slot.get("time-offset")
                        raw_x = slot.get("x-offset")
                        if raw_time is None or raw_x is None:
                            continue
                        try:
                            slots[Fraction(raw_time)] = left + float(raw_x)
                        except Exception:
                            continue
                    stacks.append(
                        {
                            "measure_index": measure_index,
                            "left": left,
                            "right": right,
                            "slots": slots,
                        }
                    )
                    measure_index += 1

                if not stacks:
                    continue
                if not staff_x:
                    staff_x = [min(s["left"] for s in stacks), max(s["right"] for s in stacks)]
                if not staff_y:
                    raise RuntimeError(
                        f"OMR sheet {page_index + 1}, system {system_index + 1} has no staff-line coordinates."
                    )
                systems.append(
                    {
                        "system_index": system_index,
                        "staff_left": min(staff_x),
                        "staff_right": max(staff_x),
                        "staff_top": min(staff_y),
                        "staff_bottom": max(staff_y),
                        "stacks": stacks,
                    }
                )

            pages.append(
                {
                    "page_index": page_index,
                    "source_width": picture_width,
                    "source_height": picture_height,
                    "systems": systems,
                }
            )

    if measure_index != measure_count:
        raise RuntimeError(
            f"OMR/PDF registration has {measure_index} measure stacks but analysis has {measure_count} measures."
        )
    return pages


def _nearest_slot_x(stack: dict, local_whole: Fraction) -> float | None:
    slots: dict[Fraction, float] = stack.get("slots", {})
    if local_whole in slots:
        return float(slots[local_whole])
    return None


def _interpolated_x(stack: dict, local_whole: Fraction, duration_whole: Fraction) -> float:
    """Display-only fallback for structural Level-1 positions with no attack."""
    left = float(stack["left"])
    right = float(stack["right"])
    if duration_whole <= 0:
        return left
    t = max(Fraction(0), min(duration_whole, local_whole))

    anchors: list[tuple[Fraction, float]] = [(Fraction(0), left)]
    anchors.extend(sorted((Fraction(k), float(v)) for k, v in stack.get("slots", {}).items()))
    anchors.append((duration_whole, right))

    # Collapse duplicate time anchors while preferring an actual OMR slot.
    merged: dict[Fraction, float] = {}
    for tt, xx in anchors:
        merged[tt] = xx
    ordered = sorted(merged.items())
    if t <= ordered[0][0]:
        return ordered[0][1]
    if t >= ordered[-1][0]:
        return ordered[-1][1]
    for (t0, x0), (t1, x1) in zip(ordered, ordered[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return x0
            frac = float((t - t0) / (t1 - t0))
            return x0 + frac * (x1 - x0)
    return left + float(t / duration_whole) * (right - left)


def render_original_pdf_with_level1_exact(
    pdf_path: Path,
    omr_path: Path,
    measures: list[core.Measure],
    result: dict,
) -> tuple[list[str], list[dict]]:
    """Render the exact uploaded PDF with Level-1 labels registered to OMR slots.

    For every Level-1 point that is a real attack, the x coordinate is taken
    directly from the Audiveris OMR slot for that score time. Thus the label is
    centered on the corresponding note/chord column instead of being estimated
    from proportional measure spacing.
    """
    doc = fitz.open(str(pdf_path))
    geometry = parse_omr_geometry(omr_path, len(measures))
    if len(geometry) != len(doc):
        doc.close()
        raise RuntimeError(
            f"OMR/PDF registration page mismatch: OMR has {len(geometry)} pages, PDF has {len(doc)}."
        )

    points_by_measure: dict[int, list[dict]] = {}
    for point in result.get("points", []):
        points_by_measure.setdefault(int(point["measure_index"]), []).append(point)

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
        page_diag = {
            "page_index": page_index,
            "registration": "audiveris-omr-source-pixel-slots",
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
            strip_bottom = max(20, staff_top - 10)
            strip_top = max(4, strip_bottom - max(48, int(image.width / 34)))

            draw.rounded_rectangle(
                [left, strip_top, right, strip_bottom],
                radius=6,
                fill=(242, 242, 242),
                outline=(180, 180, 180),
                width=1,
            )
            label_x = max(4, left - int(image.width * 0.055))
            title_y = strip_top + max(2, (strip_bottom - strip_top - font_title.size) // 2)
            draw.text((label_x, title_y), "L1", fill=(30, 30, 30), font=font_title)

            aligned_attacks = 0
            unmatched_attacks = 0
            structural_labels = 0
            measure_numbers: list[str] = []

            for stack in system["stacks"]:
                mi = int(stack["measure_index"])
                if not (0 <= mi < len(measures)):
                    continue
                measure = measures[mi]
                measure_numbers.append(measure.number)
                duration_whole = Fraction(measure.end - measure.start, 4)

                for point in points_by_measure.get(mi, []):
                    t = core.parse_fraction(point["time_quarter"])
                    local_quarter = t - measure.start - measure.pickup_shift
                    local_whole = Fraction(local_quarter, 4)

                    x_omr = None
                    if bool(point.get("attack")):
                        x_omr = _nearest_slot_x(stack, local_whole)
                        if x_omr is not None:
                            aligned_attacks += 1
                        else:
                            unmatched_attacks += 1
                    else:
                        structural_labels += 1

                    if x_omr is None:
                        x_omr = _interpolated_x(stack, local_whole, duration_whole)
                    x = int(round(float(x_omr) * scale_x))

                    label = str(point["label"])
                    bbox = draw.textbbox((0, 0), label, font=font_label)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    y = strip_top + max(1, (strip_bottom - strip_top - th) // 2 - 1)
                    draw.text((x - tw // 2, y), label, fill=(0, 0, 0), font=font_label)

            page_diag["systems"].append(
                {
                    "system_index": system["system_index"],
                    "measure_numbers": measure_numbers,
                    "attack_labels_omr_slot_aligned": aligned_attacks,
                    "attack_labels_without_exact_slot": unmatched_attacks,
                    "structural_nonattack_labels": structural_labels,
                    "staff_top": staff_top,
                    "analysis_top": strip_top,
                    "analysis_bottom": strip_bottom,
                }
            )

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        pages_out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        diagnostics.append(page_diag)

    doc.close()
    return pages_out, diagnostics


HTML = core.HTML.replace(
    "Clean Level-1 runtime only. The analytical engine is unchanged; the PDF view is a separate display layer using the exact uploaded PDF pages.",
    "Clean Level-1 analytical runtime unchanged. PDF attack labels are registered directly to Audiveris OMR note/chord time slots from the exact uploaded PDF pages.",
).replace(
    "Each Level-1 strip is positioned immediately above its corresponding printed system.",
    "Each Level-1 attack label is centered on the exact OMR note/chord column for that score time.",
)


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
        "mode": "level1-v04-analysis-plus-exact-omr-slot-pdf-registration",
        "analysis_runtime": core.APP_VERSION,
        "analysis_engine_contract": "dissertation-level1-clean-v0.4",
        "pdf_overlay": "audiveris-omr-source-pixel-slot-aligned",
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

    with tempfile.TemporaryDirectory(prefix="tm-level1-v5-") as tmp:
        work = Path(tmp)
        source = work / filename
        source.write_bytes(data)
        try:
            omr_path = None
            if suffix == ".pdf":
                audiveris_out = work / "audiveris"
                symbolic = core.pdf_to_musicxml(source, audiveris_out)
                omr_path = find_omr(audiveris_out, symbolic)
            else:
                symbolic = source

            measures, attacks = core.parse_score(symbolic, meter)
            result = core.analyze_level1(measures, attacks)
            result["version"] = APP_VERSION
            result["analysis_version"] = core.APP_VERSION
            result["source"] = {
                "filename": filename,
                "input_type": suffix.lstrip("."),
                "opening_meter": f"{meter[0]}/{meter[1]}",
            }

            if suffix == ".pdf" and omr_path is not None:
                pages, diagnostics = render_original_pdf_with_level1_exact(
                    source, omr_path, measures, result
                )
                result["pdf_pages"] = pages
                result["pdf_overlay_diagnostics"] = diagnostics
            else:
                result["pdf_pages"] = []
                result["pdf_overlay_diagnostics"] = []
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(422, f"Level-1 analysis failed: {exc}") from exc
