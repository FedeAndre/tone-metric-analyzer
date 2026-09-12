"""Physical page geometry for generalized Levels and derived tone-metric waves.

The validated canonical attack and structural registration paths are the only ways
score-time positions receive PDF coordinates. Wave, pivot, and tree layers reuse
those already-registered positions; none has an independent timing or coordinate-
estimation path and none can override the recursive Levels. Legacy event/default-x,
barline, pivot, tree, and alternate registration implementations remain absent.
"""


from __future__ import annotations


from dataclasses import dataclass, field


from fractions import Fraction


from io import BytesIO


from pathlib import Path, PurePosixPath


from statistics import median


from typing import Iterable


from zipfile import ZipFile


from lxml import etree


from PIL import Image


from .score_registration import build_layer_anchors_from_canonical_score
from .waves import build_wave_profile, register_wave_profile
from .pivots import build_pivot_profile, register_pivot_profile
from .trees import build_tree_profile, register_tree_profile
from .omr_project import read_omr_slots


try:
    import numpy as np
except Exception:  # pragma: no cover - Pillow-only fallback remains available
    np = None


@dataclass
class SymbolBox:
    shape: str
    x: float
    y: float
    w: float
    h: float
    interline: float = 0.0
    symbol_id: str = ""

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class AnnotationPage:
    page_index: int
    annotation_member: str
    image_ref: str
    annotation_width: float
    annotation_height: float
    noteheads: list[SymbolBox] = field(default_factory=list)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attr_float(el, *names, default=0.0) -> float:
    for name in names:
        value = el.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except Exception:
                pass
    return default


def _find_desc(root, name: str):
    for el in root.iter():
        if _local(el.tag).lower() == name.lower():
            return el
    return None


def _is_notehead_shape(shape: str) -> bool:
    s = (shape or "").lower().replace("_", "")
    return "notehead" in s


def _parse_annotation_xml(data: bytes, member: str, page_index: int) -> AnnotationPage | None:
    root = etree.fromstring(data)
    page_el = _find_desc(root, "Page")
    image_el = _find_desc(page_el if page_el is not None else root, "Image")
    size_el = _find_desc(page_el if page_el is not None else root, "Size")
    image_ref = (image_el.text or "").strip() if image_el is not None else ""
    width = _attr_float(size_el, "w", "width") if size_el is not None else 0.0
    height = _attr_float(size_el, "h", "height") if size_el is not None else 0.0

    noteheads: list[SymbolBox] = []
    for el in root.iter():
        if _local(el.tag).lower() != "symbol":
            continue
        shape = el.get("shape") or ""
        if not _is_notehead_shape(shape):
            continue
        bounds = None
        for child in el:
            if _local(child.tag).lower() == "bounds":
                bounds = child
                break
        if bounds is None:
            bounds = _find_desc(el, "Bounds")
        if bounds is None:
            continue
        x = _attr_float(bounds, "x")
        y = _attr_float(bounds, "y")
        w = _attr_float(bounds, "w", "width")
        h = _attr_float(bounds, "h", "height")
        if w <= 0 or h <= 0:
            continue
        noteheads.append(SymbolBox(
            shape=shape,
            x=x,
            y=y,
            w=w,
            h=h,
            interline=_attr_float(el, "interline"),
            symbol_id=el.get("id") or "",
        ))
    if not image_ref and not noteheads:
        return None
    return AnnotationPage(
        page_index=page_index,
        annotation_member=member,
        image_ref=image_ref,
        annotation_width=width,
        annotation_height=height,
        noteheads=noteheads,
    )


def read_annotation_pages(annotation_zip: str | Path) -> list[AnnotationPage]:
    annotation_zip = Path(annotation_zip)
    pages: list[AnnotationPage] = []
    with ZipFile(annotation_zip, "r") as zf:
        xml_members = sorted(
            [n for n in zf.namelist() if n.lower().endswith(".xml")],
            key=lambda n: n.lower(),
        )
        for member in xml_members:
            try:
                page = _parse_annotation_xml(zf.read(member), member, len(pages))
            except Exception:
                continue
            if page is not None:
                pages.append(page)
    return pages


def _find_image_member(zf: ZipFile, page: AnnotationPage) -> str | None:
    names = zf.namelist()
    if page.image_ref:
        ref = page.image_ref.replace("\\", "/")
        if ref in names:
            return ref
        ref_base = PurePosixPath(ref).name.lower()
        candidates = [n for n in names if PurePosixPath(n).name.lower() == ref_base]
        if candidates:
            return candidates[0]
    # Fallback: pair page N with the Nth raster image in the archive.
    raster_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    imgs = sorted([n for n in names if n.lower().endswith(raster_exts)], key=lambda n: n.lower())
    if page.page_index < len(imgs):
        return imgs[page.page_index]
    return None


def extract_page_images(annotation_zip: str | Path, pages: list[AnnotationPage], out_dir: str | Path) -> tuple[list[dict], list[str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    outputs: list[dict] = []
    with ZipFile(annotation_zip, "r") as zf:
        for page in pages:
            member = _find_image_member(zf, page)
            if member is None:
                warnings.append(f"No score image was found in the Audiveris annotation archive for page {page.page_index + 1}.")
                continue
            try:
                im = Image.open(BytesIO(zf.read(member)))
                im.load()
                # Browser-friendly PNG while preserving the exact Audiveris sheet image geometry.
                if im.mode not in ("RGB", "RGBA", "L"):
                    im = im.convert("RGB")
                out = out_dir / f"page-{page.page_index + 1:03d}.png"
                im.save(out, format="PNG")
                iw, ih = im.size
            except Exception as exc:
                warnings.append(f"Could not decode annotated score image for page {page.page_index + 1}: {exc}")
                continue

            aw = page.annotation_width or float(iw)
            ah = page.annotation_height or float(ih)
            sx = iw / aw if aw else 1.0
            sy = ih / ah if ah else 1.0
            outputs.append({
                "page_index": page.page_index,
                "path": out,
                "width": iw,
                "height": ih,
                "scale_x": sx,
                "scale_y": sy,
                "annotation_page": page,
            })
    return outputs, warnings


def _staff_line_bands(image_path: str | Path) -> list[dict]:
    """Detect five-line staff bands directly from the rendered score image.

    Audiveris annotation exports give us excellent notehead coordinates, but an empty
    staff can contain *no noteheads at all*.  Notehead-only geometry therefore cannot
    know that such a staff belongs to the score system above it.  This image-based
    helper recovers long, approximately equally spaced sets of five horizontal staff
    lines.  It is intentionally used only to expand vertical system bounds; noteheads
    remain the authoritative x/time anchors for the analysis.
    """
    if np is None:
        return []
    try:
        with Image.open(image_path) as src:
            gray = src.convert("L")
            width, height = gray.size
            # Keep the row projection cheap on very large scans without changing the
            # geometry returned to the caller.
            scale = min(1.0, 1800.0 / max(width, 1))
            if scale < 1.0:
                work = gray.resize((max(1, round(width * scale)), max(1, round(height * scale))))
            else:
                work = gray
            arr = np.asarray(work, dtype=np.uint8)
    except Exception:
        return []

    h, w = arr.shape[:2]
    if h < 30 or w < 80:
        return []
    x0, x1 = int(w * 0.04), int(w * 0.96)
    if x1 <= x0:
        x0, x1 = 0, w
    roi = arr[:, x0:x1]
    # Staff lines are among the darkest and longest horizontal components.  A fairly
    # permissive threshold handles both clean PDFs and older grayscale scans; the
    # five-line equal-spacing test below rejects ordinary text rules and titles.
    dark = roi < 190
    row_fraction = dark.mean(axis=1)
    high = float(np.percentile(row_fraction, 99.5)) if len(row_fraction) else 0.0
    row_threshold = max(0.14, min(0.42, high * 0.48))
    candidate_rows = np.flatnonzero(row_fraction >= row_threshold).tolist()
    if not candidate_rows:
        return []

    # Collapse line thickness / antialiasing into one center per horizontal rule.
    line_groups: list[list[int]] = [[candidate_rows[0]]]
    for y in candidate_rows[1:]:
        if y - line_groups[-1][-1] <= 2:
            line_groups[-1].append(y)
        else:
            line_groups.append([y])
    centers = [sum(g) / len(g) for g in line_groups]
    strengths = [max(float(row_fraction[y]) for y in g) for g in line_groups]

    candidates: list[dict] = []
    for i in range(len(centers) - 4):
        ys = centers[i:i + 5]
        gaps = [ys[j + 1] - ys[j] for j in range(4)]
        spacing = float(median(gaps))
        if spacing < 3.0 or spacing > 45.0:
            continue
        tol = max(1.6, spacing * 0.28)
        if max(abs(g - spacing) for g in gaps) > tol:
            continue
        score = sum(strengths[i:i + 5]) / 5.0
        # Long lines should normally be quite strong.  Keep this threshold lower than
        # the row threshold because noteheads/stems can locally interrupt staff lines.
        if score < max(0.12, row_threshold * 0.82):
            continue
        candidates.append({
            "first": i,
            "last": i + 4,
            "top": ys[0],
            "bottom": ys[-1],
            "center": (ys[0] + ys[-1]) / 2.0,
            "spacing": spacing,
            "score": score,
        })

    # Prefer the strongest candidate when overlapping five-line windows are possible,
    # then restore top-to-bottom order.
    chosen: list[dict] = []
    for cand in sorted(candidates, key=lambda c: (-c["score"], c["top"])):
        if any(not (cand["bottom"] < x["top"] - cand["spacing"] * 0.5 or cand["top"] > x["bottom"] + cand["spacing"] * 0.5) for x in chosen):
            continue
        chosen.append(cand)
    chosen.sort(key=lambda c: c["top"])

    inv = 1.0 / scale if scale > 0 else 1.0
    out: list[dict] = []
    for c in chosen:
        spacing = c["spacing"] * inv
        # Reserve a little space outside the five lines so an empty staff's ledger/
        # voice material cannot be mistaken for inter-system whitespace.
        out.append({
            "top": max(0.0, c["top"] * inv - 1.8 * spacing),
            "bottom": min(float(height), c["bottom"] * inv + 1.8 * spacing),
            "center": c["center"] * inv,
            "spacing": spacing,
        })
    return out


def _visual_system_bands(image_path: str | Path, expected_count: int | None) -> list[dict]:
    """Group detected staff bands into complete visual score systems.

    This includes empty staves because the grouping is based on staff lines rather than
    noteheads.  When a symbolic system count is known, the largest inter-staff gaps are
    used as the boundaries between systems, mirroring the established notehead-cluster
    strategy but with the complete page geometry available.
    """
    staves = _staff_line_bands(image_path)
    if not staves or not expected_count or expected_count <= 0:
        return []
    n = min(int(expected_count), len(staves))
    if n <= 0:
        return []
    if n == 1:
        return [{"top": staves[0]["top"], "bottom": staves[-1]["bottom"]}]
    if len(staves) < n:
        return []

    gaps: list[tuple[float, int]] = []
    for i in range(len(staves) - 1):
        # Use staff centers rather than already-expanded bounds; the latter can overlap
        # within a multi-staff system by design.
        gap = staves[i + 1]["center"] - staves[i]["center"]
        gaps.append((gap, i))
    boundaries = {i for _, i in sorted(gaps, reverse=True)[: n - 1]}
    systems: list[dict] = []
    start = 0
    for i in range(len(staves) - 1):
        if i in boundaries:
            chunk = staves[start:i + 1]
            systems.append({"top": chunk[0]["top"], "bottom": chunk[-1]["bottom"]})
            start = i + 1
    chunk = staves[start:]
    if chunk:
        systems.append({"top": chunk[0]["top"], "bottom": chunk[-1]["bottom"]})
    return systems if len(systems) == n else []


def _staff_clusters(noteheads: list[SymbolBox]) -> list[list[SymbolBox]]:
    """Group noteheads into approximate physical staves.

    This is deliberately a tighter grouping than score-system grouping.  It gives us
    stable vertical bands first; several adjacent staff bands can then be merged into
    one multi-staff organ/piano system.
    """
    if not noteheads:
        return []
    interlines = [b.interline for b in noteheads if b.interline > 0]
    il = median(interlines) if interlines else max(8.0, median([b.h for b in noteheads]) * 1.5)
    threshold = max(18.0, 4.5 * il)
    ordered = sorted(noteheads, key=lambda b: b.cy)
    staves: list[list[SymbolBox]] = [[ordered[0]]]
    last_y = ordered[0].cy
    for box in ordered[1:]:
        if box.cy - last_y > threshold:
            staves.append([box])
        else:
            staves[-1].append(box)
        last_y = box.cy
    return staves


def _system_clusters(noteheads: list[SymbolBox], expected_count: int | None = None) -> list[list[SymbolBox]]:
    """Group noteheads into complete score systems, not individual staves.

    When MusicXML tells us how many systems occur on a page, use that count to merge
    vertically adjacent staff clusters.  The ``expected_count - 1`` largest gaps
    between staff bands become system boundaries.  This prevents an organ system
    (manual staves plus pedal) from being rendered as several independent analyses.

    If no reliable symbolic system count is available, fall back to the older
    annotation-only vertical-gap heuristic.
    """
    if not noteheads:
        return []
    staves = _staff_clusters(noteheads)
    if expected_count and expected_count > 0 and staves:
        n = min(int(expected_count), len(staves))
        if n == 1:
            return [[b for staff in staves for b in staff]]
        # Gap between the visible extents of consecutive note-bearing staff bands.
        gaps: list[tuple[float, int]] = []
        for i in range(len(staves) - 1):
            upper_bottom = max(b.y + b.h for b in staves[i])
            lower_top = min(b.y for b in staves[i + 1])
            gaps.append((lower_top - upper_bottom, i))
        boundaries = {i for _, i in sorted(gaps, reverse=True)[: n - 1]}
        systems: list[list[SymbolBox]] = []
        current: list[SymbolBox] = []
        for i, staff in enumerate(staves):
            current.extend(staff)
            if i in boundaries:
                systems.append(current)
                current = []
        if current:
            systems.append(current)
        if len(systems) == n:
            return systems

    interlines = [b.interline for b in noteheads if b.interline > 0]
    il = median(interlines) if interlines else max(8.0, median([b.h for b in noteheads]) * 1.5)
    threshold = max(30.0, 16.0 * il)
    ordered = sorted(noteheads, key=lambda b: b.cy)
    systems: list[list[SymbolBox]] = [[ordered[0]]]
    last_y = ordered[0].cy
    for box in ordered[1:]:
        if box.cy - last_y > threshold:
            systems.append([box])
        else:
            systems[-1].append(box)
        last_y = box.cy
    return systems


def _attack_columns(system: list[SymbolBox]) -> list[dict]:
    """Collapse only *genuinely simultaneous* noteheads into physical attack columns.

    Earlier builds used a tolerance close to one full staff interline.  That was too
    permissive for tightly engraved beamed passages: two successive eighth-note
    noteheads can be less than one interline apart horizontally, so adjacent attacks
    were occasionally merged into one column.  Once a real attack disappears from
    the physical column sequence, every later onset in that measure can shift by one.

    The canonical visual grouping deliberately prefers *under-merging* to over-merging.  Simultaneous chord
    noteheads normally share nearly the same x coordinate, while sequential attacks
    occupy clearly separate columns.  A small tolerance (about one third of an
    interline, capped by notehead width) preserves chords but never swallows the next
    dense rhythmic attack.  Extra split chord columns are harmless downstream because
    the monotonic onset matcher can skip extras; a merged sequential attack is not.
    """
    if not system:
        return []
    interlines = [b.interline for b in system if b.interline > 0]
    widths = [b.w for b in system if b.w > 0]
    il = median(interlines) if interlines else max(8.0, median(widths) * 1.3 if widths else 8.0)
    med_w = median(widths) if widths else il * 0.65

    # Keep the merge radius intentionally conservative.  The width-based cap matters
    # on large scans where the exported interline is itself large.
    x_tolerance = max(2.5, min(0.34 * il, 0.55 * med_w))

    ordered = sorted(system, key=lambda b: b.cx)
    columns: list[list[SymbolBox]] = [[ordered[0]]]
    running_x = ordered[0].cx
    for box in ordered[1:]:
        if abs(box.cx - running_x) <= x_tolerance:
            columns[-1].append(box)
            running_x = sum(b.cx for b in columns[-1]) / len(columns[-1])
        else:
            columns.append([box])
            running_x = box.cx
    out = []
    for boxes in columns:
        out.append({
            "boxes": boxes,
            "cx": sum(b.cx for b in boxes) / len(boxes),
            "cy": sum(b.cy for b in boxes) / len(boxes),
        })
    return out


def _expected_system_counts(symbolic_groups: list[dict], layout_known: bool) -> dict[int, int]:
    if not layout_known:
        return {}
    by_page: dict[int, set[int]] = {}
    for row in symbolic_groups:
        try:
            page = int(row.get("layout_page", 0))
            system = int(row.get("layout_system", 0))
        except Exception:
            continue
        by_page.setdefault(page, set()).add(system)
    return {page: len(systems) for page, systems in by_page.items() if systems}


def build_physical_groups(page_outputs: list[dict], expected_system_counts: dict[int, int] | None = None) -> list[dict]:
    groups: list[dict] = []
    expected_system_counts = expected_system_counts or {}
    for p in page_outputs:
        page: AnnotationPage = p["annotation_page"]
        sx, sy = p["scale_x"], p["scale_y"]
        expected = expected_system_counts.get(p["page_index"])
        systems = _system_clusters(page.noteheads, expected)
        # Notehead-only bounds miss empty staves. Recover complete visual staff systems
        # from the rendered page and use them only when they agree with the symbolic
        # system count/order. This keeps OMR noteheads authoritative for x/time mapping.
        visual_bands = _visual_system_bands(p["path"], expected)
        use_visual_bands = len(visual_bands) == len(systems) and len(systems) > 0
        for system_index, system in enumerate(systems):
            system_top = min((b.y for b in system), default=0.0) * sy
            system_bottom = max((b.y + b.h for b in system), default=0.0) * sy
            if use_visual_bands:
                system_top = min(system_top, float(visual_bands[system_index]["top"]))
                system_bottom = max(system_bottom, float(visual_bands[system_index]["bottom"]))
            system_left = min((b.x for b in system), default=0.0) * sx
            system_right = max((b.x + b.w for b in system), default=0.0) * sx
            for group_index, col in enumerate(_attack_columns(system)):
                boxes = [{
                    "x": b.x * sx,
                    "y": b.y * sy,
                    "w": b.w * sx,
                    "h": b.h * sy,
                    "shape": b.shape,
                    "id": b.symbol_id,
                } for b in col["boxes"]]
                groups.append({
                    "page_index": p["page_index"],
                    "system_index": system_index,
                    "group_index": group_index,
                    "boxes": boxes,
                    "cx": col["cx"] * sx,
                    "cy": col["cy"] * sy,
                    "system_top": system_top,
                    "system_bottom": system_bottom,
                    "system_left": system_left,
                    "system_right": system_right,
                })
    return groups


def build_normalized_overlay(annotation_zip: str | Path, visual_groups: list[dict], layout_known: bool, analysis_result: dict, out_dir: str | Path, omr_path: str | Path | None = None) -> dict:
    """Build the PDF overlay for recursive Levels, structural points, waves, pivots, and trees.

    The current build retains one canonical registration architecture. Normal attack
    labels retain canonical score time and may shift only their display x to the true
    notehead center. Parenthetical structural articulations use the separate display-
    only placement path. The wave
    envelope is derived afterward and may reuse only those existing anchors.
    Pivots are derived from that fixed wave and may reuse only wave anchors. Trees
    are the final read-only derivative: actual sonic events are placed at their
    lowest occupied Level and branches reuse only existing actual wave anchors.

    Earlier event-to-note, engraving-coordinate, barline, nearest-column, legacy
    pivot, and legacy tree registration pipelines are absent and cannot mutate or
    override the result.
    """
    pages = read_annotation_pages(annotation_zip)
    page_outputs, warnings = extract_page_images(
        annotation_zip, pages, Path(out_dir) / "annotation_pages"
    )
    expected_counts = _expected_system_counts(visual_groups, layout_known)
    if not expected_counts and omr_path is not None:
        try:
            _slots, omr_layout_meta = read_omr_slots(omr_path)
            expected_counts = {
                int(row["page_index"]): int(row["systems"])
                for row in omr_layout_meta.get("pages", [])
                if row.get("systems") is not None
            }
        except Exception as exc:
            warnings.append(f"Could not recover OMR system counts for overlay layout: {exc}")
    physical_groups = build_physical_groups(page_outputs, expected_counts)

    dims = {
        p["page_index"]: (float(p["width"]), float(p["height"]))
        for p in page_outputs
    }
    per_page = {
        idx: {
            "page_index": idx,
            "overlays": [],
            "layer_anchors": [],
            "structural_anchors": [],
            "wave_anchors": [],
            "pivot_anchors": [],
            "tree_nodes": [],
            "tree_branches": [],
            "system_bounds": [],
        }
        for idx in dims
    }

    # System bounds are used only to reserve visual room above each complete score
    # system.  They do not determine any layer x-coordinate or recursive level.
    systems_by_page: dict[int, dict[int, dict]] = {}
    for g in physical_groups:
        page_idx = int(g.get("page_index", 0))
        sys_idx = int(g.get("system_index", 0))
        bucket = systems_by_page.setdefault(page_idx, {}).setdefault(sys_idx, {
            "top": float(g.get("system_top", g.get("cy", 0))),
            "bottom": float(g.get("system_bottom", g.get("cy", 0))),
            "left": float(g.get("system_left", g.get("cx", 0))),
            "right": float(g.get("system_right", g.get("cx", 0))),
        })
        bucket["top"] = min(bucket["top"], float(g.get("system_top", bucket["top"])))
        bucket["bottom"] = max(bucket["bottom"], float(g.get("system_bottom", bucket["bottom"])))
        bucket["left"] = min(bucket["left"], float(g.get("system_left", bucket["left"])))
        bucket["right"] = max(bucket["right"], float(g.get("system_right", bucket["right"])))

    for page_idx, systems in systems_by_page.items():
        pw, ph = dims.get(page_idx, (1.0, 1.0))
        ordered = sorted(systems.items(), key=lambda kv: kv[1]["top"])
        for pos, (sys_idx, b) in enumerate(ordered):
            if pos + 1 < len(ordered):
                next_top = ordered[pos + 1][1]["top"]
                guard_bottom = (b["bottom"] + next_top) / 2.0
            else:
                guard_bottom = b["bottom"]
            per_page[page_idx]["system_bounds"].append({
                "system_index": int(sys_idx),
                "top_norm": b["top"] / ph,
                "bottom_norm": b["bottom"] / ph,
                "guard_bottom_norm": guard_bottom / ph,
                "left_norm": b["left"] / pw,
                "right_norm": b["right"] / pw,
            })

    if omr_path is not None:
        anchors_by_page, structural_by_page, layer_stats, layer_warnings = (
            build_layer_anchors_from_canonical_score(analysis_result, dims)
        )
        # The wave is a read-only envelope of the already-computed recursive grid.
        # Its score-time points are built before PDF placement and are then matched
        # only to the existing attack/structural anchors above.
        wave_profile = build_wave_profile(analysis_result)
        wave_by_page, wave_stats, wave_warnings = register_wave_profile(
            wave_profile, anchors_by_page, structural_by_page
        )
        # Pivots are a read-only derivative of the completed wave profile.
        # Dissertation-style pivot regions reuse only the two adjacent existing
        # wave anchors spanning the first descent; they cannot alter score time,
        # attacks, Levels, or wave coordinates.
        pivot_profile = build_pivot_profile(wave_profile)
        pivot_by_page, pivot_stats, pivot_warnings = register_pivot_profile(
            pivot_profile, wave_by_page
        )
        # Trees remain a read-only derivative. They use only actual sonic-event
        # points already present in the fixed wave profile.
        # Each event is placed at its lowest occupied Level and branch topology is
        # determined in score time before reusing the exact existing wave anchors.
        tree_profile = build_tree_profile(wave_profile)
        tree_nodes_by_page, tree_branches_by_page, tree_stats, tree_warnings = register_tree_profile(
            tree_profile, wave_by_page
        )
        layer_stats.update(wave_stats)
        layer_stats.update(pivot_stats)
        layer_stats.update(tree_stats)
        layer_warnings.extend(wave_warnings)
        layer_warnings.extend(pivot_warnings)
        layer_warnings.extend(tree_warnings)
        canonical_meta = analysis_result.get("canonical_score_meta") or {}
        try:
            import json as _json
            debug_path = Path(out_dir).parent / "omr-canonical-score.json"
            debug_path.write_text(
                _json.dumps(canonical_meta, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            layer_warnings.append(f"Could not write canonical-score debug audit: {exc}")
        layer_stats["canonical_score_meta"] = {
            "available": bool(canonical_meta.get("available")),
            "column_count": int(canonical_meta.get("column_count", 0) or 0),
            "attack_column_count": int(canonical_meta.get("attack_column_count", 0) or 0),
            "measure_count": len(canonical_meta.get("measures", [])),
            "warnings": list(canonical_meta.get("warnings", [])),
        }
    else:
        anchors_by_page = {}
        structural_by_page = {}
        wave_by_page = {}
        pivot_by_page = {}
        tree_nodes_by_page = {}
        tree_branches_by_page = {}
        expected_by_level: dict[int, int] = {}
        for seg in analysis_result.get("segments", []):
            for event in seg.get("events", []):
                for raw_level in event.get("tone_metric_levels", []) or []:
                    try:
                        level = int(raw_level)
                    except Exception:
                        continue
                    if level > 0:
                        expected_by_level[level] = expected_by_level.get(level, 0) + 1
        levels_enabled = sorted(expected_by_level)
        layer_stats = {
            "final_layer_anchors": 0,
            "layer_anchor_registration": "canonical-score-required",
            "levels_enabled": levels_enabled,
            "max_layer_level": max(levels_enabled, default=0),
            "layer_hits_expected": sum(expected_by_level.values()),
            "layer_hits_mapped": 0,
            "layer_hits_without_visual_attack": sum(expected_by_level.values()),
            "per_level": {
                str(level): {"expected": count, "mapped": 0, "missing": count}
                for level, count in sorted(expected_by_level.items())
            },
            "structural_parenthetical_positions": 0,
            "structural_parenthetical_labels_expected": 0,
            "structural_parenthetical_labels_mapped": 0,
            "structural_parenthetical_labels_missing": 0,
            "structural_per_level": {},
            "structural_registration_sources": {},
            "structural_reasons": {},
            "wave_profile_points_expected": 0,
            "wave_profile_points_mapped": 0,
            "wave_profile_points_missing": 0,
            "wave_attack_points_mapped": 0,
            "wave_parenthetical_points_mapped": 0,
            "wave_max_height": max(levels_enabled, default=0),
            "wave_registration": "canonical-score-required",
            "wave_coordinate_synthesis": False,
            "wave_missing_points": [],
            "wave_duplicate_existing_anchor_keys": [],
            "pivot_profile_expected": 0,
            "pivot_profile_mapped": 0,
            "pivot_profile_missing": 0,
            "simple_pivots": 0,
            "compound_pivots": 0,
            "cross_system_pivots": 0,
            "pivot_visual_pieces": 0,
            "pivot_registration": "canonical-score-required",
            "pivot_coordinate_synthesis": False,
            "pivot_missing": [],
            "pivot_duplicate_wave_anchor_keys": [],
            "tree_event_nodes_expected": 0,
            "tree_event_nodes_mapped": 0,
            "tree_event_nodes_missing": 0,
            "tree_branches_expected": 0,
            "tree_branches_mapped": 0,
            "tree_branches_missing": 0,
            "tree_roots": 0,
            "tree_cross_system_branches": 0,
            "tree_visual_pieces": 0,
            "tree_registration": "canonical-score-required",
            "tree_coordinate_synthesis": False,
            "tree_missing_nodes": [],
            "tree_missing_branches": [],
            "tree_duplicate_actual_wave_anchor_keys": [],
        }
        for level, count in expected_by_level.items():
            layer_stats[f"level{level}_anchors_mapped"] = 0
            layer_stats[f"level{level}_hits_expected"] = count
            layer_stats[f"level{level}_hits_without_visual_attack"] = count
        layer_warnings = [
            "Saved Audiveris .omr project unavailable; event Levels were not placed on the PDF."
        ]

    for page_idx, anchors in anchors_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]["layer_anchors"] = anchors
    for page_idx, anchors in structural_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]["structural_anchors"] = anchors
    for page_idx, anchors in wave_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]["wave_anchors"] = anchors
    for page_idx, anchors in pivot_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]["pivot_anchors"] = anchors
    for page_idx, anchors in tree_nodes_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]["tree_nodes"] = anchors
    for page_idx, branches in tree_branches_by_page.items():
        if page_idx in per_page:
            per_page[page_idx]["tree_branches"] = branches

    warnings.extend(layer_warnings)
    stats = dict(layer_stats)
    stats["validation_scope"] = "generalized-levels-plus-parenthetical-structural-articulations-plus-derived-wave-envelope-plus-derived-pivots-plus-derived-trees"
    stats["active_registration_pipeline"] = "canonical-score-time->recursive-levels->existing-level-anchors->derived-wave-envelope->derived-pivots->derived-trees->pdf"
    layer_anchor_count = sum(len(page.get("layer_anchors", [])) for page in per_page.values())
    return {
        "available": layer_anchor_count > 0,
        "pages": [per_page[k] for k in sorted(per_page)],
        "max_level": int(layer_stats.get("max_layer_level", 0) or 0),
        "matching": stats,
        "warnings": warnings,
    }
