"""Physical PDF display geometry for the Tone-Metric Analyzer.

This module is deliberately downstream of musical analysis.  Symbolic score time
and the recursive Levels engine are authoritative.  PDF/OMR geometry may only:

1. recover page images and score-system bounds; and
2. display already-established symbolic attack anchors.

It cannot create, delete, split, merge, reorder, or retime musical events.  There
is no x-cluster attack reconstruction, nearest-column timing fallback, or alternate
event pipeline in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from statistics import median
from zipfile import ZipFile

from lxml import etree
from PIL import Image

from .score_registration import build_layer_anchors_from_symbolic_layout
from .waves import build_wave_profile, register_wave_profile
from .pivots import build_pivot_profile, register_pivot_profile
from .trees import build_tree_profile, register_tree_profile


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
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


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
    if root is None:
        return None
    for el in root.iter():
        if _local(el.tag).lower() == name.lower():
            return el
    return None


def _is_notehead_shape(shape: str) -> bool:
    return "notehead" in (shape or "").lower().replace("_", "")


def _parse_annotation_xml(data: bytes, member: str, page_index: int) -> AnnotationPage | None:
    root = etree.fromstring(data)
    page_el = _find_desc(root, "Page")
    base = page_el if page_el is not None else root
    image_el = _find_desc(base, "Image")
    size_el = _find_desc(base, "Size")
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
    pages: list[AnnotationPage] = []
    with ZipFile(Path(annotation_zip), "r") as zf:
        members = sorted(
            [n for n in zf.namelist() if n.lower().endswith(".xml")],
            key=str.lower,
        )
        for member in members:
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
        base = PurePosixPath(ref).name.lower()
        matches = [n for n in names if PurePosixPath(n).name.lower() == base]
        if matches:
            return matches[0]
    raster_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    images = sorted([n for n in names if n.lower().endswith(raster_exts)], key=str.lower)
    if page.page_index < len(images):
        return images[page.page_index]
    return None


def extract_page_images(
    annotation_zip: str | Path,
    pages: list[AnnotationPage],
    out_dir: str | Path,
) -> tuple[list[dict], list[str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    outputs: list[dict] = []
    with ZipFile(Path(annotation_zip), "r") as zf:
        for page in pages:
            member = _find_image_member(zf, page)
            if member is None:
                warnings.append(
                    f"No score image was found in the annotation archive for page {page.page_index + 1}."
                )
                continue
            try:
                im = Image.open(BytesIO(zf.read(member)))
                im.load()
                if im.mode not in ("RGB", "RGBA", "L"):
                    im = im.convert("RGB")
                out = out_dir / f"page-{page.page_index + 1:03d}.png"
                im.save(out, format="PNG")
                iw, ih = im.size
            except Exception as exc:
                warnings.append(
                    f"Could not decode annotated score image for page {page.page_index + 1}: {exc}"
                )
                continue

            aw = page.annotation_width or float(iw)
            ah = page.annotation_height or float(ih)
            outputs.append({
                "page_index": page.page_index,
                "path": out,
                "width": iw,
                "height": ih,
                "scale_x": iw / aw if aw else 1.0,
                "scale_y": ih / ah if ah else 1.0,
                "annotation_page": page,
            })
    return outputs, warnings


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
    return {page: len(values) for page, values in by_page.items() if values}


def _staff_clusters(noteheads: list[SymbolBox]) -> list[list[SymbolBox]]:
    """Group noteheads vertically into note-bearing staff bands.

    These bands are display geometry only.  They never determine attack identity or
    musical time.
    """
    if not noteheads:
        return []
    interlines = [b.interline for b in noteheads if b.interline > 0]
    heights = [b.h for b in noteheads if b.h > 0]
    il = median(interlines) if interlines else max(8.0, median(heights) * 1.5 if heights else 8.0)
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


def _system_clusters(noteheads: list[SymbolBox], expected_count: int | None) -> list[list[SymbolBox]]:
    """Merge note-bearing staff bands into page systems for display bounds only."""
    staves = _staff_clusters(noteheads)
    if not staves:
        return []
    if expected_count and expected_count > 0:
        n = min(int(expected_count), len(staves))
        if n == 1:
            return [[b for staff in staves for b in staff]]
        gaps: list[tuple[float, int]] = []
        for i in range(len(staves) - 1):
            upper = max(b.y + b.h for b in staves[i])
            lower = min(b.y for b in staves[i + 1])
            gaps.append((lower - upper, i))
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

    # Conservative display fallback when the MusicXML page/system count is absent.
    interlines = [b.interline for b in noteheads if b.interline > 0]
    il = median(interlines) if interlines else 8.0
    threshold = max(30.0, 16.0 * il)
    ordered = sorted(noteheads, key=lambda b: b.cy)
    systems = [[ordered[0]]]
    last_y = ordered[0].cy
    for box in ordered[1:]:
        if box.cy - last_y > threshold:
            systems.append([box])
        else:
            systems[-1].append(box)
        last_y = box.cy
    return systems


def _system_bounds(
    page_outputs: list[dict],
    expected_system_counts: dict[int, int],
) -> dict[int, list[dict]]:
    by_page: dict[int, list[dict]] = {}
    for p in page_outputs:
        page: AnnotationPage = p["annotation_page"]
        sx, sy = float(p["scale_x"]), float(p["scale_y"])
        pw, ph = float(p["width"]), float(p["height"])
        systems = _system_clusters(page.noteheads, expected_system_counts.get(p["page_index"]))
        rows: list[dict] = []
        raw = []
        for system_index, system in enumerate(systems):
            raw.append({
                "system_index": system_index,
                "top": min((b.y for b in system), default=0.0) * sy,
                "bottom": max((b.y + b.h for b in system), default=0.0) * sy,
                "left": min((b.x for b in system), default=0.0) * sx,
                "right": max((b.x + b.w for b in system), default=0.0) * sx,
            })
        for pos, row in enumerate(raw):
            next_top = raw[pos + 1]["top"] if pos + 1 < len(raw) else row["bottom"]
            guard_bottom = (row["bottom"] + next_top) / 2.0 if pos + 1 < len(raw) else row["bottom"]
            rows.append({
                "system_index": int(row["system_index"]),
                "top_norm": row["top"] / ph if ph else 0.0,
                "bottom_norm": row["bottom"] / ph if ph else 0.0,
                "guard_bottom_norm": guard_bottom / ph if ph else 0.0,
                "left_norm": row["left"] / pw if pw else 0.0,
                "right_norm": row["right"] / pw if pw else 1.0,
            })
        by_page[p["page_index"]] = rows
    return by_page


def build_normalized_overlay(
    annotation_zip: str | Path,
    visual_groups: list[dict],
    layout_known: bool,
    analysis_result: dict,
    out_dir: str | Path,
) -> dict:
    """Attach display geometry to an already-complete symbolic analysis.

    Registration is one-way: symbolic attack key -> visual note layout -> PDF display.
    Waves, pivots, and trees are derived afterward and reuse only those existing
    anchors.  No PDF object can flow backward into score time or the recursive grid.
    """
    pages = read_annotation_pages(annotation_zip)
    page_outputs, warnings = extract_page_images(
        annotation_zip, pages, Path(out_dir) / "annotation_pages"
    )
    dims = {
        p["page_index"]: (float(p["width"]), float(p["height"]))
        for p in page_outputs
    }
    expected_counts = _expected_system_counts(visual_groups, layout_known)
    bounds_by_page = _system_bounds(page_outputs, expected_counts)

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
            "system_bounds": bounds_by_page.get(idx, []),
        }
        for idx in dims
    }

    anchors_by_page, structural_by_page, layer_stats, layer_warnings = (
        build_layer_anchors_from_symbolic_layout(analysis_result, dims)
    )

    wave_profile = build_wave_profile(analysis_result)
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

    layer_stats.update(wave_stats)
    layer_stats.update(pivot_stats)
    layer_stats.update(tree_stats)
    layer_warnings.extend(wave_warnings)
    layer_warnings.extend(pivot_warnings)
    layer_warnings.extend(tree_warnings)

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
    stats["validation_scope"] = (
        "symbolic-attacks-plus-independent-metric-grid-plus-read-only-wave-pivot-tree-display"
    )
    stats["active_registration_pipeline"] = (
        "symbolic-score-time->recursive-levels->symbolic-attack-key->note-layout->pdf-display"
    )
    stats["physical_geometry_timing_authority"] = False
    stats["physical_geometry_event_authority"] = False
    layer_anchor_count = sum(len(page.get("layer_anchors", [])) for page in per_page.values())
    return {
        "available": layer_anchor_count > 0,
        "pages": [per_page[k] for k in sorted(per_page)],
        "max_level": int(layer_stats.get("max_layer_level", 0) or 0),
        "matching": stats,
        "warnings": warnings,
    }
