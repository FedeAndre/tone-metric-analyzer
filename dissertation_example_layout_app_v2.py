from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import fitz

import dissertation_example_layout_app as layout

# Latest presentation layer.  The validated analytical engine is still untouched.
# This module corrects only how already-computed analysis is registered and drawn
# above printed score systems.
current = layout.current
base = layout.base
old = layout.old
app = layout.app
APP_VERSION = "1.3.1-dissertation-example-layout-audited"
app.version = APP_VERSION


def _canonical_overlay_audited(levels_payload: dict, canonical_meta: dict) -> dict:
    """Register systems before deriving wave/pivot/tree visual spans.

    The previous presentation wrapper recovered printed systems only *after*
    pivots and tree branches had already been classified as same-system spans.
    On OMR pages whose system id was collapsed, that could place pivot/tree
    material in the wrong printed line.  Here the analytical Levels are unchanged;
    only visual system membership is recovered first, then every derivative is
    registered against those corrected anchors.
    """
    dims = current._page_dimensions(canonical_meta)
    visual_levels = deepcopy(levels_payload)
    visual_levels["canonical_score_meta"] = canonical_meta

    anchors_by_page, structural_by_page, stats, warnings = current.build_layer_anchors_from_canonical_score(
        visual_levels, dims
    )

    # Recover/retain printed-system membership before downstream registration.
    initial_bounds = current._system_bounds(canonical_meta, dims)
    bounds_by_page: dict[int, list[dict]] = {}
    for page in sorted(dims):
        probe = {
            "page_index": page,
            "layer_anchors": list(anchors_by_page.get(page, [])),
            "structural_anchors": list(structural_by_page.get(page, [])),
            "wave_anchors": [],
            "pivot_anchors": [],
            "tree_nodes": [],
            "tree_branches": [],
            "system_bounds": list(initial_bounds.get(page, [])),
        }
        layout._recover_visual_systems(probe)
        # Rows are the same mutable dictionaries returned in anchors_by_page /
        # structural_by_page, so recovered system ids are now authoritative for
        # all subsequent visual registration.
        bounds_by_page[page] = list(probe.get("system_bounds", []) or initial_bounds.get(page, []))

    wave_profile = current.build_wave_profile(visual_levels)
    wave_by_page, wave_stats, wave_warnings = current.register_wave_profile(
        wave_profile, anchors_by_page, structural_by_page
    )
    pivot_profile = current.build_pivot_profile(wave_profile)
    pivot_by_page, pivot_stats, pivot_warnings = current.register_pivot_profile(
        pivot_profile, wave_by_page
    )
    tree_profile = current.build_tree_profile(wave_profile)
    tree_nodes_by_page, tree_branches_by_page, tree_stats, tree_warnings = current.register_tree_profile(
        tree_profile, wave_by_page
    )

    stats = dict(stats)
    stats.update(wave_stats)
    stats.update(pivot_stats)
    stats.update(tree_stats)
    warnings = list(warnings) + list(wave_warnings) + list(pivot_warnings) + list(tree_warnings)

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
            "system_bounds": list(bounds_by_page.get(page, [])),
        })

    layer_anchor_count = sum(len(row["layer_anchors"]) for row in pages)
    stats["active_registration_pipeline"] = (
        "saved-omr-canonical-score->recursive-levels->recover-printed-systems->"
        "wave->pivots->trees->original-pdf"
    )
    stats["visual_geometry_source"] = "saved-audiveris-omr"
    stats["musicxml_layout_required"] = False
    stats["presentation_system_alignment"] = "recovered-before-wave-pivot-tree-registration"
    stats["presentation_style"] = "dissertation-example-full-grid-grey-bands-step-wave-pivot-bars"
    return {
        "available": layer_anchor_count > 0,
        "pages": pages,
        "max_level": int(stats.get("max_layer_level", 0) or 0),
        "matching": stats,
        "warnings": list(dict.fromkeys(str(x) for x in warnings if str(x).strip())),
    }


# Replace the previously active presentation registration path rather than stacking
# another downstream recovery pass on top of it.
current._canonical_overlay = _canonical_overlay_audited


def _pdf_draw_band_audited(page: fitz.Page, op: dict, m: dict, band_top: float, width: float) -> None:
    """Draw a dissertation-style complete recursive grid for one printed system."""
    left_label_x = max(6.0, m["left"] - 70.0)
    page.insert_text((left_label_x, band_top + 13.0), "Measures", fontsize=10.5, color=(0, 0, 0), overlay=True)
    for number, x in layout._pdf_measure_centers(op, m["system_index"], width):
        page.insert_text((x - 3.0, band_top + 13.0), number, fontsize=10.5, color=(0, 0, 0), overlay=True)
    page.insert_text((left_label_x + 16.0, band_top + 28.0), "Levels", fontsize=8.5, color=(0, 0, 0), overlay=True)
    for level in range(m["max_level"], 0, -1):
        y = layout._pdf_level_y(m, level, band_top)
        page.insert_text((left_label_x + 28.0, y + 3.0), str(level), fontsize=8.5, color=(0, 0, 0), overlay=True)

    attacks = [
        {**r, "_structural": False}
        for r in op.get("layer_anchors", []) or []
        if int(r.get("system_index", 0)) == m["system_index"]
    ]
    structural = [
        {**r, "_structural": True}
        for r in op.get("structural_anchors", []) or []
        if int(r.get("system_index", 0)) == m["system_index"]
    ]
    points = sorted(attacks + structural, key=lambda r: float(r.get("cx_norm", 0.0)))
    xs = [float(r.get("cx_norm", 0.0)) * width for r in points]
    for i, row in enumerate(points):
        x = xs[i]
        l = m["left"] if i == 0 else (xs[i - 1] + x) / 2.0
        r = m["right"] if i + 1 == len(xs) else (x + xs[i + 1]) / 2.0
        if r <= l:
            continue
        for raw in row.get("levels", []) or []:
            try:
                level = int(raw)
            except Exception:
                continue
            y = layout._pdf_level_y(m, level, band_top)
            shade = max(0.48, min(0.93, 0.94 - (level - 1) * 0.07))
            page.draw_rect(
                fitz.Rect(l, y - m["row_h"] * 0.45, r, y + m["row_h"] * 0.45),
                color=None,
                fill=(shade, shade, shade),
                overlay=True,
            )
            value = f"({level})" if row.get("_structural") else str(level)
            fontsize = 7.8 if row.get("_structural") else 8.5
            dx = 4.0 if row.get("_structural") else 2.5
            page.insert_text((x - dx, y + 3.0), value, fontsize=fontsize, color=(0, 0, 0), overlay=True)

    wave_rows = [
        r for r in op.get("wave_anchors", []) or []
        if int(r.get("system_index", 0)) == m["system_index"]
    ]
    wave_rows.sort(key=lambda r: float(r.get("cx_norm", 0.0)))
    if wave_rows:
        wx = [float(r.get("cx_norm", 0.0)) * width for r in wave_rows]
        shape = page.new_shape()
        last_y = None
        for i, row in enumerate(wave_rows):
            x = wx[i]
            seg_l = m["left"] if i == 0 else (wx[i - 1] + x) / 2.0
            seg_r = m["right"] if i + 1 == len(wx) else (x + wx[i + 1]) / 2.0
            try:
                level = int(row.get("height", 1) or 1)
            except Exception:
                level = 1
            y = layout._pdf_level_y(m, level, band_top) - m["row_h"] * 0.45
            if last_y is not None:
                shape.draw_line(fitz.Point(seg_l, last_y), fitz.Point(seg_l, y))
            shape.draw_line(fitz.Point(seg_l, y), fitz.Point(seg_r, y))
            last_y = y
        shape.finish(color=(0.16, 0.16, 0.16), width=0.8)
        shape.commit(overlay=True)

    page.insert_text((left_label_x + 16.0, band_top + m["pivot_y"] + 3.0), "Pivots", fontsize=8.5, color=(0, 0, 0), overlay=True)
    for row in op.get("pivot_anchors", []) or []:
        if int(row.get("system_index", 0)) != m["system_index"]:
            continue
        role = row.get("span_role") or "complete"
        if role == "complete":
            try:
                x1 = float(row.get("start_cx_norm")) * width
                x2 = float(row.get("end_cx_norm")) * width
            except Exception:
                continue
        else:
            try:
                x = float(row.get("cx_norm")) * width
            except Exception:
                continue
            if role == "start-endpoint":
                x1, x2 = x, min(m["right"], x + 24.0)
            else:
                x1, x2 = max(m["left"], x - 24.0), x
        shade = 0.58 if str(row.get("kind", "")).lower() == "compound" else 0.78
        y = band_top + m["pivot_y"] - 5.0
        page.draw_rect(fitz.Rect(min(x1, x2), y, max(x1, x2), y + 10.0), color=None, fill=(shade, shade, shade), overlay=True)

    for row in op.get("tree_branches", []) or []:
        if int(row.get("system_index", 0)) != m["system_index"] or row.get("span_role") not in (None, "", "complete"):
            continue
        try:
            x1 = float(row.get("source_cx_norm")) * width
            x2 = float(row.get("target_cx_norm")) * width
            y1 = layout._pdf_level_y(m, int(row.get("source_level", 1)), band_top)
            y2 = layout._pdf_level_y(m, int(row.get("target_level", 1)), band_top)
        except Exception:
            continue
        page.draw_line(fitz.Point(x1, y1), fitz.Point(x2, y2), color=(0.35, 0.35, 0.35), width=0.45, overlay=True)


# The reflow writer in the current layout resolves this global function at call
# time, so replacing it updates the downloadable annotated PDF without changing
# any analysis or source-score geometry.
layout._pdf_draw_band = _pdf_draw_band_audited


# Replace only the browser's grid drawing function.  Full-grid modes now give
# structural (parenthetical) positions the same grey-band cell treatment visible
# in the dissertation example instead of leaving visual holes between attacks.
_html = base.HTML
_start = _html.find(" function drawLayers(svg,pp,m,w,structural){")
_end = _html.find("\n function drawWave(svg,pp,m,w){", _start)
if _start < 0 or _end < 0:
    raise RuntimeError("Current dissertation-style drawLayers block was not found.")

_DRAW_LAYERS = r''' function drawLayers(svg,pp,m,w,structural){
  const attacks=(pp?.layer_anchors||[]).filter(r=>Number(r.system_index||0)===m.k).map(r=>({...r,_structural:false}));
  const structuralRows=structural?(pp?.structural_anchors||[]).filter(r=>Number(r.system_index||0)===m.k).map(r=>({...r,_structural:true})):[];
  const rows=[...attacks,...structuralRows].sort((a,b)=>Number(a.cx_norm)-Number(b.cx_norm));
  const xs=rows.map(r=>Number(r.cx_norm)*w);
  for(let i=0;i<rows.length;i++){
    const row=rows[i],x=xs[i],l=i===0?m.left:(xs[i-1]+x)/2,r=i+1===rows.length?m.right:(x+xs[i+1])/2;
    if(!(r>l))continue;
    for(const raw of row.levels||[]){
      const level=Number(raw);if(!Number.isFinite(level))continue;
      const y=yLevel(m,level),shade=Math.max(115,Math.min(240,240-(level-1)*18));
      svg.appendChild(svgEl('rect',{x:l,y:y-m.rowH*.45,width:r-l,height:m.rowH*.9,fill:`rgb(${shade},${shade},${shade})`}));
      text(svg,x,y,row._structural?'('+level+')':String(level),row._structural?'tm-example-struct':'tm-example-level');
    }
  }
 }'''
base.HTML = _html[:_start] + _DRAW_LAYERS + _html[_end:]
