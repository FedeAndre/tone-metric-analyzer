from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import fitz

import dissertation_exact_pdf_app_v2 as current

# Presentation-only layer. The validated analysis remains in dissertation_app.py
# and the single Audiveris pass / exact uploaded-PDF machinery remains in
# dissertation_exact_pdf_app_v2.py. This module changes only the score presentation.
base = current.base
old = current.old
app = current.app
APP_VERSION = "1.3.0-dissertation-example-layout"
app.version = APP_VERSION

_OBSOLETE_LAYOUT_WARNING = (
    "The MusicXML export did not provide usable note layout coordinates, "
    "so coloring on the original PDF may be unavailable or approximate for this score."
)
_original_warnings = base._warnings


def _warnings_current_pdf_pipeline(result: dict, parser_warnings: list[str]) -> list[str]:
    return [
        w for w in _original_warnings(result, parser_warnings)
        if str(w).strip() != _OBSOLETE_LAYOUT_WARNING
    ]


base._warnings = _warnings_current_pdf_pipeline


def _chronological_key(row: dict) -> tuple:
    try:
        measure = int(row.get("measure_index", -1))
    except Exception:
        measure = -1
    try:
        event = int(row.get("event_index", -1))
    except Exception:
        event = -1
    try:
        onset = float(row.get("onset_quarter", 0) or 0)
    except Exception:
        onset = 0.0
    return measure, event, onset


def _recover_visual_systems(page: dict) -> None:
    """Presentation-only recovery of printed score-system membership.

    Saved Audiveris geometry is authoritative when it already yields multiple
    systems. If it collapses a page to one system, chronological x resets recover
    the printed lines without changing score time or analytical values.
    """
    layers = list(page.get("layer_anchors", []) or [])
    if len(layers) < 2:
        return
    existing = {int(r.get("system_index", 0)) for r in layers}
    if len(existing) > 1:
        return

    ordered = sorted(layers, key=_chronological_key)
    groups: list[list[dict]] = [[]]
    prev_x = None
    prev_y = None
    for row in ordered:
        try:
            x = float(row.get("cx_norm", 0.0))
        except Exception:
            x = 0.0
        try:
            y = float(row.get("cy_norm", 0.0))
        except Exception:
            y = 0.0
        new_line = False
        if prev_x is not None:
            if (prev_x - x) >= 0.22 and prev_x >= 0.48 and x <= 0.64:
                new_line = True
            elif prev_y is not None and abs(y - prev_y) >= 0.07 and (prev_x - x) >= 0.09:
                new_line = True
        if new_line and groups[-1]:
            groups.append([])
        groups[-1].append(row)
        prev_x, prev_y = x, y

    groups = [g for g in groups if g]
    if len(groups) <= 1:
        return

    event_to_system: dict[int, int] = {}
    measure_votes: dict[int, Counter] = defaultdict(Counter)
    system_centers: dict[int, float] = {}
    for sys_idx, rows in enumerate(groups):
        ys = []
        for row in rows:
            row["system_index"] = sys_idx
            row["physical_system_index"] = sys_idx
            try:
                event_to_system[int(row.get("event_index", -1))] = sys_idx
            except Exception:
                pass
            try:
                measure_votes[int(row.get("measure_index", -1))][sys_idx] += 1
            except Exception:
                pass
            try:
                y = float(row.get("cy_norm", 0.0))
                if 0.0 < y < 1.0:
                    ys.append(y)
            except Exception:
                pass
        system_centers[sys_idx] = sum(ys) / len(ys) if ys else (sys_idx + 0.5) / len(groups)

    measure_to_system = {
        measure: votes.most_common(1)[0][0]
        for measure, votes in measure_votes.items()
        if votes
    }

    def choose_system(row: dict) -> int:
        for field, mapping in (("event_index", event_to_system), ("measure_index", measure_to_system)):
            try:
                key = int(row.get(field, -1))
                if key in mapping:
                    return mapping[key]
            except Exception:
                pass
        try:
            y = float(row.get("cy_norm", 0.0))
            if 0.0 < y < 1.0:
                return min(system_centers, key=lambda k: abs(system_centers[k] - y))
        except Exception:
            pass
        return 0

    for key in ("structural_anchors", "wave_anchors", "pivot_anchors", "tree_nodes", "tree_branches"):
        for row in page.get(key, []) or []:
            sys_idx = choose_system(row)
            row["system_index"] = sys_idx
            if "physical_system_index" in row:
                row["physical_system_index"] = sys_idx

    bounds = []
    all_rows = []
    for key in ("layer_anchors", "wave_anchors", "tree_nodes"):
        all_rows.extend(page.get(key, []) or [])
    for sys_idx in range(len(groups)):
        rows = [r for r in all_rows if int(r.get("system_index", 0)) == sys_idx]
        xs, ys = [], []
        for row in rows:
            try:
                x = float(row.get("cx_norm", 0.0))
                if 0.0 <= x <= 1.0:
                    xs.append(x)
            except Exception:
                pass
            try:
                y = float(row.get("cy_norm", 0.0))
                if 0.0 < y < 1.0:
                    ys.append(y)
            except Exception:
                pass
        if not ys:
            ys = [system_centers[sys_idx]]
        bounds.append({
            "system_index": sys_idx,
            "top_norm": max(0.0, min(ys) - 0.035),
            "bottom_norm": min(1.0, max(ys) + 0.035),
            "guard_bottom_norm": min(1.0, max(ys) + 0.035),
            "left_norm": max(0.0, min(xs) - 0.018) if xs else 0.04,
            "right_norm": min(1.0, max(xs) + 0.018) if xs else 0.96,
        })
    bounds.sort(key=lambda r: r["top_norm"])
    page["system_bounds"] = bounds


_original_canonical_overlay = current._canonical_overlay


def _canonical_overlay_per_printed_line(levels_payload: dict, canonical_meta: dict) -> dict:
    overlay = _original_canonical_overlay(levels_payload, canonical_meta)
    for page in overlay.get("pages", []) or []:
        _recover_visual_systems(page)
    matching = dict(overlay.get("matching", {}) or {})
    matching["presentation_system_alignment"] = "one-analysis-band-per-printed-score-system"
    matching["presentation_style"] = "dissertation-example-grey-level-bands-step-wave-pivot-bars"
    overlay["matching"] = matching
    return overlay


current._canonical_overlay = _canonical_overlay_per_printed_line


def _pdf_system_models(overlay_page: dict, page_rect: fitz.Rect) -> list[dict]:
    w, h = float(page_rect.width), float(page_rect.height)
    bounds = {
        int(b.get("system_index", 0)): b
        for b in overlay_page.get("system_bounds", []) or []
    }
    keys = set(bounds)
    for field in ("layer_anchors", "structural_anchors", "wave_anchors", "pivot_anchors", "tree_nodes", "tree_branches"):
        for row in overlay_page.get(field, []) or []:
            keys.add(int(row.get("system_index", 0)))

    models = []
    for k in sorted(keys):
        b = bounds.get(k, {})
        rows = [r for r in overlay_page.get("layer_anchors", []) or [] if int(r.get("system_index", 0)) == k]
        ys = []
        xs = []
        max_level = 1
        for field in ("layer_anchors", "structural_anchors", "wave_anchors", "tree_nodes"):
            for row in overlay_page.get(field, []) or []:
                if int(row.get("system_index", 0)) != k:
                    continue
                try:
                    x = float(row.get("cx_norm", 0.0)) * w
                    if 0 <= x <= w:
                        xs.append(x)
                except Exception:
                    pass
                try:
                    y = float(row.get("cy_norm", 0.0)) * h
                    if 0 < y < h:
                        ys.append(y)
                except Exception:
                    pass
                for raw in row.get("levels", []) or []:
                    try:
                        max_level = max(max_level, int(raw))
                    except Exception:
                        pass
                for name in ("height", "lowest_level"):
                    try:
                        if row.get(name) is not None:
                            max_level = max(max_level, int(row.get(name)))
                    except Exception:
                        pass
        if b:
            top = float(b.get("top_norm", 0.0)) * h
            bottom = float(b.get("bottom_norm", b.get("top_norm", 0.0))) * h
            left = float(b.get("left_norm", 0.0)) * w
            right = float(b.get("right_norm", 1.0)) * w
        elif ys:
            top, bottom = min(ys), max(ys)
            left, right = (min(xs), max(xs)) if xs else (0.0, w)
        else:
            continue
        pad = max(24.0, h * 0.025)
        band_row_h = 14.0
        levels_top = 34.0
        pivot_y = levels_top + max_level * band_row_h + 8.0
        band_h = pivot_y + 20.0
        models.append({
            "system_index": k,
            "top": top,
            "bottom": bottom,
            "left": max(0.0, left),
            "right": min(w, right),
            "max_level": max_level,
            "pad": pad,
            "row_h": band_row_h,
            "levels_top": levels_top,
            "pivot_y": pivot_y,
            "band_h": band_h,
        })
    models.sort(key=lambda m: m["top"])
    return models


def _pdf_strip_ranges(models: list[dict], page_height: float) -> tuple[float, list[tuple[dict, float, float]]]:
    if not models:
        return 0.0, []
    expanded_top = [max(0.0, m["top"] - m["pad"]) for m in models]
    expanded_bottom = [min(page_height, m["bottom"] + m["pad"]) for m in models]
    boundaries = [
        max(expanded_bottom[i], min(expanded_top[i + 1], (expanded_bottom[i] + expanded_top[i + 1]) / 2.0))
        if expanded_top[i + 1] >= expanded_bottom[i]
        else (models[i]["bottom"] + models[i + 1]["top"]) / 2.0
        for i in range(len(models) - 1)
    ]
    first = expanded_top[0]
    rows = []
    start = first
    for i, m in enumerate(models):
        end = boundaries[i] if i < len(boundaries) else page_height
        end = max(start + 1.0, min(page_height, end))
        rows.append((m, start, end))
        start = end
    return first, rows


def _pdf_level_y(m: dict, level: int, band_top: float) -> float:
    return band_top + m["levels_top"] + (m["max_level"] - int(level)) * m["row_h"] + m["row_h"] * 0.5


def _pdf_measure_centers(op: dict, system_index: int, width: float) -> list[tuple[str, float]]:
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for field in ("layer_anchors", "structural_anchors", "wave_anchors"):
        for row in op.get(field, []) or []:
            if int(row.get("system_index", 0)) != system_index:
                continue
            try:
                mi = int(row.get("measure_index", -1))
                mn = str(row.get("measure_number", mi + 1))
                x = float(row.get("cx_norm", 0.0)) * width
            except Exception:
                continue
            groups[(mi, mn)].append(x)
    out = []
    for (mi, mn), xs in sorted(groups.items()):
        if xs:
            out.append((mn, (min(xs) + max(xs)) / 2.0))
    return out


def _pdf_draw_band(page: fitz.Page, op: dict, m: dict, band_top: float, width: float) -> None:
    left_label_x = max(6.0, m["left"] - 70.0)
    page.insert_text((left_label_x, band_top + 13.0), "Measures", fontsize=10.5, color=(0, 0, 0), overlay=True)
    for number, x in _pdf_measure_centers(op, m["system_index"], width):
        page.insert_text((x - 3.0, band_top + 13.0), number, fontsize=10.5, color=(0, 0, 0), overlay=True)
    page.insert_text((left_label_x + 16.0, band_top + 28.0), "Levels", fontsize=8.5, color=(0, 0, 0), overlay=True)

    for level in range(m["max_level"], 0, -1):
        y = _pdf_level_y(m, level, band_top)
        page.insert_text((left_label_x + 28.0, y + 3.0), str(level), fontsize=8.5, color=(0, 0, 0), overlay=True)

    attacks = [
        r for r in op.get("layer_anchors", []) or []
        if int(r.get("system_index", 0)) == m["system_index"]
    ]
    attacks.sort(key=lambda r: float(r.get("cx_norm", 0.0)))
    xs = [float(r.get("cx_norm", 0.0)) * width for r in attacks]
    for i, row in enumerate(attacks):
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
            y = _pdf_level_y(m, level, band_top)
            shade = max(0.48, min(0.93, 0.94 - (level - 1) * 0.07))
            page.draw_rect(
                fitz.Rect(l, y - m["row_h"] * 0.45, r, y + m["row_h"] * 0.45),
                color=None,
                fill=(shade, shade, shade),
                overlay=True,
            )
            page.insert_text((x - 2.5, y + 3.0), str(level), fontsize=8.5, color=(0, 0, 0), overlay=True)

    for row in op.get("structural_anchors", []) or []:
        if int(row.get("system_index", 0)) != m["system_index"]:
            continue
        try:
            x = float(row.get("cx_norm", 0.0)) * width
        except Exception:
            continue
        for raw in row.get("levels", []) or []:
            try:
                level = int(raw)
            except Exception:
                continue
            y = _pdf_level_y(m, level, band_top)
            page.insert_text((x - 4.0, y + 3.0), f"({level})", fontsize=7.8, color=(0.18, 0.18, 0.18), overlay=True)

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
            y = _pdf_level_y(m, level, band_top) - m["row_h"] * 0.45
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

    # Preserve the existing tree feature in the same analytical band.
    for row in op.get("tree_branches", []) or []:
        if int(row.get("system_index", 0)) != m["system_index"] or row.get("span_role") not in (None, "", "complete"):
            continue
        try:
            x1 = float(row.get("source_cx_norm")) * width
            x2 = float(row.get("target_cx_norm")) * width
            y1 = _pdf_level_y(m, int(row.get("source_level", 1)), band_top)
            y2 = _pdf_level_y(m, int(row.get("target_level", 1)), band_top)
        except Exception:
            continue
        page.draw_line(fitz.Point(x1, y1), fitz.Point(x2, y2), color=(0.35, 0.35, 0.35), width=0.45, overlay=True)


def _write_reflowed_annotated_pdf(source_pdf: Path, out_pdf: Path, overlay: dict) -> None:
    """Preserve every original PDF pixel/vector region, inserting analysis bands.

    The original page is partitioned into non-overlapping horizontal strips. Each
    strip is copied exactly once, and a dissertation-style analysis band is inserted
    immediately before each printed score system. The analysis itself is unchanged.
    """
    src = fitz.open(source_pdf)
    out = fitz.open()
    by_page = {int(p.get("page_index", 0)): p for p in overlay.get("pages", []) or []}
    for page_index, src_page in enumerate(src):
        op = by_page.get(page_index)
        w, h = float(src_page.rect.width), float(src_page.rect.height)
        if not op:
            dst = out.new_page(width=w, height=h)
            dst.show_pdf_page(dst.rect, src, page_index)
            continue
        models = _pdf_system_models(op, src_page.rect)
        first, strips = _pdf_strip_ranges(models, h)
        if not strips:
            dst = out.new_page(width=w, height=h)
            dst.show_pdf_page(dst.rect, src, page_index)
            continue
        total_h = h + sum(float(m["band_h"]) for m, _a, _b in strips)
        dst = out.new_page(width=w, height=total_h)
        yout = 0.0
        if first > 0:
            dst.show_pdf_page(fitz.Rect(0, 0, w, first), src, page_index, clip=fitz.Rect(0, 0, w, first))
            yout = first
        for m, start, end in strips:
            _pdf_draw_band(dst, op, m, yout, w)
            yout += float(m["band_h"])
            strip_h = end - start
            if strip_h > 0:
                dst.show_pdf_page(
                    fitz.Rect(0, yout, w, yout + strip_h),
                    src,
                    page_index,
                    clip=fitz.Rect(0, start, w, end),
                )
                yout += strip_h
    out.save(out_pdf, garbage=3, deflate=True)
    out.close()
    src.close()


old._write_annotated_pdf = _write_reflowed_annotated_pdf

_STYLE = r'''<style id="tm-dissertation-example-style">
.tm-score-surface{background:#fff}.tm-score-strip{display:block;background-repeat:no-repeat;background-color:#fff}.tm-analysis-band{display:block;background:#fff;position:relative}.tm-analysis-band svg{display:block;background:#fff;border:0;border-radius:0;overflow:visible}.tm-example-measures{font:700 15px Georgia,serif;fill:#111}.tm-example-measure-number{font:700 15px Georgia,serif;fill:#111;text-anchor:middle}.tm-example-label{font:700 12px Georgia,serif;fill:#111}.tm-example-row{font:700 11px Georgia,serif;fill:#111;text-anchor:middle;dominant-baseline:middle}.tm-example-level{font:700 11px Georgia,serif;fill:#111;text-anchor:middle;dominant-baseline:middle}.tm-example-struct{font:700 10px Georgia,serif;fill:#111;text-anchor:middle;dominant-baseline:middle}.tm-example-wave{fill:none;stroke:#222;stroke-width:1.4;stroke-linejoin:miter;stroke-linecap:square}.tm-example-tree{fill:none;stroke:#555;stroke-width:.9;opacity:.72}.tm-example-pivot-simple{fill:#c6c6c6}.tm-example-pivot-compound{fill:#929292}
</style>'''
base.HTML = base.HTML.replace("</head>", _STYLE + "</head>", 1)

_start = base.HTML.find(" function buildModels(pp,w,h){")
_end = base.HTML.find("\n function wire(state){", _start)
if _start < 0 or _end < 0:
    raise RuntimeError("Current exact-PDF renderer block was not found for replacement.")

_RENDERER = r''' function buildModels(pp,w,h){
  const bounds=groupsBySystem((pp?.system_bounds||[]).map(b=>({...b,system_index:Number(b.system_index||0)})));
  const keys=new Set();
  for(const b of pp?.system_bounds||[])keys.add(Number(b.system_index||0));
  for(const f of ['layer_anchors','structural_anchors','wave_anchors','pivot_anchors','tree_nodes','tree_branches'])for(const r of pp?.[f]||[])keys.add(Number(r.system_index||0));
  const out=[];
  for(const k of keys){
    const b=(bounds.get(k)||[])[0]||{};
    let maxLevel=1;const xs=[],ys=[];
    for(const f of ['layer_anchors','structural_anchors','wave_anchors','tree_nodes'])for(const r of pp?.[f]||[]){if(Number(r.system_index||0)!==k)continue;const x=Number(r.cx_norm)*w,y=Number(r.cy_norm)*h;if(Number.isFinite(x))xs.push(x);if(Number.isFinite(y)&&y>0&&y<h)ys.push(y);for(const q of r.levels||[]){const n=Number(q);if(Number.isFinite(n))maxLevel=Math.max(maxLevel,n)}for(const q of [r.height,r.lowest_level]){const n=Number(q);if(Number.isFinite(n))maxLevel=Math.max(maxLevel,n)}}
    let top=Number(b.top_norm)*h,bottom=Number(b.bottom_norm)*h,left=Number(b.left_norm)*w,right=Number(b.right_norm)*w;
    if(!Number.isFinite(top))top=ys.length?Math.min(...ys):0;if(!Number.isFinite(bottom))bottom=ys.length?Math.max(...ys):top+1;if(!Number.isFinite(left))left=xs.length?Math.min(...xs):0;if(!Number.isFinite(right))right=xs.length?Math.max(...xs):w;
    const rowH=18,levelsTop=42,pivotY=levelsTop+maxLevel*rowH+10,bandH=pivotY+24,pad=Math.max(34,h*.026);
    out.push({k,top,bottom,left:Math.max(0,left),right:Math.min(w,right),maxLevel,rowH,levelsTop,pivotY,bandH,pad});
  }
  out.sort((a,b)=>a.top-b.top);return out;
 }
 function stripLayout(models,h){if(!models.length)return{header:h,strips:[]};const et=models.map(m=>Math.max(0,m.top-m.pad)),eb=models.map(m=>Math.min(h,m.bottom+m.pad)),bounds=[];for(let i=0;i<models.length-1;i++){let q=et[i+1]>=eb[i]?(eb[i]+et[i+1])/2:(models[i].bottom+models[i+1].top)/2;bounds.push(Math.max(0,Math.min(h,q)))}const first=et[0],strips=[];let start=first;for(let i=0;i<models.length;i++){let end=i<bounds.length?bounds[i]:h;end=Math.max(start+1,Math.min(h,end));strips.push({m:models[i],start,end});start=end}return{header:first,strips}}
 function yLevel(m,l){return m.levelsTop+(m.maxLevel-(Number(l)||1))*m.rowH+m.rowH/2}
 function text(svg,x,y,value,cls,anchor){const e=svgEl('text',{x,y,class:cls});if(anchor)e.setAttribute('text-anchor',anchor);e.textContent=value;svg.appendChild(e);return e}
 function measureCenters(pp,m,w){const g=new Map();for(const f of ['layer_anchors','structural_anchors','wave_anchors'])for(const r of pp?.[f]||[]){if(Number(r.system_index||0)!==m.k)continue;const mi=Number(r.measure_index??-1),mn=String(r.measure_number??(mi+1)),x=Number(r.cx_norm)*w;if(!Number.isFinite(x))continue;const key=mi+'|'+mn;if(!g.has(key))g.set(key,{mi,mn,xs:[]});g.get(key).xs.push(x)}return[...g.values()].sort((a,b)=>a.mi-b.mi).map(q=>({n:q.mn,x:(Math.min(...q.xs)+Math.max(...q.xs))/2}))}
 function drawFrame(svg,pp,m,w,showPivots){const lx=Math.max(8,m.left-95);text(svg,lx,20,'Measures','tm-example-measures');for(const q of measureCenters(pp,m,w))text(svg,q.x,20,q.n,'tm-example-measure-number');text(svg,lx+20,38,'Levels','tm-example-label');for(let l=m.maxLevel;l>=1;l--)text(svg,lx+28,yLevel(m,l),String(l),'tm-example-row');if(showPivots)text(svg,lx+20,m.pivotY+5,'Pivots','tm-example-label')}
 function drawLayers(svg,pp,m,w,structural){const rows=(pp?.layer_anchors||[]).filter(r=>Number(r.system_index||0)===m.k).sort((a,b)=>Number(a.cx_norm)-Number(b.cx_norm)),xs=rows.map(r=>Number(r.cx_norm)*w);for(let i=0;i<rows.length;i++){const row=rows[i],x=xs[i],l=i===0?m.left:(xs[i-1]+x)/2,r=i+1===rows.length?m.right:(x+xs[i+1])/2;if(!(r>l))continue;for(const raw of row.levels||[]){const level=Number(raw);if(!Number.isFinite(level))continue;const y=yLevel(m,level),shade=Math.max(115,Math.min(240,240-(level-1)*18));svg.appendChild(svgEl('rect',{x:l,y:y-m.rowH*.45,width:r-l,height:m.rowH*.9,fill:`rgb(${shade},${shade},${shade})`}));text(svg,x,y,String(level),'tm-example-level')}}if(structural)for(const row of pp?.structural_anchors||[]){if(Number(row.system_index||0)!==m.k)continue;const x=Number(row.cx_norm)*w;if(!Number.isFinite(x))continue;for(const raw of row.levels||[]){const level=Number(raw);if(Number.isFinite(level))text(svg,x,yLevel(m,level),'('+level+')','tm-example-struct')}}}
 function drawWave(svg,pp,m,w){const rows=(pp?.wave_anchors||[]).filter(r=>Number(r.system_index||0)===m.k).sort((a,b)=>Number(a.cx_norm)-Number(b.cx_norm));if(!rows.length)return;const xs=rows.map(r=>Number(r.cx_norm)*w);let d='';let py=null;for(let i=0;i<rows.length;i++){const x=xs[i],l=i===0?m.left:(xs[i-1]+x)/2,r=i+1===rows.length?m.right:(x+xs[i+1])/2,level=Number(rows[i].height||1),y=yLevel(m,level)-m.rowH*.45;if(py===null)d=`M ${l} ${y} L ${r} ${y}`;else d+=` L ${l} ${py} L ${l} ${y} L ${r} ${y}`;py=y}svg.appendChild(svgEl('path',{d,class:'tm-example-wave'}))}
 function drawPivots(svg,pp,m,w){for(const r of pp?.pivot_anchors||[]){if(Number(r.system_index||0)!==m.k)continue;const role=r.span_role||'complete';let x1,x2;if(role==='complete'){x1=Number(r.start_cx_norm)*w;x2=Number(r.end_cx_norm)*w}else{const x=Number(r.cx_norm)*w;if(role==='start-endpoint'){x1=x;x2=Math.min(m.right,x+34)}else{x1=Math.max(m.left,x-34);x2=x}}if(![x1,x2].every(Number.isFinite))continue;svg.appendChild(svgEl('rect',{x:Math.min(x1,x2),y:m.pivotY-7,width:Math.max(5,Math.abs(x2-x1)),height:14,class:String(r.kind||'').toLowerCase()==='compound'?'tm-example-pivot-compound':'tm-example-pivot-simple'}))}}
 function drawTrees(svg,pp,m,w){for(const r of pp?.tree_branches||[]){if(Number(r.system_index||0)!==m.k||(r.span_role&&r.span_role!=='complete'))continue;const x1=Number(r.source_cx_norm)*w,x2=Number(r.target_cx_norm)*w,y1=yLevel(m,Number(r.source_level||1)),y2=yLevel(m,Number(r.target_level||1));if([x1,x2,y1,y2].every(Number.isFinite))svg.appendChild(svgEl('line',{x1,y1,x2,y2,class:'tm-example-tree'}))}}
 function bandSvg(pp,m,w,mode){const svg=svgEl('svg',{viewBox:`0 0 ${w} ${m.bandH}`,width:w,height:m.bandH,preserveAspectRatio:'none'}),showLayers=['attacks','full','fullwave','fullanalysis','fulltree'].includes(mode),structural=['full','fullwave','fullanalysis','fulltree'].includes(mode),showWave=['waves','fullwave','wavepivots','fullanalysis','fulltree'].includes(mode),showPivots=['wavepivots','fullanalysis','fulltree'].includes(mode),showTrees=['trees','fulltree'].includes(mode);drawFrame(svg,pp,m,w,showPivots);if(showLayers)drawLayers(svg,pp,m,w,structural);if(showWave)drawWave(svg,pp,m,w);if(showPivots)drawPivots(svg,pp,m,w);if(showTrees)drawTrees(svg,pp,m,w);return svg}
 function scoreStrip(url,w,h,start,end){const d=document.createElement('div');d.className='tm-score-strip';d.style.width=w+'px';d.style.height=Math.max(1,end-start)+'px';d.style.backgroundImage=`url("${url}")`;d.style.backgroundSize=`${w}px ${h}px`;d.style.backgroundPosition=`0px -${start}px`;return d}
 function renderPage(state){const page=state.pages[state.pageIndex],pp=(state.overlay.pages||[]).find(p=>Number(p.page_index)===Number(page.index))||{},w=Number(page.width),h=Number(page.height),surface=document.getElementById('tm-score-surface'),mode=state.mode;surface.innerHTML='';surface.style.width=w+'px';if(mode==='none'){const img=document.createElement('img');img.src=page.url;img.alt='Exact uploaded PDF page '+(state.pageIndex+1);img.width=w;img.height=h;surface.appendChild(img);surface.style.height=h+'px'}else{const models=buildModels(pp,w,h),layout=stripLayout(models,h);let total=0;if(layout.header>0){surface.appendChild(scoreStrip(page.url,w,h,0,layout.header));total+=layout.header}for(const s of layout.strips){const band=document.createElement('div');band.className='tm-analysis-band';band.style.width=w+'px';band.style.height=s.m.bandH+'px';band.appendChild(bandSvg(pp,s.m,w,mode));surface.appendChild(band);total+=s.m.bandH;surface.appendChild(scoreStrip(page.url,w,h,s.start,s.end));total+=s.end-s.start}if(!layout.strips.length){const img=document.createElement('img');img.src=page.url;img.width=w;img.height=h;surface.appendChild(img);total=h}surface.style.height=total+'px'}surface.style.transform=`scale(${state.zoom/100})`;surface.style.marginRight=Math.max(0,(state.zoom/100-1)*w)+'px';surface.style.marginBottom=Math.max(0,(state.zoom/100-1)*Number.parseFloat(surface.style.height||h))+'px';document.getElementById('tm-page-label').textContent=`Page ${state.pageIndex+1} / ${state.pages.length}`;document.getElementById('tm-page-prev').disabled=state.pageIndex===0;document.getElementById('tm-page-next').disabled=state.pageIndex===state.pages.length-1;document.querySelectorAll('#tm-score-toolbar button[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));const legend={attacks:'Original score systems with dissertation-style Levels above each line.',full:'Original score systems with complete recursive Levels; structural positions remain parenthetical.',waves:'Original score systems with the wave drawn as a stepped contour above each line.',fullwave:'Levels + stepped wave above each score line.',wavepivots:'Stepped wave + dissertation-style pivot bars above each score line.',fullanalysis:'Levels + structural positions + stepped wave + pivot bars above each score line.',trees:'Tree relationships in the analytical band above each score line.',fulltree:'Complete Levels / wave / pivots / tree display above each score line.',none:'Exact uploaded PDF with no analysis layer.'};document.getElementById('tm-score-legend').textContent=legend[mode]||''}
'''

base.HTML = base.HTML[:_start] + _RENDERER + base.HTML[_end:]
