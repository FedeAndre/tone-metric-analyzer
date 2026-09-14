from __future__ import annotations

from pathlib import Path

import fitz

import dissertation_exact_pdf_app as exact

# Presentation-only refinement. Analytical generation remains in the exact-PDF
# wrapper and the dissertation analysis engine; this module only changes where
# the already-computed graphics are drawn and how large their labels appear.
APP_VERSION = "1.1.1-dissertation-system-aligned"
app = exact.app
app.version = APP_VERSION


def _aligned_system_models(overlay_page: dict, page_rect: fitz.Rect) -> dict[int, dict]:
    width, height = float(page_rect.width), float(page_rect.height)
    bounds = {
        int(b.get("system_index", 0)): b
        for b in overlay_page.get("system_bounds", []) or []
    }

    system_ids: set[int] = set(bounds)
    ys: dict[int, list[float]] = {}
    xs: dict[int, list[float]] = {}

    for key in ("layer_anchors", "wave_anchors", "tree_nodes"):
        for row in overlay_page.get(key, []) or []:
            k = int(row.get("system_index", 0))
            system_ids.add(k)
            try:
                y = float(row.get("cy_norm")) * height
            except Exception:
                y = 0.0
            try:
                x = float(row.get("cx_norm")) * width
            except Exception:
                x = 0.0
            if 0.01 * height < y < 0.99 * height:
                ys.setdefault(k, []).append(y)
            if 0.0 < x < width:
                xs.setdefault(k, []).append(x)

    for key in ("structural_anchors", "pivot_anchors", "tree_branches"):
        for row in overlay_page.get(key, []) or []:
            system_ids.add(int(row.get("system_index", 0)))

    models: dict[int, dict] = {}
    y_pad = max(12.0, min(26.0, height * 0.022))

    for sys_idx in sorted(system_ids):
        b = bounds.get(sys_idx)
        by = sorted(ys.get(sys_idx, []))
        bx = sorted(xs.get(sys_idx, []))

        if by:
            top = max(0.0, by[0] - y_pad)
            bottom = min(height, by[-1] + y_pad)
        elif b is not None:
            top = float(b.get("top_norm", 0.0)) * height
            bottom = float(b.get("bottom_norm", b.get("top_norm", 0.0))) * height
        else:
            continue

        if bottom <= top:
            continue

        if b is not None:
            b_left = float(b.get("left_norm", 0.0)) * width
            b_right = float(b.get("right_norm", 1.0)) * width
        else:
            b_left, b_right = 0.0, width

        if b is not None and b_right > b_left:
            left, right = b_left, b_right
        elif bx:
            x_pad = max(8.0, min(22.0, width * 0.015))
            left = max(0.0, bx[0] - x_pad)
            right = min(width, bx[-1] + x_pad)
        else:
            left, right = 0.0, width

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
            vals = []
            for raw in row.get("levels", []) or []:
                try:
                    vals.append(int(raw))
                except Exception:
                    pass
            for field in ("height", "lowest_level"):
                if row.get(field) is not None:
                    try:
                        vals.append(int(row.get(field)))
                    except Exception:
                        pass
            if vals:
                m["max_level"] = max(m["max_level"], max(vals))

    ordered = sorted(models.items(), key=lambda kv: kv[1]["top"])
    prev_bottom = 0.0
    for _, m in ordered:
        intervals = max(1, int(m["max_level"]) - 1)
        band_bottom = max(8.0, m["top"] - 8.0)
        gap_top = prev_bottom + 5.0
        available = max(10.0, band_bottom - gap_top)
        fitted = available / intervals
        row_h = max(8.5, min(12.0, fitted))
        m["row_h"] = row_h
        m["grid_bottom"] = band_bottom
        m["grid_top"] = band_bottom - row_h * intervals
        prev_bottom = m["bottom"]

    return models


def _aligned_write_annotated_pdf(source_pdf: Path, out_pdf: Path, overlay: dict) -> None:
    doc = fitz.open(source_pdf)
    by_page = {int(p.get("page_index", 0)): p for p in overlay.get("pages", []) or []}
    for page_index, page in enumerate(doc):
        op = by_page.get(page_index)
        if not op:
            continue
        rect = page.rect
        width = float(rect.width)
        models = _aligned_system_models(op, rect)

        def level_y(model: dict, level: int) -> float:
            return float(model["grid_bottom"]) - (max(1, int(level)) - 1) * float(model["row_h"])

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
                y = level_y(m, level)
                page.insert_text(
                    (x - 3.0, y + 3.0),
                    str(level),
                    fontsize=8.8,
                    color=(0, 0, 0),
                    overlay=True,
                )

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
                y = level_y(m, level)
                page.insert_text(
                    (x - 4.0, y + 3.0),
                    f"({level})",
                    fontsize=8.0,
                    color=(0.25, 0.25, 0.25),
                    overlay=True,
                )

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
                    y = level_y(m, int(row.get("height", 1) or 1))
                except Exception:
                    continue
                pts.append(fitz.Point(x, y))
            if len(pts) > 1:
                shape = page.new_shape()
                for a, b in zip(pts, pts[1:]):
                    shape.draw_line(a, b)
                shape.finish(color=(0.15, 0.15, 0.15), width=0.9)
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
            y = level_y(m, level)
            page.draw_circle(
                fitz.Point(x, y),
                2.3,
                color=(0, 0, 0),
                fill=(0.45, 0.45, 0.45),
                width=0.6,
                overlay=True,
            )

        for row in op.get("tree_branches", []) or []:
            if row.get("span_role") not in (None, "", "complete"):
                continue
            m = models.get(int(row.get("system_index", 0)))
            if not m:
                continue
            try:
                x1 = float(row.get("source_cx_norm")) * width
                x2 = float(row.get("target_cx_norm")) * width
                y1 = level_y(m, int(row.get("source_level", 1)))
                y2 = level_y(m, int(row.get("target_level", 1)))
            except Exception:
                continue
            page.draw_line(
                fitz.Point(x1, y1),
                fitz.Point(x2, y2),
                color=(0.35, 0.35, 0.35),
                width=0.65,
                overlay=True,
            )

    doc.save(out_pdf, garbage=3, deflate=True)
    doc.close()


exact._system_models = _aligned_system_models
exact._write_annotated_pdf = _aligned_write_annotated_pdf

base = exact.base

_style_replacements = {
    ".tm-level{font:700 9px Arial,sans-serif;": ".tm-level{font:700 14px Arial,sans-serif;",
    ".tm-struct{font:italic 8.5px Arial,sans-serif;": ".tm-struct{font:italic 13px Arial,sans-serif;",
    ".tm-row-label{font:7.5px Arial,sans-serif;": ".tm-row-label{font:11px Arial,sans-serif;",
    ".tm-band-title{font:700 8px Arial,sans-serif;": ".tm-band-title{font:700 12px Arial,sans-serif;",
    ".tm-level-guide{stroke:#111;stroke-width:.55;": ".tm-level-guide{stroke:#111;stroke-width:.75;",
    ".tm-wave{fill:none;stroke:#111;stroke-width:1.15;": ".tm-wave{fill:none;stroke:#111;stroke-width:1.7;",
}
for old, new in _style_replacements.items():
    if base.HTML.count(old) != 1:
        raise RuntimeError(f"Expected exactly one current score-overlay style token: {old}")
    base.HTML = base.HTML.replace(old, new, 1)

start_token = " function buildModels(pp,w,h){"
end_token = "\n function yLevel(m,l){"
start = base.HTML.find(start_token)
end = base.HTML.find(end_token, start)
if start < 0 or end < 0:
    raise RuntimeError("Current exact-PDF buildModels function was not found exactly once.")

new_build_models = r''' function buildModels(pp,w,h){
  const bounds=groupsBySystem((pp?.system_bounds||[]).map(b=>({...b,system_index:Number(b.system_index||0)})));
  const allKeys=new Set();
  for(const b of pp?.system_bounds||[])allKeys.add(Number(b.system_index||0));
  for(const key of ['layer_anchors','structural_anchors','wave_anchors','pivot_anchors','tree_nodes','tree_branches'])
    for(const r of pp?.[key]||[])allKeys.add(Number(r.system_index||0));

  const ys=new Map(),xs=new Map();
  const add=(map,k,v)=>{if(!Number.isFinite(v))return;if(!map.has(k))map.set(k,[]);map.get(k).push(v)};
  for(const key of ['layer_anchors','wave_anchors','tree_nodes']){
    for(const r of pp?.[key]||[]){
      const k=Number(r.system_index||0),y=Number(r.cy_norm)*h,x=Number(r.cx_norm)*w;
      if(y>h*.01&&y<h*.99)add(ys,k,y);
      if(x>0&&x<w)add(xs,k,x);
    }
  }

  const models=new Map(),ypad=Math.max(24,Math.min(54,h*.024)),xpad=Math.max(12,Math.min(30,w*.015));
  for(const k of allKeys){
    const b=(bounds.get(k)||[])[0]||null,yy=[...(ys.get(k)||[])].sort((a,b)=>a-b),xx=[...(xs.get(k)||[])].sort((a,b)=>a-b);
    let top,bottom,left,right;
    if(yy.length){
      top=Math.max(0,yy[0]-ypad);
      bottom=Math.min(h,yy[yy.length-1]+ypad);
    }else if(b){
      top=Number(b.top_norm??0)*h;
      bottom=Number(b.bottom_norm??b.top_norm??0)*h;
    }else continue;

    if(b&&Number(b.right_norm)>Number(b.left_norm)){
      left=Number(b.left_norm??0)*w;
      right=Number(b.right_norm??1)*w;
    }else if(xx.length){
      left=Math.max(0,xx[0]-xpad);
      right=Math.min(w,xx[xx.length-1]+xpad);
    }else{
      left=0;right=w;
    }

    let maxLevel=1;
    for(const key of ['layer_anchors','structural_anchors','wave_anchors','tree_nodes']){
      for(const r of pp?.[key]||[]){
        if(Number(r.system_index||0)!==k)continue;
        for(const x of r.levels||[]){const n=Number(x);if(Number.isFinite(n))maxLevel=Math.max(maxLevel,n)}
        for(const q of [r.height,r.lowest_level]){const n=Number(q);if(Number.isFinite(n))maxLevel=Math.max(maxLevel,n)}
      }
    }
    models.set(k,{k,top,bottom,left,right,maxLevel});
  }

  const ordered=[...models.values()].sort((a,b)=>a.top-b.top);
  let prevBottom=0;
  for(const m of ordered){
    const intervals=Math.max(1,m.maxLevel-1),bandBottom=Math.max(12,m.top-12),gapTop=prevBottom+7,available=Math.max(12,bandBottom-gapTop);
    const fitted=available/intervals;
    m.rowH=Math.max(12,Math.min(18,fitted));
    m.gridBottom=bandBottom;
    m.gridTop=bandBottom-m.rowH*intervals;
    prevBottom=m.bottom;
  }
  return models;
}'''

base.HTML = base.HTML[:start] + new_build_models + base.HTML[end:]
