from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lxml import etree

sys.path.insert(0, "/app")
from core import Box, get_box, loc, _interline, _system_bounds, _head_fill_ratio, _pixel_beam_levels

TARGETS = {7, 28, 34, 39}
OUT = Path("/work/visual_heads")
OUT.mkdir(parents=True, exist_ok=True)


def frac_label(x: Fraction | None) -> str | None:
    return None if x is None else str(x)


def sheet_number(name: str) -> int:
    m = re.search(r"sheet#(\d+)", name, re.I)
    return int(m.group(1)) if m else 10**9


def chord_duration_with_levels(by, chord_heads, chord_stem, flags, dots, head_is_black, cid, levels):
    node = by[cid]
    hs = chord_heads.get(cid, [])
    if not hs:
        return None
    if any(head_is_black.get(h, False) for h in hs):
        base = Fraction(1, 4)
    else:
        shapes = [(by[h].get("shape") or "").upper() for h in hs if h in by]
        base = Fraction(2) if any("BREVE" in s for s in shapes) else Fraction(1) if any("WHOLE" in s for s in shapes) else Fraction(1, 2)
    st = chord_stem.get(cid)
    lev = max(levels.get(st or "", 0), flags.get(st or "", 0))
    d0 = base / (2 ** lev)
    dn = max([dots.get(h, 0) for h in hs] or [0])
    return d0 * sum(Fraction(1, 2**i) for i in range(dn + 1))


def main(omr_path: str):
    results = {}
    global_measure_base = 0

    with ZipFile(omr_path) as archive:
        members = sorted(
            [n for n in archive.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=lambda n: (sheet_number(n), n.lower()),
        )

        for page_index, member in enumerate(members):
            root = etree.fromstring(archive.read(member))
            by = {e.get("id"): e for e in root.iter() if e.get("id")}
            il = _interline(root)
            pic = next((e for e in root.iter() if loc(e.tag) == "picture"), None)
            W = float(pic.get("width"))
            H = float(pic.get("height"))
            binary_name = member.rsplit("/", 1)[0] + "/BINARY.png"
            binary_img = Image.open(BytesIO(archive.read(binary_name))).convert("L")
            binary = np.array(binary_img) < 128

            chord_heads = defaultdict(list)
            chord_stem = {}
            semantic_beams = defaultdict(set)
            flags = defaultdict(int)
            dots = defaultdict(int)
            slur_heads = defaultdict(dict)
            slur_ext = defaultdict(set)
            explicit_ties = set()

            for rel in (e for e in root.iter() if loc(e.tag) == "relation"):
                child = next(iter(rel), None)
                if child is None:
                    continue
                k = loc(child.tag)
                s, t = rel.get("source"), rel.get("target")
                if k == "containment" and s in by and t in by:
                    if loc(by[s].tag) == "head-chord" and loc(by[t].tag) == "head":
                        chord_heads[s].append(t)
                elif k == "chord-stem":
                    chord_stem[s] = t
                elif k == "beam-stem":
                    semantic_beams[t].add(s)
                elif k == "flag-stem":
                    shape = (by.get(s).get("shape") if by.get(s) is not None else "") or ""
                    m = re.search(r"FLAG_(\d+)", shape)
                    flags[t] = max(flags[t], int(m.group(1)) if m else 1)
                elif k == "augmentation":
                    dots[t] += 1
                elif k == "slur-head":
                    sl = by.get(s)
                    side = (child.get("side") or "").upper()
                    if sl is not None and loc(sl.tag) == "slur":
                        slur_heads[s][side] = t
                        if (sl.get("tie") or "").lower() == "true":
                            explicit_ties.add(s)

            for sid, node in by.items():
                if loc(node.tag) == "slur":
                    for attr in ("left-extension", "right-extension"):
                        q = node.get(attr)
                        if q:
                            slur_ext[sid].add(q)
                            slur_ext[q].add(sid)

            tied_right = set()
            tie_starts = set()
            seen = set()
            for sid in slur_heads:
                if sid in seen:
                    continue
                comp = []
                stack = [sid]
                seen.add(sid)
                while stack:
                    q = stack.pop()
                    comp.append(q)
                    for n in slur_ext.get(q, set()):
                        if n not in seen:
                            seen.add(n)
                            stack.append(n)
                lefts = [slur_heads[s]["LEFT"] for s in comp if "LEFT" in slur_heads.get(s, {})]
                rights = [slur_heads[s]["RIGHT"] for s in comp if "RIGHT" in slur_heads.get(s, {})]
                explicit = any(s in explicit_ties for s in comp)
                inferred = False
                if not explicit and len(comp) > 1 and lefts and rights:
                    lp = {by[h].get("pitch") for h in lefts if h in by}
                    rp = {by[h].get("pitch") for h in rights if h in by}
                    inferred = bool(lp & rp)
                if explicit or inferred:
                    tied_right.update(rights)
                    tie_starts.update(lefts)

            head_is_black = {}
            for hid, node in by.items():
                if loc(node.tag) != "head":
                    continue
                b = get_box(node)
                if b is None:
                    continue
                shape = (node.get("shape") or "").upper()
                ratio = _head_fill_ratio(binary, b)
                semantic_black = "BLACK" in shape
                semantic_void = "VOID" in shape
                corrected = semantic_black
                if semantic_void and ratio >= 0.80:
                    corrected = True
                elif semantic_black and ratio <= 0.75:
                    corrected = False
                head_is_black[hid] = corrected

            page_measure_base = global_measure_base

            for sy, system in enumerate([e for e in root.iter() if loc(e.tag) == "system"]):
                top, bottom = _system_bounds(system, H)
                stacks = [e for e in system if loc(e.tag) == "stack"]
                staffids = {e.get("id") for e in system.iter() if loc(e.tag) == "staff" and e.get("id")}
                sys_base = page_measure_base
                bounds = [(float(s.get("left")), float(s.get("right"))) for s in stacks]

                chords = {}
                by_ms = defaultdict(list)
                for cid, node in by.items():
                    if loc(node.tag) != "head-chord" or node.get("staff") not in staffids:
                        continue
                    cb = get_box(node)
                    if cb is None:
                        continue
                    mis = [i for i, (l, r) in enumerate(bounds) if l <= cb.cx <= r]
                    if len(mis) != 1:
                        continue
                    mi = mis[0]
                    hs = chord_heads.get(cid, [])
                    hbs = [get_box(by[h]) for h in hs if h in by and get_box(by[h])]
                    if not hbs:
                        continue
                    attack = [h for h in hs if h not in tied_right]
                    ax = [get_box(by[h]).cx for h in attack if h in by and get_box(by[h])]
                    allx = [b.cx for b in hbs]
                    stem = chord_stem.get(cid)
                    sb = get_box(by.get(stem))
                    direction = None
                    if sb:
                        direction = "up" if sb.cx >= median(allx) else "down"
                    rec = {
                        "id": cid,
                        "global_measure": sys_base + mi + 1,
                        "local_measure": mi + 1,
                        "staff": node.get("staff") or "",
                        "box": cb,
                        "heads": list(hs),
                        "attack_heads": list(attack),
                        "stem": stem,
                        "direction": direction,
                        "x": float(median(ax or allx)),
                    }
                    chords[cid] = rec
                    by_ms[(mi, rec["staff"])].append(rec)

                semantic_edges = {}
                pixel_edges = {}
                for (mi, staff), arr in by_ms.items():
                    arr = sorted([c for c in arr if c["stem"] and c["direction"]], key=lambda c: c["x"])
                    for a, b in zip(arr, arr[1:]):
                        if a["direction"] != b["direction"]:
                            continue
                        key = frozenset((a["id"], b["id"]))
                        common = len(semantic_beams.get(a["stem"], set()) & semantic_beams.get(b["stem"], set()))
                        if common:
                            semantic_edges[key] = common
                        sa, sb = get_box(by.get(a["stem"])), get_box(by.get(b["stem"]))
                        if sa and sb:
                            ta = (sa.cx, sa.y if a["direction"] == "up" else sa.y + sa.h, a["direction"])
                            tb = (sb.cx, sb.y if b["direction"] == "up" else sb.y + sb.h, b["direction"])
                            lv = _pixel_beam_levels(binary, ta, tb, il)
                            if lv:
                                pixel_edges[key] = lv

                levels = {s: len(bs) for s, bs in semantic_beams.items()}
                for pair, lv in pixel_edges.items():
                    # Visual-only completion: use image-detected beams, without any slot timing.
                    for cid in pair:
                        c = chords.get(cid)
                        if c and c["stem"]:
                            levels[c["stem"]] = max(levels.get(c["stem"], 0), lv)

                for c in chords.values():
                    c["duration"] = chord_duration_with_levels(
                        by, chord_heads, chord_stem, flags, dots, head_is_black, c["id"], levels
                    )

                for mi, (left, right) in enumerate(bounds):
                    global_measure = sys_base + mi + 1
                    if global_measure not in TARGETS:
                        continue

                    measure_chords = [c for c in chords.values() if c["global_measure"] == global_measure]
                    measure_chords.sort(key=lambda c: (c["x"], c["staff"], c["id"]))

                    rows = []
                    for c in measure_chords:
                        head_rows = []
                        for hid in c["heads"]:
                            hb = get_box(by.get(hid))
                            if hb is None:
                                continue
                            head_rows.append({
                                "id": hid,
                                "x": round(hb.cx, 3),
                                "y": round(hb.cy, 3),
                                "shape": (by[hid].get("shape") or ""),
                                "pitch": by[hid].get("pitch"),
                                "tied_continuation": hid in tied_right,
                                "tie_start": hid in tie_starts,
                                "attack": hid not in tied_right,
                            })
                        rows.append({
                            "chord_id": c["id"],
                            "staff": c["staff"],
                            "x": round(c["x"], 3),
                            "x_normalized": round((c["x"] - left) / max(1.0, right - left), 6),
                            "direction": c["direction"],
                            "duration_whole_note_units": frac_label(c["duration"]),
                            "duration_quarter_units": frac_label(c["duration"] * 4 if c["duration"] is not None else None),
                            "head_count": len(head_rows),
                            "attack_head_count": sum(h["attack"] for h in head_rows),
                            "heads": head_rows,
                        })

                    # Create an auditable image crop with notehead dots only.
                    x0 = max(0, int(round(left - 1.0 * il)))
                    x1 = min(int(W), int(round(right + 1.0 * il)))
                    y0 = max(0, int(round(top - 2.5 * il)))
                    y1 = min(int(H), int(round(bottom + 2.5 * il)))
                    crop_gray = np.where(binary[y0:y1, x0:x1], 0, 255).astype(np.uint8)
                    crop = Image.fromarray(crop_gray, mode="L").convert("RGB")

                    header_h = 88
                    canvas = Image.new("RGB", (crop.width, crop.height + header_h), "white")
                    canvas.paste(crop, (0, header_h))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((12, 8), f"Measure {global_measure}: visual notehead detections; no Audiveris timing used", fill=(0, 0, 0))
                    draw.text((12, 32), "green = attack representative   blue = same attack/chord   yellow = tied continuation   red = unresolved duration", fill=(0, 0, 0))
                    draw.text((12, 56), f"bar x={left:.1f}..{right:.1f}  interline={il:.1f}", fill=(0, 0, 0))

                    radius = max(6, int(round(0.34 * il)))
                    width = max(3, int(round(0.12 * il)))

                    for c in measure_chords:
                        hdata = []
                        for hid in c["heads"]:
                            hb = get_box(by.get(hid))
                            if hb is not None:
                                hdata.append((hid, hb))
                        attack_heads = [(hid, hb) for hid, hb in hdata if hid not in tied_right]
                        tied_heads = [(hid, hb) for hid, hb in hdata if hid in tied_right]

                        if c["duration"] is None:
                            for hid, hb in hdata:
                                cx, cy = hb.cx - x0, hb.cy - y0 + header_h
                                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(220, 0, 0), width=width)
                        else:
                            # One green marker means one TMA attack candidate; other noteheads in the chord are blue.
                            if attack_heads:
                                rep_i = min(range(len(attack_heads)), key=lambda i: abs(attack_heads[i][1].cy - median([b.cy for _, b in attack_heads])))
                                for i, (hid, hb) in enumerate(attack_heads):
                                    cx, cy = hb.cx - x0, hb.cy - y0 + header_h
                                    color = (0, 170, 0) if i == rep_i else (0, 105, 230)
                                    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=width)
                            for hid, hb in tied_heads:
                                cx, cy = hb.cx - x0, hb.cy - y0 + header_h
                                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(240, 175, 0), width=width)

                    out_img = OUT / f"measure_{global_measure}_visual_heads.png"
                    canvas.save(out_img)

                    results[str(global_measure)] = {
                        "page": page_index + 1,
                        "system": sy + 1,
                        "bar_bounds_x": [left, right],
                        "system_bounds_y": [top, bottom],
                        "interline": il,
                        "chords": rows,
                    }

                page_measure_base += len(stacks)

            global_measure_base = page_measure_base

    (OUT / "visual_head_diagnostics.json").write_text(json.dumps(results, indent=2))
    for m in sorted(TARGETS):
        r = results.get(str(m))
        if not r:
            print(f"VISUAL_MEASURE m{m}: MISSING")
            continue
        compact = [
            {
                "xnorm": c["x_normalized"],
                "dur_q": c["duration_quarter_units"],
                "heads": c["head_count"],
                "attack_heads": c["attack_head_count"],
                "staff": c["staff"],
            }
            for c in r["chords"]
        ]
        print(f"VISUAL_MEASURE m{m}: " + json.dumps(compact, separators=(",", ":")))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: visual_heads.py SCORE.omr")
    main(sys.argv[1])
