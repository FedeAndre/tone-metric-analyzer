from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import json
import re

import numpy as np
from PIL import Image
from lxml import etree

from core import (
    Box,
    get_box,
    loc,
    _interline,
    _system_bounds,
    _head_fill_ratio,
    _pixel_beam_levels,
    _rest_duration,
)


def _sheet_number(name: str) -> int:
    m = re.search(r"sheet#(\d+)", name, re.I)
    return int(m.group(1)) if m else 10**9


def _fraction_text(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def extract_visual_attacks(omr_path: str | Path):
    """
    Reconstruct attacks without using Audiveris slot/time-offset timing.

    Used evidence:
    - page/system/measure geometry
    - notehead/chord/rest objects
    - stem, beam, flag, augmentation-dot relations
    - tie/slur geometry
    - Audiveris voice membership only as object grouping, never as timing
    - x-order inside each voice

    Each voice is rebuilt from duration accumulation beginning at measure time 0.
    A voice is accepted only if its total reconstructed duration matches the
    notated/nominal measure length exactly. Anything else is diagnostic failure.
    """
    omr_path = Path(omr_path)
    measures_out = {}
    global_measure_base = 0

    with ZipFile(omr_path) as archive:
        members = sorted(
            [n for n in archive.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=lambda n: (_sheet_number(n), n.lower()),
        )
        if not members:
            raise ValueError("Audiveris project contains no sheet XML")

        for page_index, member in enumerate(members):
            root = etree.fromstring(archive.read(member))
            by = {e.get("id"): e for e in root.iter() if e.get("id")}
            il = _interline(root)

            pic = next((e for e in root.iter() if loc(e.tag) == "picture"), None)
            if pic is None:
                raise ValueError(f"{member}: missing picture geometry")
            W = float(pic.get("width") or 0)
            H = float(pic.get("height") or 0)
            binary_name = member.rsplit("/", 1)[0] + "/BINARY.png"
            binary = np.array(Image.open(BytesIO(archive.read(binary_name))).convert("L")) < 128

            chord_heads = defaultdict(list)
            chord_stem = {}
            semantic_beams = defaultdict(set)
            flags = defaultdict(int)
            dots = defaultdict(int)
            rest_child = {}
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
                    elif loc(by[s].tag) == "rest-chord" and loc(by[t].tag) == "rest":
                        rest_child[s] = t
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
                parts = [e for e in system if loc(e.tag) == "part"]
                staffids = {
                    e.get("id") for e in system.iter()
                    if loc(e.tag) == "staff" and e.get("id")
                }
                sys_base = page_measure_base
                bounds = [(float(s.get("left")), float(s.get("right"))) for s in stacks]
                nominal_lengths = [Fraction(s.get("expected") or s.get("duration") or "1") for s in stacks]

                chord_meta = {}
                by_ms = defaultdict(list)

                for cid, node in by.items():
                    typ = loc(node.tag)
                    if typ not in ("head-chord", "rest-chord"):
                        continue
                    if node.get("staff") not in staffids:
                        continue
                    cb = get_box(node)
                    if cb is None:
                        continue
                    mis = [i for i, (l, r) in enumerate(bounds) if l <= cb.cx <= r]
                    if len(mis) != 1:
                        continue
                    mi = mis[0]
                    stem = chord_stem.get(cid)
                    direction = None
                    if typ == "head-chord":
                        hs = chord_heads.get(cid, [])
                        hbs = [get_box(by[h]) for h in hs if h in by and get_box(by[h])]
                        if not hbs:
                            continue
                        allx = [b.cx for b in hbs]
                        sb = get_box(by.get(stem))
                        if sb:
                            direction = "up" if sb.cx >= median(allx) else "down"
                    else:
                        hs = []
                    chord_meta[cid] = {
                        "id": cid,
                        "type": typ,
                        "page": page_index,
                        "system": sy,
                        "measure_local": mi,
                        "measure": sys_base + mi + 1,
                        "staff": node.get("staff") or "",
                        "box": cb,
                        "x": cb.cx,
                        "y": cb.cy,
                        "heads": list(hs),
                        "attack_heads": [h for h in hs if h not in tied_right],
                        "tied_heads": [h for h in hs if h in tied_right],
                        "stem": stem,
                        "direction": direction,
                    }
                    by_ms[(mi, node.get("staff") or "")].append(chord_meta[cid])

                # Visual beam detection supplements semantic beam relations.
                pixel_edges = {}
                for (mi, staff), arr in by_ms.items():
                    arr = sorted(
                        [c for c in arr if c["type"] == "head-chord" and c["stem"] and c["direction"]],
                        key=lambda c: c["x"],
                    )
                    for a, b in zip(arr, arr[1:]):
                        if a["direction"] != b["direction"]:
                            continue
                        sa, sb = get_box(by.get(a["stem"])), get_box(by.get(b["stem"]))
                        if not sa or not sb:
                            continue
                        ta = (sa.cx, sa.y if a["direction"] == "up" else sa.y + sa.h, a["direction"])
                        tb = (sb.cx, sb.y if b["direction"] == "up" else sb.y + sb.h, b["direction"])
                        lv = _pixel_beam_levels(binary, ta, tb, il)
                        if lv:
                            pixel_edges[frozenset((a["id"], b["id"]))] = lv

                levels = {s: len(bs) for s, bs in semantic_beams.items()}
                for pair, lv in pixel_edges.items():
                    for cid in pair:
                        c = chord_meta.get(cid)
                        if c and c["stem"]:
                            levels[c["stem"]] = max(levels.get(c["stem"], 0), lv)

                def duration(cid):
                    node = by[cid]
                    typ = loc(node.tag)
                    if typ == "rest-chord":
                        rid = rest_child.get(cid)
                        rr = by.get(rid)
                        base = _rest_duration(rr.get("shape") if rr is not None else "")
                        if base is None:
                            return None
                        dn = dots.get(rid or "", 0)
                        return base * sum(Fraction(1, 2**i) for i in range(dn + 1))
                    hs = chord_heads.get(cid, [])
                    if not hs:
                        return None
                    if any(head_is_black.get(h, False) for h in hs):
                        base = Fraction(1, 4)
                    else:
                        shapes = [(by[h].get("shape") or "").upper() for h in hs if h in by]
                        base = (
                            Fraction(2) if any("BREVE" in s for s in shapes)
                            else Fraction(1) if any("WHOLE" in s for s in shapes)
                            else Fraction(1, 2)
                        )
                    st = chord_stem.get(cid)
                    lev = max(levels.get(st or "", 0), flags.get(st or "", 0))
                    d0 = base / (2 ** lev)
                    dn = max([dots.get(h, 0) for h in hs] or [0])
                    return d0 * sum(Fraction(1, 2**i) for i in range(dn + 1))

                for c in chord_meta.values():
                    c["duration"] = duration(c["id"])

                # Rebuild each Audiveris voice from object membership + x-order only.
                # No slot/time-offset values are read.
                voices_by_measure = defaultdict(list)
                referenced = set()
                voice_hints_by_measure = defaultdict(list)

                for pi, part in enumerate(parts):
                    measures = [e for e in part if loc(e.tag) == "measure"]
                    for mi, measure in enumerate(measures):
                        if mi >= len(stacks):
                            continue
                        for voice in [e for e in measure if loc(e.tag) == "voice"]:
                            ids = []
                            for ent in [e for e in voice.iter() if loc(e.tag) == "entry"]:
                                val = next((c for c in ent if loc(c.tag) == "value"), None)
                                if val is None or (val.get("status") or "").upper() != "BEGIN":
                                    continue
                                cid = val.get("chord")
                                if cid in chord_meta and chord_meta[cid]["measure_local"] == mi and cid not in ids:
                                    ids.append(cid)
                            ids.sort(key=lambda cid: (chord_meta[cid]["x"], cid))
                            if ids:
                                referenced.update(ids)
                                hint = {
                                    "part": pi,
                                    "voice": voice.get("id") or "1",
                                    "ids": ids,
                                }
                                voices_by_measure[mi].append(hint)
                                voice_hints_by_measure[mi].append({
                                    "part": pi,
                                    "voice": voice.get("id") or "1",
                                    "ids": list(ids),
                                })

                # Recover unreferenced beam-linked chords into a uniquely matching voice.
                beam_adj = defaultdict(set)
                for stem, beams in semantic_beams.items():
                    cids = [cid for cid, c in chord_meta.items() if c["stem"] == stem]
                    for cid in cids:
                        for other, oc in chord_meta.items():
                            if other == cid or oc["measure_local"] != c["measure_local"]:
                                continue
                        # actual adjacency is added below from geometric pixel pairs
                for pair in pixel_edges:
                    a, b = tuple(pair)
                    beam_adj[a].add(b)
                    beam_adj[b].add(a)

                # semantic beam ids connect stems, so convert common beam membership to chord adjacency
                by_beam = defaultdict(list)
                for stem, beams in semantic_beams.items():
                    cids = [cid for cid, c in chord_meta.items() if c["stem"] == stem]
                    for beam in beams:
                        by_beam[beam].extend(cids)
                for cids in by_beam.values():
                    uniq = list(dict.fromkeys(cids))
                    for i, a in enumerate(uniq):
                        for b in uniq[i + 1:]:
                            beam_adj[a].add(b)
                            beam_adj[b].add(a)

                for mi, voices in voices_by_measure.items():
                    changed = True
                    while changed:
                        changed = False
                        for c in [x for x in chord_meta.values() if x["measure_local"] == mi and x["id"] not in referenced and x["type"] == "head-chord"]:
                            candidates = []
                            neighbors = beam_adj.get(c["id"], set())
                            for vi, v in enumerate(voices):
                                if any(n in v["ids"] for n in neighbors):
                                    candidates.append(vi)
                            if len(candidates) == 1:
                                vi = candidates[0]
                                voices[vi]["ids"].append(c["id"])
                                voices[vi]["ids"].sort(key=lambda cid: (chord_meta[cid]["x"], cid))
                                referenced.add(c["id"])
                                changed = True

                for mi, stack in enumerate(stacks):
                    global_m = sys_base + mi + 1
                    nominal = nominal_lengths[mi]
                    attacks = defaultdict(list)
                    voice_rows = []
                    unresolved = []
                    voices = voices_by_measure.get(mi, [])

                    for v in voices:
                        t = Fraction(0)
                        events = []
                        ok = True
                        for cid in v["ids"]:
                            c = chord_meta[cid]
                            d = c["duration"]
                            if d is None:
                                ok = False
                                unresolved.append({"reason": "unknown-duration", "chord": cid})
                                break
                            onset = t
                            events.append({
                                "cid": cid,
                                "type": c["type"],
                                "x": c["x"],
                                "onset": onset,
                                "duration": d,
                                "attack": bool(c["attack_heads"]) if c["type"] == "head-chord" else False,
                            })
                            if c["type"] == "head-chord" and c["attack_heads"]:
                                attacks[onset].append(cid)
                            t += d
                        if t != nominal:
                            ok = False
                            unresolved.append({
                                "reason": "voice-span-mismatch",
                                "part": v["part"],
                                "voice": v["voice"],
                                "span": _fraction_text(t),
                                "nominal": _fraction_text(nominal),
                            })
                        voice_rows.append({
                            "part": v["part"],
                            "voice": v["voice"],
                            "ok": ok,
                            "span": _fraction_text(t),
                            "events": [
                                {
                                    "cid": e["cid"],
                                    "type": e["type"],
                                    "x": round(e["x"], 3),
                                    "onset": _fraction_text(e["onset"]),
                                    "duration": _fraction_text(e["duration"]),
                                    "attack": e["attack"],
                                }
                                for e in events
                            ],
                        })

                    orphan_attacks = [
                        c["id"] for c in chord_meta.values()
                        if c["measure_local"] == mi
                        and c["type"] == "head-chord"
                        and c["attack_heads"]
                        and c["id"] not in referenced
                    ]
                    for cid in orphan_attacks:
                        unresolved.append({"reason": "orphan-attack", "chord": cid})

                    measure_ok = bool(voices) and not unresolved
                    attack_rows = [
                        {
                            "onset_whole_units": _fraction_text(t),
                            "onset_quarter_units": _fraction_text(t * 4),
                            "chords": sorted(cids),
                        }
                        for t, cids in sorted(attacks.items())
                    ]

                    event_rows = []
                    for ec in sorted(
                        [q for q in chord_meta.values() if q["measure_local"] == mi],
                        key=lambda q: (q["x"], q["staff"], q["id"])
                    ):
                        hs = ec.get("heads", [])
                        fill_ratios = []
                        head_shapes = []
                        dot_count = 0
                        for hid in hs:
                            hb = get_box(by.get(hid))
                            if hb is not None:
                                fill_ratios.append(round(_head_fill_ratio(binary, hb), 4))
                            hn = by.get(hid)
                            if hn is not None:
                                head_shapes.append(hn.get("shape") or "")
                            dot_count = max(dot_count, dots.get(hid, 0))
                        rest_shape = None
                        if ec["type"] == "rest-chord":
                            rr = by.get(rest_child.get(ec["id"]))
                            rest_shape = rr.get("shape") if rr is not None else None
                            dot_count = max(dot_count, dots.get(rest_child.get(ec["id"]) or "", 0))
                        st = ec.get("stem")
                        event_rows.append({
                            "id": ec["id"],
                            "kind": "note" if ec["type"] == "head-chord" else "rest",
                            "staff": ec["staff"],
                            "x": round(float(ec["x"]), 3),
                            "y": round(float(ec["y"]), 3),
                            "direction": ec.get("direction"),
                            "duration_whole_units": None if ec.get("duration") is None else _fraction_text(ec["duration"]),
                            "attack": bool(ec.get("attack_heads")),
                            "head_count": len(hs),
                            "attack_head_count": len(ec.get("attack_heads", [])),
                            "tied_head_count": len(ec.get("tied_heads", [])),
                            "head_shapes": head_shapes,
                            "head_fill_ratios": fill_ratios,
                            "stem_present": bool(st),
                            "semantic_beam_level": len(semantic_beams.get(st or "", set())),
                            "resolved_beam_level": levels.get(st or "", 0),
                            "flag_level": flags.get(st or "", 0),
                            "dot_count": dot_count,
                            "rest_shape": rest_shape,
                        })

                    measure_ids = {e["id"] for e in event_rows}
                    beam_edges = []
                    seen_beam_edges = set()
                    for a in sorted(measure_ids):
                        for b in sorted(beam_adj.get(a, set())):
                            if b not in measure_ids or a == b:
                                continue
                            k = tuple(sorted((a, b)))
                            if k in seen_beam_edges:
                                continue
                            seen_beam_edges.add(k)
                            beam_edges.append(list(k))

                    measures_out[global_m] = {
                        "measure": global_m,
                        "page": page_index + 1,
                        "system": sy + 1,
                        "nominal_whole_units": _fraction_text(nominal),
                        "bar_left_x": bounds[mi][0],
                        "bar_right_x": bounds[mi][1],
                        "interline": il,
                        "ok": measure_ok,
                        "attack_count": len(attack_rows) if measure_ok else None,
                        "attacks": attack_rows if measure_ok else [],
                        "voices": voice_rows,
                        "voice_hints": voice_hints_by_measure.get(mi, []),
                        "events": event_rows,
                        "beam_edges": beam_edges,
                        "unresolved": unresolved,
                    }

                page_measure_base += len(stacks)

            global_measure_base = page_measure_base

    return measures_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("omr")
    ap.add_argument("--json", default="visual_attack_fullscore.json")
    args = ap.parse_args()

    data = extract_visual_attacks(args.omr)
    Path(args.json).write_text(json.dumps(data, indent=2))
    ok = [m for m, r in data.items() if r["ok"]]
    bad = [m for m, r in data.items() if not r["ok"]]
    print(f"VISUAL_FULLSCORE measures={len(data)} ok={len(ok)} unresolved={len(bad)}")
    print("VISUAL_FULLSCORE_OK", ok)
    print("VISUAL_FULLSCORE_UNRESOLVED", bad)
    for m in (7, 28, 34, 39):
        r = data.get(m)
        print(f"TARGET m{m}: {json.dumps(r, separators=(',', ':')) if r else 'MISSING'}")
