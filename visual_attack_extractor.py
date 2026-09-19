from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from lxml import etree

sys.path.insert(0, "/app")
from core import get_box, loc, _interline, _system_bounds, _head_fill_ratio, _pixel_beam_levels, _rest_duration

OUT = Path("/work/visual_attacks")
OUT.mkdir(parents=True, exist_ok=True)

# Visual simultaneity tolerance.  The Buxtehude scan's main interline is 21 px;
# this corresponds to about 5 px and is deliberately much tighter than a note spacing.
ALIGN_IL = 0.50


def sheet_number(name: str) -> int:
    m = re.search(r"sheet#(\d+)", name, re.I)
    return int(m.group(1)) if m else 10**9


def qstr(v: Fraction | None) -> str | None:
    return None if v is None else str(v * 4)


def fstr(v: Fraction | None) -> str | None:
    return None if v is None else str(v)


def median_x(vals):
    vals = list(vals)
    return float(median(vals)) if vals else None


def chord_duration(by, chord_heads, chord_stem, rest_child, flags, dots, head_is_black, cid, levels):
    node = by[cid]
    typ = loc(node.tag)
    if typ == "rest-chord":
        rr = by.get(rest_child.get(cid))
        return _rest_duration(rr.get("shape") if rr is not None else "")
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


def dedupe_x_onsets(points, tol):
    """Return visual x anchors as median x for each exact onset."""
    by_onset = defaultdict(list)
    for x, onset in points:
        if x is not None and onset is not None:
            by_onset[onset].append(float(x))
    return {o: float(median(xs)) for o, xs in by_onset.items()}


def inferred_start_from_anchors(local_events, anchors, expected, il):
    """
    Infer one voice's start offset from visual coincidence only.

    local_events: [(tau, x), ...], where tau is cumulative duration from the
    first event and x is the graphical event position.
    anchors: exact-onset -> median visual x from already resolved voices.

    Every close visual coincidence votes for start = anchor_onset - tau.
    A start is accepted only if all close coincidences agree exactly.
    """
    votes = []
    tol = ALIGN_IL * il
    for tau, x in local_events:
        if x is None:
            continue
        for onset, ax in anchors.items():
            if abs(x - ax) <= tol:
                s = onset - tau
                if s >= 0:
                    votes.append(s)
    if not votes:
        return None
    counts = defaultdict(int)
    for s in votes:
        counts[s] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    best, n = ranked[0]
    # Do not accept contradictory equally supported offsets.
    if len(ranked) > 1 and ranked[1][1] == n and ranked[1][0] != best:
        return None
    return best


def repair_voice_span_to_meter(vr, chords, meter):
    """
    Repair exactly one visually implausible binary duration when a voice is
    visually anchored at onset 0 but its durations miss the meter by one
    power-of-two step.

    This targets the common OMR failure where an isolated flag is missed
    (duration too long) or a spurious beam/flag is added (duration too short).
    Selection is based only on graphical spacing to the next event.
    """
    delta = meter - vr["span"]
    if delta == 0 or not vr["events"]:
        return None

    candidates = []
    evs = vr["events"]
    for i, (tau, x, cid) in enumerate(evs):
        ch = chords[cid]
        d = ch.get("duration")
        if d is None or ch.get("kind") != "note":
            continue
        next_x = evs[i+1][1] if i+1 < len(evs) else None
        if next_x is None:
            continue
        gap = max(0.0, float(next_x) - float(x))
        dq = float(d * 4)
        if dq <= 0:
            continue
        spacing_ratio = gap / dq

        # Voice is too long: one duration may need one extra flag => halve it.
        if delta < 0 and d / 2 == -delta:
            candidates.append(("halve", i, cid, d / 2, spacing_ratio))
        # Voice is too short: one duration may have one spurious flag/beam => double it.
        if delta > 0 and d == delta:
            candidates.append(("double", i, cid, d * 2, spacing_ratio))

    if not candidates:
        return None

    if delta < 0:
        ranked = sorted(candidates, key=lambda z: (z[4], z[1]))
        best = ranked[0]
        if len(ranked) > 1 and not (ranked[1][4] > best[4] * 1.12):
            return None
    else:
        ranked = sorted(candidates, key=lambda z: (-z[4], z[1]))
        best = ranked[0]
        if len(ranked) > 1 and not (best[4] > ranked[1][4] * 1.12):
            return None

    action, _, cid, new_d, ratio = best
    old_d = chords[cid]["duration"]
    chords[cid]["duration"] = new_d
    vr["span"] += new_d - old_d

    # Rebuild cumulative local times after changing the visual duration.
    local = Fraction(0)
    rebuilt = []
    for _, x, qid in vr["events"]:
        rebuilt.append((local, x, qid))
        local += chords[qid]["duration"]
    vr["events"] = rebuilt
    vr["span"] = local
    return {
        "action": action,
        "chord_id": cid,
        "old_duration": str(old_d),
        "new_duration": str(new_d),
        "spacing_ratio": ratio,
    }


def cluster_hit_chords(chords, il):
    """Merge note attacks that have the same reconstructed onset."""
    out = defaultdict(list)
    for c in chords:
        if c.get("onset") is not None and c.get("attack_heads"):
            out[c["onset"]].append(c)
    return out


def choose_rep_head(chords, by):
    heads = []
    for c in chords:
        for hid in c.get("attack_heads", []):
            hb = get_box(by.get(hid))
            if hb is not None:
                heads.append((hid, hb, c))
    if not heads:
        return None
    tx = median([h[1].cx for h in heads])
    return min(heads, key=lambda h: (abs(h[1].cx - tx), h[1].cx, h[1].cy))


def main(omr_path: str):
    score = {
        "engine": "visual-attack-extractor-v1",
        "timing_source": "visual durations + sequence order + meter + visual alignment; Audiveris slot onsets ignored",
        "measures": {},
        "totals": {},
    }
    global_measure_base = 0
    page_images = {}

    with ZipFile(omr_path) as archive:
        members = sorted(
            [n for n in archive.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=lambda n: (sheet_number(n), n.lower()),
        )
        if not members:
            raise ValueError("No sheet XML in OMR")

        for page_index, member in enumerate(members):
            root = etree.fromstring(archive.read(member))
            by = {e.get("id"): e for e in root.iter() if e.get("id")}
            il = _interline(root)
            pic = next((e for e in root.iter() if loc(e.tag) == "picture"), None)
            if pic is None:
                raise ValueError("Missing picture geometry")
            W, H = float(pic.get("width")), float(pic.get("height"))
            binary_name = member.rsplit("/", 1)[0] + "/BINARY.png"
            binary_img = Image.open(BytesIO(archive.read(binary_name))).convert("L")
            binary = np.array(binary_img) < 128
            page_images[page_index] = (binary, W, H)

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
                comp, stack = [], [sid]
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
                staffids = {e.get("id") for e in system.iter() if loc(e.tag) == "staff" and e.get("id")}
                sys_base = page_measure_base
                bounds = [(float(s.get("left")), float(s.get("right"))) for s in stacks]
                expected = [Fraction(s.get("expected") or "1") for s in stacks]

                # Graphical chord/rest objects in this system.
                chords = {}
                by_ms = defaultdict(list)
                for cid, node in by.items():
                    typ = loc(node.tag)
                    if typ not in ("head-chord", "rest-chord") or node.get("staff") not in staffids:
                        continue
                    cb = get_box(node)
                    if cb is None:
                        # Some rest chords have no own bounds; use contained rest.
                        if typ == "rest-chord":
                            cb = get_box(by.get(rest_child.get(cid)))
                    if cb is None:
                        continue
                    mis = [i for i, (l, r) in enumerate(bounds) if l <= cb.cx <= r]
                    if len(mis) != 1:
                        continue
                    mi = mis[0]
                    hs = chord_heads.get(cid, []) if typ == "head-chord" else []
                    hbs = [get_box(by[h]) for h in hs if h in by and get_box(by[h])]
                    attack = [h for h in hs if h not in tied_right]
                    ax = [get_box(by[h]).cx for h in attack if h in by and get_box(by[h])]
                    allx = [b.cx for b in hbs]
                    stem = chord_stem.get(cid)
                    direction = None
                    if typ == "head-chord" and allx:
                        sb = get_box(by.get(stem))
                        if sb:
                            direction = "up" if sb.cx >= median(allx) else "down"
                    x = float(median(ax or allx)) if (ax or allx) else float(cb.cx)
                    rec = {
                        "id": cid,
                        "kind": "note" if typ == "head-chord" else "rest",
                        "measure": sys_base + mi + 1,
                        "mi": mi,
                        "staff": node.get("staff") or "",
                        "box": cb,
                        "heads": list(hs),
                        "attack_heads": list(attack),
                        "stem": stem,
                        "direction": direction,
                        "x": x,
                        "duration": None,
                        "referenced": False,
                        "voice_key": None,
                        "onset": None,
                        "status": "unresolved",
                    }
                    chords[cid] = rec
                    by_ms[(mi, rec["staff"])].append(rec)

                # Image-derived beam completion; no timing data.
                pixel_edges = {}
                for (mi, staff), arr in by_ms.items():
                    arr = sorted([c for c in arr if c["kind"] == "note" and c["stem"] and c["direction"]], key=lambda c: c["x"])
                    for a, b in zip(arr, arr[1:]):
                        if a["direction"] != b["direction"]:
                            continue
                        sa, sb = get_box(by.get(a["stem"])), get_box(by.get(b["stem"]))
                        if sa and sb:
                            ta = (sa.cx, sa.y if a["direction"] == "up" else sa.y + sa.h, a["direction"])
                            tb = (sb.cx, sb.y if b["direction"] == "up" else sb.y + sb.h, b["direction"])
                            lv = _pixel_beam_levels(binary, ta, tb, il)
                            if lv:
                                pixel_edges[frozenset((a["id"], b["id"]))] = lv

                levels = {s: len(bs) for s, bs in semantic_beams.items()}
                for pair, lv in pixel_edges.items():
                    for cid in pair:
                        c = chords.get(cid)
                        if c and c["stem"]:
                            levels[c["stem"]] = max(levels.get(c["stem"], 0), lv)

                for cid, c in chords.items():
                    c["duration"] = chord_duration(
                        by, chord_heads, chord_stem, rest_child, flags, dots,
                        head_is_black, cid, levels
                    )

                # Build voice membership/order using entry relationships only.
                # Crucially, the slot time-offset is never read.
                measure_voice_sequences = defaultdict(list)
                for mi, stack_node in enumerate(stacks):
                    slot_x = {
                        s.get("id"): float(s.get("x-offset") or 0) + bounds[mi][0]
                        for s in stack_node if loc(s.tag) == "slot"
                    }
                    for pi, part in enumerate(parts):
                        measures = [e for e in part if loc(e.tag) == "measure"]
                        if mi >= len(measures):
                            continue
                        for voice in [e for e in measures[mi] if loc(e.tag) == "voice"]:
                            seq = []
                            seen_ids = set()
                            for ent in [e for e in voice.iter() if loc(e.tag) == "entry"]:
                                key = next((c for c in ent if loc(c.tag) == "key"), None)
                                val = next((c for c in ent if loc(c.tag) == "value"), None)
                                if key is None or val is None or (val.get("status") or "").upper() != "BEGIN":
                                    continue
                                cid = val.get("chord")
                                sid = (key.text or "").strip()
                                if cid in seen_ids or cid not in chords:
                                    continue
                                seen_ids.add(cid)
                                x = chords[cid]["x"]
                                seq.append((x, cid))
                            seq.sort(key=lambda q: (q[0], chords[q[1]]["x"], q[1]))
                            if seq:
                                vk = f"p{pi+1}:v{voice.get('id') or '1'}"
                                measure_voice_sequences[mi].append((vk, seq))
                                for _, cid in seq:
                                    chords[cid]["referenced"] = True
                                    chords[cid]["voice_key"] = vk

                # Resolve each measure independently.
                for mi, (left, right) in enumerate(bounds):
                    gm = sys_base + mi + 1
                    meter = expected[mi]  # whole-note units
                    seqs = measure_voice_sequences.get(mi, [])
                    voice_records = []
                    unresolved_reasons = []

                    for vk, seq in seqs:
                        local = Fraction(0)
                        events = []
                        valid = True
                        for x, cid in seq:
                            c = chords[cid]
                            d = c["duration"]
                            if d is None:
                                valid = False
                                unresolved_reasons.append(f"{vk}:{cid}:unknown-duration")
                                break
                            events.append((local, float(x), cid))
                            local += d
                        voice_records.append({
                            "key": vk,
                            "events": events,
                            "span": local,
                            "valid": valid,
                            "start": None,
                            "status": "unresolved",
                        })

                    anchors_points = []

                    # Pass 1: voices whose visual duration exactly fills the bar.
                    for vr in voice_records:
                        if not vr["valid"]:
                            continue
                        if vr["span"] == meter:
                            vr["start"] = Fraction(0)
                            vr["status"] = "meter-filled"
                            for tau, x, cid in vr["events"]:
                                chords[cid]["onset"] = tau
                                chords[cid]["status"] = "resolved"
                                anchors_points.append((x, tau))
                        elif vr["span"] > meter:
                            vr["status"] = "overfull"
                            unresolved_reasons.append(f"{vr['key']}:overfull:{vr['span']}>{meter}")

                    # Pass 1b: if a non-fitting voice begins on the same visual
                    # column as a securely resolved onset-0 event, require start=0 and
                    # try one visually supported binary duration repair.
                    anchors0 = [x for x, onset in anchors_points if onset == 0]
                    for vr in voice_records:
                        if not vr["valid"] or vr["start"] is not None or not vr["events"]:
                            continue
                        first_x = vr["events"][0][1]
                        if not anchors0 or min(abs(first_x - ax) for ax in anchors0) > ALIGN_IL * il:
                            continue
                        repair = repair_voice_span_to_meter(vr, chords, meter)
                        if repair is None or vr["span"] != meter:
                            continue
                        vr["start"] = Fraction(0)
                        vr["status"] = "meter-repaired-from-visual-spacing"
                        vr["repair"] = repair
                        for tau, x, cid in vr["events"]:
                            chords[cid]["onset"] = tau
                            chords[cid]["status"] = "resolved-meter-repaired"
                            anchors_points.append((x, tau))

                    # Pass 2: use exact visual coincidence with resolved voices.
                    changed = True
                    while changed:
                        changed = False
                        anchors = dedupe_x_onsets(anchors_points, ALIGN_IL * il)
                        for vr in voice_records:
                            if not vr["valid"] or vr["start"] is not None or vr["span"] > meter:
                                continue
                            local_events = [(tau, x) for tau, x, _ in vr["events"]]
                            s = inferred_start_from_anchors(local_events, anchors, meter, il)
                            if s is None:
                                continue
                            if s + vr["span"] > meter:
                                continue
                            vr["start"] = s
                            vr["status"] = "visual-aligned"
                            for tau, x, cid in vr["events"]:
                                onset = s + tau
                                chords[cid]["onset"] = onset
                                chords[cid]["status"] = "resolved"
                                anchors_points.append((x, onset))
                            changed = True

                    # Pass 3: edge-constrained short voices.  These are accepted only
                    # when their first/last graphical event is visibly against a bar edge.
                    edge_tol = 1.5 * il
                    for vr in voice_records:
                        if not vr["valid"] or vr["start"] is not None or vr["span"] > meter or not vr["events"]:
                            continue
                        first_x = vr["events"][0][1]
                        last_tau, last_x, last_cid = vr["events"][-1]
                        last_end = last_tau + chords[last_cid]["duration"]
                        candidate = None
                        status = None
                        if first_x - left <= edge_tol:
                            candidate, status = Fraction(0), "left-edge"
                        elif right - last_x <= edge_tol:
                            candidate, status = meter - vr["span"], "right-edge"
                        if candidate is not None and candidate >= 0 and candidate + vr["span"] <= meter:
                            vr["start"] = candidate
                            vr["status"] = status
                            for tau, x, cid in vr["events"]:
                                onset = candidate + tau
                                chords[cid]["onset"] = onset
                                chords[cid]["status"] = "resolved"
                                anchors_points.append((x, onset))

                    # Unreferenced graphical notes are accepted only when they visually
                    # coincide with an already resolved exact onset.
                    anchors = dedupe_x_onsets(anchors_points, ALIGN_IL * il)
                    unreferenced = [c for c in chords.values() if c["measure"] == gm and not c["referenced"]]
                    for c in unreferenced:
                        close = [(abs(c["x"] - ax), onset) for onset, ax in anchors.items() if abs(c["x"] - ax) <= ALIGN_IL * il]
                        if close:
                            close.sort()
                            if len(close) == 1 or close[0][1] == close[1][1]:
                                c["onset"] = close[0][1]
                                c["status"] = "visual-coincident-unreferenced"
                                anchors_points.append((c["x"], c["onset"]))

                    # Final visual-only rescue: any still-unresolved attack whose
                    # notehead column uniquely coincides with an already resolved onset
                    # inherits that onset.  This handles parallel/duplicate voice layers
                    # without trusting their semantic timing.
                    anchors = dedupe_x_onsets(anchors_points, ALIGN_IL * il)
                    for c in [q for q in chords.values() if q["measure"] == gm and q["kind"] == "note" and q["attack_heads"] and q["onset"] is None]:
                        close = sorted(
                            (abs(c["x"] - ax), onset)
                            for onset, ax in anchors.items()
                            if abs(c["x"] - ax) <= ALIGN_IL * il
                        )
                        if close and (len(close) == 1 or close[0][0] + 0.10*il < close[1][0] or close[0][1] == close[1][1]):
                            c["onset"] = close[0][1]
                            c["status"] = "visual-coincident-rescue"
                            anchors_points.append((c["x"], c["onset"]))

                    unresolved_voices = [vr["key"] for vr in voice_records if vr["valid"] and vr["start"] is None]
                    unresolved_notes = [
                        c["id"] for c in chords.values()
                        if c["measure"] == gm and c["kind"] == "note" and c["attack_heads"] and c["onset"] is None
                    ]
                    # Recompute final diagnostics after all visual repairs/rescues.
                    # Earlier provisional overfull states must not survive after a
                    # successful meter repair.
                    final_reasons = [r for r in unresolved_reasons if ":unknown-duration" in r]
                    for vr in voice_records:
                        if not vr["valid"]:
                            continue
                        if vr["start"] is None:
                            if vr["span"] > meter:
                                final_reasons.append(f"{vr['key']}:overfull:{vr['span']}>{meter}")
                            else:
                                final_reasons.append(f"{vr['key']}:unresolved")
                    if unresolved_notes:
                        final_reasons.append("unresolved-attacks:" + ",".join(unresolved_notes))
                    unresolved_reasons = final_reasons

                    measure_chords = [c for c in chords.values() if c["measure"] == gm]
                    hits = cluster_hit_chords(measure_chords, il)
                    hit_rows = []
                    for onset in sorted(hits):
                        cs = hits[onset]
                        rep = choose_rep_head(cs, by)
                        hit_rows.append({
                            "onset_whole_units": fstr(onset),
                            "onset_quarter_units": str(onset * 4),
                            "x": None if rep is None else round(rep[1].cx, 3),
                            "chord_ids": [c["id"] for c in cs],
                        })

                    # For TMA, unresolved voice bookkeeping is acceptable when every
                    # visible attack has nevertheless received a unique visual onset.
                    # Overfull/unknown-duration reasons remain fatal unless repaired.
                    fatal_reasons = [
                        r for r in unresolved_reasons
                        if not r.startswith("unresolved-voices:") and not r.startswith("unresolved-attacks:")
                    ]
                    complete = (not unresolved_notes) and (not fatal_reasons)
                    score["measures"][str(gm)] = {
                        "page": page_index + 1,
                        "system": sy + 1,
                        "meter_whole_units": str(meter),
                        "bar_bounds_x": [left, right],
                        "system_bounds_y": [top, bottom],
                        "interline": il,
                        "complete": complete,
                        "unresolved_reasons": unresolved_reasons,
                        "voice_records": [
                            {
                                "key": vr["key"],
                                "span_whole_units": str(vr["span"]),
                                "start_whole_units": None if vr["start"] is None else str(vr["start"]),
                                "status": vr["status"],
                                "repair": vr.get("repair"),
                            }
                            for vr in voice_records
                        ],
                        "hits": hit_rows,
                        "chords": [
                            {
                                "id": c["id"],
                                "kind": c["kind"],
                                "staff": c["staff"],
                                "x": round(c["x"], 3),
                                "duration_whole_units": fstr(c["duration"]),
                                "duration_quarter_units": qstr(c["duration"]),
                                "onset_whole_units": fstr(c["onset"]),
                                "onset_quarter_units": qstr(c["onset"]),
                                "status": c["status"],
                                "referenced": c["referenced"],
                                "direction": c["direction"],
                                "heads": [
                                    {
                                        "id": hid,
                                        "x": round(get_box(by[hid]).cx, 3) if get_box(by.get(hid)) else None,
                                        "y": round(get_box(by[hid]).cy, 3) if get_box(by.get(hid)) else None,
                                        "shape": (by[hid].get("shape") or "") if hid in by else "",
                                        "pitch": by[hid].get("pitch") if hid in by else None,
                                        "tied_continuation": hid in tied_right,
                                        "tie_start": hid in tie_starts,
                                        "attack": hid not in tied_right,
                                    }
                                    for hid in c["heads"]
                                ],
                            }
                            for c in sorted(measure_chords, key=lambda z:(z["x"], z["staff"], z["id"]))
                        ],
                    }

                    # Per-measure colored audit.
                    x0 = max(0, int(round(left - 1.0 * il)))
                    x1 = min(int(W), int(round(right + 1.0 * il)))
                    y0 = max(0, int(round(top - 2.5 * il)))
                    y1 = min(int(H), int(round(bottom + 2.5 * il)))
                    crop_gray = np.where(binary[y0:y1, x0:x1], 0, 255).astype(np.uint8)
                    crop = Image.fromarray(crop_gray, mode="L").convert("RGB")
                    header_h = 108
                    canvas = Image.new("RGB", (crop.width, crop.height + header_h), "white")
                    canvas.paste(crop, (0, header_h))
                    draw = ImageDraw.Draw(canvas)
                    status_text = "COMPLETE" if complete else "UNRESOLVED"
                    draw.text((12, 8), f"Measure {gm} — visual attack reconstruction — {status_text}", fill=(0,0,0))
                    draw.text((12, 32), "green=counted attack  blue=same chord  yellow=tied continuation  red=unresolved", fill=(0,0,0))
                    onsets = ", ".join(r["onset_quarter_units"] for r in hit_rows)
                    draw.text((12, 56), "TMA onsets (quarter units): " + (onsets or "none"), fill=(0,0,0))
                    if unresolved_reasons:
                        draw.text((12, 80), ("; ".join(unresolved_reasons))[:150], fill=(180,0,0))
                    else:
                        draw.text((12, 80), f"{len(hit_rows)} attacks; all reconstructed without Audiveris slot timing", fill=(0,110,0))

                    radius = max(6, int(round(0.34 * il)))
                    width = max(3, int(round(0.12 * il)))
                    for c in measure_chords:
                        if c["kind"] != "note":
                            continue
                        hdata = [(hid, get_box(by.get(hid))) for hid in c["heads"]]
                        hdata = [(hid, hb) for hid, hb in hdata if hb is not None]
                        attacks = [(hid, hb) for hid, hb in hdata if hid not in tied_right]
                        tied = [(hid, hb) for hid, hb in hdata if hid in tied_right]
                        if c["onset"] is None:
                            for hid, hb in hdata:
                                cx, cy = hb.cx-x0, hb.cy-y0+header_h
                                draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), outline=(220,0,0), width=width)
                            continue
                        if attacks:
                            rep_i = min(range(len(attacks)), key=lambda i: abs(attacks[i][1].cy - median([b.cy for _,b in attacks])))
                            for i,(hid,hb) in enumerate(attacks):
                                cx, cy = hb.cx-x0, hb.cy-y0+header_h
                                color = (0,170,0) if i == rep_i else (0,105,230)
                                draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), outline=color, width=width)
                        for hid,hb in tied:
                            cx, cy = hb.cx-x0, hb.cy-y0+header_h
                            draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), outline=(240,175,0), width=width)

                    canvas.save(OUT / f"measure_{gm:03d}_visual_attacks.png")

                page_measure_base += len(stacks)
            global_measure_base = page_measure_base

    measures = score["measures"]
    complete = [int(m) for m,v in measures.items() if v["complete"]]
    unresolved = [int(m) for m,v in measures.items() if not v["complete"]]
    total_hits_complete = sum(len(v["hits"]) for v in measures.values() if v["complete"])
    score["totals"] = {
        "measure_count": len(measures),
        "complete_measure_count": len(complete),
        "unresolved_measure_count": len(unresolved),
        "complete_measures": complete,
        "unresolved_measures": unresolved,
        "total_hits_in_complete_measures": total_hits_complete,
    }
    (OUT / "visual_attack_diagnostics.json").write_text(json.dumps(score, indent=2))

    print("VISUAL_FULL_SCORE_SUMMARY=" + json.dumps(score["totals"], separators=(",",":")))
    for m in sorted(measures, key=lambda x:int(x)):
        v = measures[m]
        print(
            f"VISUAL_ATTACKS m{m}: complete={v['complete']} count={len(v['hits'])} "
            f"onsets={[h['onset_quarter_units'] for h in v['hits']]} "
            f"reasons={v['unresolved_reasons']}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: visual_attack_extractor.py SCORE.omr")
    main(sys.argv[1])
