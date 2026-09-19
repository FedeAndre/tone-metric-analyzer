from __future__ import annotations

import json
import math
import sys
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path
from statistics import median

from z3 import (
    And, Bool, If, Implies, Int, IntVal, Or, Optimize, Sum, sat
)

TICKS_PER_WHOLE = 128


def F(v) -> Fraction:
    return Fraction(str(v))


def to_ticks(v: Fraction) -> int:
    q = v * TICKS_PER_WHOLE
    if q.denominator != 1:
        raise ValueError(f"duration/onset {v} is not representable on {TICKS_PER_WHOLE} ticks/whole")
    return int(q)


def tick_text(t: int) -> str:
    f = Fraction(t, TICKS_PER_WHOLE)
    return str(f)


def quarter_text(t: int) -> str:
    return str(Fraction(t, TICKS_PER_WHOLE) * 4)


def _pow2_distance(a: int, b: int) -> int:
    if a <= 0 or b <= 0:
        return 99
    return int(round(abs(math.log2(a / b)) * 4))


def _dot_apply(base_ticks: int, dots: int) -> int | None:
    f = Fraction(base_ticks, 1) * sum(Fraction(1, 2**i) for i in range(dots + 1))
    return int(f) if f.denominator == 1 else None


def visual_duration_guess(event: dict) -> tuple[int | None, int]:
    """Return (ticks, confidence 0..3) from note pixels/symbol relations only."""
    primary = event.get("duration_whole_units")
    primary_ticks = to_ticks(F(primary)) if primary else None
    if event.get("kind") == "rest":
        return primary_ticks, 3 if primary_ticks else 0

    shapes = [str(x or "").upper() for x in event.get("head_shapes", [])]
    fills = [float(x) for x in event.get("head_fill_ratios", []) if x is not None]
    fill = median(fills) if fills else None
    stem = bool(event.get("stem_present"))
    sem_beam = int(event.get("semantic_beam_level") or 0)
    resolved_beam = int(event.get("resolved_beam_level") or 0)
    flag = int(event.get("flag_level") or 0)
    level = max(sem_beam, resolved_beam, flag)
    dots = int(event.get("dot_count") or 0)

    if any("BREVE" in s for s in shapes):
        base = 256
        conf = 3
    elif any("WHOLE" in s for s in shapes):
        base = 128
        conf = 3
    else:
        semantic_black = any("BLACK" in s for s in shapes)
        semantic_void = any("VOID" in s for s in shapes)

        if fill is not None and fill >= 0.82:
            black = True
            fill_conf = 3
        elif fill is not None and fill <= 0.62:
            black = False
            fill_conf = 3
        elif semantic_black and not semantic_void:
            black = True
            fill_conf = 2
        elif semantic_void and not semantic_black:
            black = False
            fill_conf = 2
        elif fill is not None:
            black = fill >= 0.72
            fill_conf = 1
        else:
            return primary_ticks, 0

        if black:
            base = 32
            if level:
                base = max(1, base // (2**level))
            conf = min(3, fill_conf + (1 if level == max(sem_beam, flag) and level > 0 else 0))
        else:
            base = 64 if stem else 128
            conf = fill_conf

    val = _dot_apply(base, dots)
    if val is None:
        return primary_ticks, max(0, conf - 1)
    return val, conf


def duration_candidates(event: dict, meter_ticks: int) -> list[tuple[int, int, str]]:
    """
    General visual duration hypotheses.
    Costs are evidence costs only; no measure-specific values are used.
    """
    ptxt = event.get("duration_whole_units")
    if not ptxt:
        return []
    primary = to_ticks(F(ptxt))
    visual, vconf = visual_duration_guess(event)

    vals: dict[int, tuple[int, str]] = {}

    def put(v: int | None, cost: int, why: str):
        if v is None or v <= 0 or v > meter_ticks:
            return
        old = vals.get(v)
        if old is None or cost < old[0]:
            vals[v] = (cost, why)

    # Primary recognition remains evidence, not authority.
    put(primary, 0 if visual == primary else (2 if vconf <= 1 else 4), "primary")

    if visual is not None:
        put(visual, max(0, 3 - vconf), "visual")
        # One missed/spurious beam or flag.
        if visual % 2 == 0:
            put(visual // 2, 6 + max(0, 2 - vconf), "visual+beam")
        put(visual * 2, 6 + max(0, 2 - vconf), "visual-beam")

    # Generic local alternatives around the primary interpretation.
    if primary % 2 == 0:
        put(primary // 2, 7, "primary+beam")
    put(primary * 2, 7, "primary-beam")

    # Dotted/undotted alternatives.
    if primary % 2 == 0:
        put(primary * 3 // 2, 8, "primary+dotted")
    if primary % 3 == 0:
        put(primary * 2 // 3, 8, "primary-undotted")

    # Rest shapes are much less ambiguous than note beaming. Keep alternatives expensive.
    if event.get("kind") == "rest":
        vals = {
            v: ((0 if v == primary else max(12, c)), why)
            for v, (c, why) in vals.items()
        }

    out = [(v, c, why) for v, (c, why) in vals.items()]
    out.sort(key=lambda z: (z[1], z[0]))
    return out


def connected_components(edges: list[tuple[str, str]], valid: set[str]) -> list[list[str]]:
    adj = defaultdict(set)
    for a, b in edges:
        if a in valid and b in valid:
            adj[a].add(b)
            adj[b].add(a)
    out = []
    seen = set()
    for s in adj:
        if s in seen:
            continue
        q = [s]
        seen.add(s)
        comp = []
        while q:
            x = q.pop()
            comp.append(x)
            for n in adj[x]:
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        out.append(comp)
    return out


def solve_measure(row: dict, ambiguity_check: bool = True) -> dict:
    events = [dict(e) for e in row.get("events", [])]
    meter_ticks = to_ticks(F(row["nominal_whole_units"]))
    il = float(row.get("interline") or 20.0)

    if not events:
        return {
            "measure": row["measure"], "ok": False, "reason": "no-events",
            "attacks": [], "attack_count": None
        }

    by_id = {e["id"]: e for e in events}
    ids = list(by_id)

    # Duration hypotheses.
    cand = {}
    for e in events:
        cs = duration_candidates(e, meter_ticks)
        if not cs:
            return {
                "measure": row["measure"], "ok": False,
                "reason": f"no-duration-candidates:{e['id']}",
                "attacks": [], "attack_count": None
            }
        cand[e["id"]] = cs

    # Soft OMR voice-adjacency hints. Timing values themselves are never read.
    hint_edges = set()
    for vh in row.get("voice_hints", []):
        seq = [x for x in vh.get("ids", []) if x in by_id]
        for a, b in zip(seq, seq[1:]):
            if a != b:
                hint_edges.add((a, b))

    # Hard beam chains from graphical beam connectivity only.
    beam_edges = [tuple(x) for x in row.get("beam_edges", []) if len(x) == 2]
    hard_edges = set()
    for comp in connected_components(beam_edges, set(ids)):
        comp = sorted(comp, key=lambda z: (float(by_id[z]["x"]), z))
        for a, b in zip(comp, comp[1:]):
            if float(by_id[b]["x"]) > float(by_id[a]["x"]) + 0.10 * il:
                hard_edges.add((a, b))

    opt = Optimize()
    opt.set(timeout=8000)

    t = {i: Int(f"t_{row['measure']}_{i}") for i in ids}
    d = {i: Int(f"d_{row['measure']}_{i}") for i in ids}
    start = {i: Bool(f"start_{row['measure']}_{i}") for i in ids}
    end = {i: Bool(f"end_{row['measure']}_{i}") for i in ids}

    choose = {}
    penalty_terms = []

    for i in ids:
        e = by_id[i]
        opt.add(t[i] >= 0, t[i] < meter_ticks)
        ch = []
        for k, (ticks, cost, why) in enumerate(cand[i]):
            b = Bool(f"dur_{row['measure']}_{i}_{k}")
            choose[(i, k)] = b
            ch.append(b)
            penalty_terms.append(If(b, IntVal(cost * 5), IntVal(0)))
        opt.add(Sum([If(x, 1, 0) for x in ch]) == 1)
        opt.add(d[i] == Sum([If(choose[(i, k)], ticks, 0) for k in range(len(cand[i]))]))
        opt.add(d[i] > 0, t[i] + d[i] <= meter_ticks)

    # Candidate path edges. Every graphical event participates exactly once.
    edge = {}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)

    hard_edge_set = set(hard_edges)
    hint_edge_set = set(hint_edges)

    def can_follow(a: str, b: str) -> bool:
        ea, eb = by_id[a], by_id[b]
        xa, xb = float(ea["x"]), float(eb["x"])
        if (a, b) in hard_edge_set or (a, b) in hint_edge_set:
            return xb > xa - 0.05 * il
        if ea.get("staff") != eb.get("staff"):
            return False
        return xb > xa + 0.18 * il

    for a in ids:
        for b in ids:
            if a == b or not can_follow(a, b):
                continue
            key = (a, b)
            v = Bool(f"edge_{row['measure']}_{a}_{b}")
            edge[key] = v
            outgoing[a].append(v)
            incoming[b].append(v)
            opt.add(Implies(v, t[b] == t[a] + d[a]))

            ea, eb = by_id[a], by_id[b]
            cost = 0
            if key in hard_edge_set:
                cost = 0
            elif key in hint_edge_set:
                cost = 1
            else:
                da, db = ea.get("direction"), eb.get("direction")
                cost = 3 if da and db and da != db else 2
                gap = float(eb["x"]) - float(ea["x"])
                if gap > 8.0 * il:
                    cost += 2
            penalty_terms.append(If(v, IntVal(cost), IntVal(0)))

    # Exact path cover.
    for i in ids:
        opt.add(
            Sum([If(start[i], 1, 0)] + [If(x, 1, 0) for x in incoming[i]]) == 1
        )
        opt.add(
            Sum([If(end[i], 1, 0)] + [If(x, 1, 0) for x in outgoing[i]]) == 1
        )
        # Starts/ends are allowed, but gratuitous fragmentation is discouraged.
        penalty_terms.append(If(start[i], 4, 0))
        penalty_terms.append(If(end[i], 1, 0))
        # Mid-measure voice starts are legal but require evidence.
        penalty_terms.append(If(And(start[i], t[i] > 0), 2, 0))

    # Graphical beam chains are hard musical continuity.
    for a, b in hard_edge_set:
        if (a, b) not in edge:
            return {
                "measure": row["measure"], "ok": False,
                "reason": f"beam-edge-not-representable:{a}->{b}",
                "attacks": [], "attack_count": None
            }
        opt.add(edge[(a, b)])

    # OMR voice membership is only a soft adjacency suggestion.
    for a, b in hint_edge_set:
        if (a, b) in edge:
            penalty_terms.append(If(edge[(a, b)], 0, 3))

    # Global engraving order: substantial horizontal separation cannot reverse time.
    # Equality is allowed for horizontally displaced simultaneous voices.
    ordered = sorted(events, key=lambda e: (float(e["x"]), str(e["id"])))
    for ix, ea in enumerate(ordered):
        for eb in ordered[ix + 1:]:
            dx = float(eb["x"]) - float(ea["x"])
            if dx >= 0.85 * il:
                opt.add(t[ea["id"]] <= t[eb["id"]])
            if dx > 2.5 * il:
                break

    # Strong visual columns are soft simultaneity evidence.
    for ix, ea in enumerate(ordered):
        for eb in ordered[ix + 1:]:
            dx = abs(float(eb["x"]) - float(ea["x"]))
            if dx > 0.38 * il:
                break
            w = 7 if dx <= 0.14 * il else 3
            penalty_terms.append(If(t[ea["id"]] == t[eb["id"]], 0, w))

    # Anchor absolute bar time from the leftmost printed event.  This is a general
    # notation invariant: a measure cannot begin after its earliest note/rest glyph.
    leftmost = min(events, key=lambda e: (float(e["x"]), str(e["id"])))
    opt.add(t[leftmost["id"]] == 0)

    total_penalty = Sum(penalty_terms) if penalty_terms else IntVal(0)
    handle = opt.minimize(total_penalty)

    if opt.check() != sat:
        return {
            "measure": row["measure"], "ok": False, "reason": "unsat",
            "attacks": [], "attack_count": None
        }

    model = opt.model()
    best_pen = model.eval(total_penalty, model_completion=True).as_long()

    onset = {i: model.eval(t[i], model_completion=True).as_long() for i in ids}
    duration = {i: model.eval(d[i], model_completion=True).as_long() for i in ids}

    attack_ids = [i for i in ids if bool(by_id[i].get("attack")) and by_id[i].get("kind") == "note"]
    attack_set = sorted(set(onset[i] for i in attack_ids))

    # Detect whether a different TMA attack set exists at the same global optimum.
    ambiguous = False
    alt_attack_set = None
    if ambiguity_check and attack_ids:
        primary = list(attack_set)
        diff_terms = []
        # Some attack moves outside the primary set.
        for i in attack_ids:
            diff_terms.append(And(*[t[i] != p for p in primary]))
        # Or one primary onset disappears entirely.
        for p in primary:
            diff_terms.append(And(*[t[i] != p for i in attack_ids]))
        opt.push()
        opt.add(Or(*diff_terms))
        if opt.check() == sat:
            m2 = opt.model()
            p2 = m2.eval(total_penalty, model_completion=True).as_long()
            if p2 == best_pen:
                alt_attack_set = sorted(set(m2.eval(t[i], model_completion=True).as_long() for i in attack_ids))
                if alt_attack_set != attack_set:
                    ambiguous = True
        opt.pop()

    # Selected duration provenance.
    selected = {}
    for i in ids:
        chosen = None
        for k, (ticks, cost, why) in enumerate(cand[i]):
            if model.eval(choose[(i, k)], model_completion=True):
                chosen = {"ticks": ticks, "cost": cost, "why": why}
                break
        selected[i] = chosen

    attacks = []
    for tt in attack_set:
        members = sorted(i for i in attack_ids if onset[i] == tt)
        attacks.append({
            "onset_ticks": tt,
            "onset_whole_units": tick_text(tt),
            "onset_quarter_units": quarter_text(tt),
            "events": members,
        })

    event_rows = []
    for e in sorted(events, key=lambda x: (onset[x["id"]], float(x["x"]), x["id"])):
        i = e["id"]
        pred = None
        succ = None
        for (a, b), v in edge.items():
            if b == i and model.eval(v, model_completion=True):
                pred = a
            if a == i and model.eval(v, model_completion=True):
                succ = b
        event_rows.append({
            "id": i,
            "kind": e.get("kind"),
            "attack": bool(e.get("attack")),
            "staff": e.get("staff"),
            "x": e.get("x"),
            "onset_ticks": onset[i],
            "onset_quarter_units": quarter_text(onset[i]),
            "duration_ticks": duration[i],
            "duration_quarter_units": quarter_text(duration[i]),
            "duration_source": selected[i],
            "start": bool(model.eval(start[i], model_completion=True)),
            "end": bool(model.eval(end[i], model_completion=True)),
            "pred": pred,
            "succ": succ,
        })

    return {
        "measure": row["measure"],
        "page": row.get("page"),
        "system": row.get("system"),
        "ok": not ambiguous,
        "ambiguous": ambiguous,
        "reason": "ambiguous-optimal-attack-set" if ambiguous else None,
        "total_penalty": best_pen,
        "meter_ticks": meter_ticks,
        "attack_count": len(attacks) if not ambiguous else None,
        "attacks": attacks if not ambiguous else [],
        "primary_attack_set_quarter": [quarter_text(x) for x in attack_set],
        "alternate_attack_set_quarter": None if alt_attack_set is None else [quarter_text(x) for x in alt_attack_set],
        "events": event_rows,
    }


def solve_all(raw: dict) -> dict:
    out = {}
    for ms in sorted(raw, key=lambda x: int(x)):
        out[str(ms)] = solve_measure(raw[ms], ambiguity_check=False)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: global_csp_solver.py visual_attack_fullscore.json [out.json]")
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else "global_csp_solved.json"
    raw = json.load(open(inp))
    solved = solve_all(raw)
    Path(outp).write_text(json.dumps(solved, indent=2))

    ok = [int(m) for m, r in solved.items() if r.get("ok")]
    bad = [int(m) for m, r in solved.items() if not r.get("ok")]
    print("GLOBAL_CSP_SUMMARY=" + json.dumps({
        "measure_count": len(solved),
        "ok_count": len(ok),
        "unresolved_count": len(bad),
        "unresolved_measures": bad,
        "total_attacks": sum((r.get("attack_count") or 0) for r in solved.values()),
    }, separators=(",", ":")))
    for m in (4,5,6,7,28,34,39,46):
        r = solved.get(str(m))
        if r:
            print("GLOBAL_CSP_TARGET", m, json.dumps({
                "ok": r.get("ok"),
                "reason": r.get("reason"),
                "count": r.get("attack_count"),
                "q": [a["onset_quarter_units"] for a in r.get("attacks", [])],
                "primary": r.get("primary_attack_set_quarter"),
                "alternate": r.get("alternate_attack_set_quarter"),
                "penalty": r.get("total_penalty"),
            }, separators=(",", ":")))


if __name__ == "__main__":
    main()
