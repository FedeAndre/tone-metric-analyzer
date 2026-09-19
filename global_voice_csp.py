from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from z3 import And, Bool, If, Int, IntVal, Optimize, Or, Sum, sat

from global_csp_solver import F, TICKS_PER_WHOLE, to_ticks, quarter_text, tick_text, duration_candidates


def _beam_components(row, allowed):
    adj=defaultdict(set)
    for a,b in row.get("beam_edges",[]):
        if a in allowed and b in allowed:
            adj[a].add(b); adj[b].add(a)
    out=[]; seen=set()
    for s in list(adj):
        if s in seen: continue
        stack=[s]; seen.add(s); comp=[]
        while stack:
            q=stack.pop(); comp.append(q)
            for n in adj[q]:
                if n not in seen:
                    seen.add(n); stack.append(n)
        out.append(comp)
    return out


def build_segments(row):
    by={e["id"]:e for e in row.get("events",[])}
    used=set(); segs=[]

    for k,v in enumerate(row.get("voice_hints",[])):
        ids=[i for i in v.get("ids",[]) if i in by and i not in used]
        if not ids: continue
        ids=sorted(ids,key=lambda i:(float(by[i]["x"]),i))
        used.update(ids)
        segs.append({
            "key":f"omr:{v.get('part')}:{v.get('voice')}:{k}",
            "kind":"omr-voice","ids":ids
        })

    remain=set(by)-used
    for ci,comp in enumerate(_beam_components(row,remain)):
        ids=sorted(comp,key=lambda i:(float(by[i]["x"]),i))
        if not ids: continue
        used.update(ids)
        remain.difference_update(ids)
        segs.append({"key":f"beam:{ci}","kind":"beam-orphan","ids":ids})

    for i in sorted(remain,key=lambda i:(float(by[i]["x"]),i)):
        segs.append({"key":f"single:{i}","kind":"singleton","ids":[i]})
        used.add(i)

    assert used==set(by)
    return segs


def solve_measure(row, check_ambiguity=True):
    events=[dict(e) for e in row.get("events",[])]
    if not events:
        return {"measure":row["measure"],"ok":False,"reason":"no-events","attacks":[],"attack_count":None}
    by={e["id"]:e for e in events}
    meter=to_ticks(F(row["nominal_whole_units"]))
    il=float(row.get("interline") or 20.0)
    segs=build_segments(row)

    opt=Optimize()
    opt.set(timeout=5000)

    t={i:Int(f"vt_{row['measure']}_{i}") for i in by}
    d={i:Int(f"vd_{row['measure']}_{i}") for i in by}
    start={s["key"]:Int(f"vs_{row['measure']}_{n}") for n,s in enumerate(segs)}
    choose={}
    penalties=[]

    # Duration evidence.
    candidates={}
    for i,e in by.items():
        cs=duration_candidates(e,meter)
        if not cs:
            return {"measure":row["measure"],"ok":False,"reason":f"no-duration:{i}","attacks":[],"attack_count":None}
        candidates[i]=cs
        bs=[]
        for k,(ticks,cost,why) in enumerate(cs):
            b=Bool(f"vc_{row['measure']}_{i}_{k}")
            choose[(i,k)]=b; bs.append(b)
            penalties.append(If(b,IntVal(cost*6),0))
        opt.add(Sum([If(b,1,0) for b in bs])==1)
        opt.add(d[i]==Sum([If(choose[(i,k)],ticks,0) for k in range(len(cs))]))
        opt.add(t[i]>=0,t[i]<meter,t[i]+d[i]<=meter)

    seg_of={}
    for s in segs:
        for i in s["ids"]:
            seg_of[i]=s["key"]

    # Exact within-segment chronology from selected graphical durations.
    for s in segs:
        ids=s["ids"]
        sk=s["key"]
        opt.add(start[sk]>=0,start[sk]<meter)
        opt.add(t[ids[0]]==start[sk])
        for a,b in zip(ids,ids[1:]):
            opt.add(t[b]==t[a]+d[a])
        opt.add(t[ids[-1]]+d[ids[-1]]<=meter)

    # Basic rhythmic quantum from the score itself: no invented exotic offsets.
    prim=[]
    for e in events:
        if e.get("duration_whole_units"):
            prim.append(to_ticks(F(e["duration_whole_units"])))
    quantum=prim[0] if prim else 1
    for q in prim[1:]:
        quantum=math.gcd(quantum,q)
    quantum=max(1,quantum)
    for i in by:
        opt.add(t[i] % quantum == 0)

    ordered=sorted(events,key=lambda e:(float(e["x"]),e["id"]))

    # Earliest visible glyph defines bar onset.  All events in its very tight
    # visual column are onset 0; this works for notes, rests and tied continuations.
    minx=float(ordered[0]["x"])
    firstcol=[e for e in ordered if float(e["x"])-minx<=0.18*il]
    for e in firstcol:
        opt.add(t[e["id"]]==0)

    # Full-measure symbols/segments are absolute anchors.
    for s in segs:
        ids=s["ids"]
        # If every selected duration happens to fill the bar, its start must be 0.
        span=Sum([d[i] for i in ids])
        opt.add(If(span==meter,start[s["key"]]==0,True))

        first=by[ids[0]]
        # A tied continuation at the first event of a segment continues across the barline.
        if first.get("kind")=="note" and not first.get("attack") and int(first.get("tied_head_count") or 0)>0:
            opt.add(start[s["key"]]==0)
        # Whole-bar rest.
        if len(ids)==1 and first.get("kind")=="rest" and first.get("duration_whole_units"):
            if to_ticks(F(first["duration_whole_units"]))==meter:
                opt.add(start[s["key"]]==0)

    # Global engraving order.  Large x separation may represent equal onsets
    # in displaced voices, but never temporal reversal.
    for aidx,ea in enumerate(ordered):
        for eb in ordered[aidx+1:]:
            dx=float(eb["x"])-float(ea["x"])
            if dx>=0.80*il:
                opt.add(t[ea["id"]]<=t[eb["id"]])
            if dx>3.0*il:
                break

    # Visual simultaneity is evidence, not a hard rule.
    for aidx,ea in enumerate(ordered):
        for eb in ordered[aidx+1:]:
            dx=abs(float(eb["x"])-float(ea["x"]))
            if dx>0.55*il:
                break
            w=12 if dx<=0.12*il else 7 if dx<=0.25*il else 3
            penalties.append(If(t[ea["id"]]==t[eb["id"]],0,w))

    # Segment-edge evidence.  Left-edge starts and right-edge completions are
    # general engraving cues and are weighted rather than patched.
    left=float(row.get("bar_left_x") or minx)
    right=float(row.get("bar_right_x") or max(float(e["x"]) for e in ordered))
    for s in segs:
        ids=s["ids"]; first=by[ids[0]]; last=by[ids[-1]]
        sx=float(first["x"]); lx=float(last["x"])
        endexpr=t[ids[-1]]+d[ids[-1]]
        if sx-left <= 1.7*il:
            penalties.append(If(start[s["key"]]==0,0,8))
        if right-lx <= 2.0*il:
            penalties.append(If(endexpr==meter,0,8))
        # Avoid arbitrary floating segments when the visual evidence is otherwise equal.
        penalties.append(If(start[s["key"]]==0,0,1))
        penalties.append(If(endexpr==meter,0,1))

    # Same-direction neighboring orphan segments are likely continuations, but
    # this remains soft because voices can cross or change stem direction.
    for sa in segs:
        for sb in segs:
            if sa is sb: continue
            a=by[sa["ids"][-1]]; b=by[sb["ids"][0]]
            dx=float(b["x"])-float(a["x"])
            if dx<=0 or dx>3.5*il: continue
            if a.get("staff")!=b.get("staff"): continue
            bonus=3
            if a.get("direction") and b.get("direction") and a.get("direction")==b.get("direction"):
                bonus=5
            penalties.append(If(start[sb["key"]]==t[a["id"]]+d[a["id"]],0,bonus))

    total=Sum(penalties) if penalties else IntVal(0)
    opt.minimize(total)
    chk=opt.check()
    if chk!=sat:
        return {"measure":row["measure"],"ok":False,"reason":"unsat-or-timeout","attacks":[],"attack_count":None}

    m=opt.model()
    best=m.eval(total,model_completion=True).as_long()
    onset={i:m.eval(t[i],model_completion=True).as_long() for i in by}
    dur={i:m.eval(d[i],model_completion=True).as_long() for i in by}

    attack_ids=[i for i,e in by.items() if e.get("kind")=="note" and bool(e.get("attack"))]
    primary_set=sorted(set(onset[i] for i in attack_ids))

    ambiguous=False; alt_set=None
    if check_ambiguity and attack_ids:
        diff=[]
        for i in attack_ids:
            diff.append(And(*[t[i]!=p for p in primary_set]))
        for p in primary_set:
            diff.append(And(*[t[i]!=p for i in attack_ids]))
        opt.push(); opt.add(Or(*diff))
        if opt.check()==sat:
            m2=opt.model(); p2=m2.eval(total,model_completion=True).as_long()
            if p2==best:
                alt_set=sorted(set(m2.eval(t[i],model_completion=True).as_long() for i in attack_ids))
                ambiguous=alt_set!=primary_set
        opt.pop()

    selected={}
    for i in by:
        for k,(ticks,cost,why) in enumerate(candidates[i]):
            if bool(m.eval(choose[(i,k)],model_completion=True)):
                selected[i]={"ticks":ticks,"cost":cost,"why":why}
                break

    attacks=[]
    for tt in primary_set:
        members=sorted(i for i in attack_ids if onset[i]==tt)
        attacks.append({
            "onset_ticks":tt,
            "onset_whole_units":tick_text(tt),
            "onset_quarter_units":quarter_text(tt),
            "events":members
        })

    erows=[]
    for e in sorted(events,key=lambda q:(onset[q["id"]],float(q["x"]),q["id"])):
        i=e["id"]
        erows.append({
            "id":i,"kind":e.get("kind"),"attack":bool(e.get("attack")),
            "staff":e.get("staff"),"x":e.get("x"),"segment":seg_of[i],
            "onset_ticks":onset[i],"onset_quarter_units":quarter_text(onset[i]),
            "duration_ticks":dur[i],"duration_quarter_units":quarter_text(dur[i]),
            "duration_source":selected.get(i)
        })

    return {
        "measure":row["measure"],"page":row.get("page"),"system":row.get("system"),
        "ok":not ambiguous,"ambiguous":ambiguous,
        "reason":"ambiguous-optimal-attack-set" if ambiguous else None,
        "total_penalty":best,"quantum_ticks":quantum,"meter_ticks":meter,
        "attack_count":None if ambiguous else len(attacks),
        "attacks":[] if ambiguous else attacks,
        "primary_attack_set_quarter":[quarter_text(x) for x in primary_set],
        "alternate_attack_set_quarter":None if alt_set is None else [quarter_text(x) for x in alt_set],
        "segments":segs,"events":erows
    }


def main():
    if len(sys.argv)<2:
        raise SystemExit("usage: global_voice_csp.py global_symbol_graph.json [out.json]")
    raw=json.load(open(sys.argv[1]))
    out={}
    for ms in sorted(raw,key=lambda x:int(x)):
        out[str(ms)]=solve_measure(raw[ms],check_ambiguity=False)
    dst=sys.argv[2] if len(sys.argv)>2 else "global_voice_solved.json"
    Path(dst).write_text(json.dumps(out,indent=2))
    ok=[int(m) for m,r in out.items() if r.get("ok")]
    bad=[int(m) for m,r in out.items() if not r.get("ok")]
    print("GLOBAL_VOICE_SUMMARY="+json.dumps({
        "measure_count":len(out),"ok_count":len(ok),"unresolved_count":len(bad),
        "unresolved_measures":bad,"total_attacks":sum((r.get("attack_count") or 0) for r in out.values())
    },separators=(",",":")))
    for mm in (4,5,6,7,28,34,39,46):
        r=out[str(mm)]
        print("GLOBAL_VOICE_TARGET",mm,json.dumps({
            "ok":r.get("ok"),"reason":r.get("reason"),"count":r.get("attack_count"),
            "q":[a["onset_quarter_units"] for a in r.get("attacks",[])],
            "primary":r.get("primary_attack_set_quarter"),"penalty":r.get("total_penalty")
        },separators=(",",":")))


if __name__=="__main__":
    main()
