from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path


def F(s: str) -> Fraction:
    return Fraction(s)


def _voice_raw(voice):
    events=[]
    for e in voice.get("events",[]):
        events.append({
            "cid":e["cid"],
            "type":e["type"],
            "x":float(e["x"]),
            "onset":F(e["onset"]),
            "duration":F(e["duration"]),
            "attack":bool(e["attack"]),
        })
    return events


_STANDARD_DURS = sorted({
    Fraction(1,1), Fraction(7,8), Fraction(3,4), Fraction(1,2),
    Fraction(7,16), Fraction(3,8), Fraction(1,4),
    Fraction(7,32), Fraction(3,16), Fraction(1,8),
    Fraction(7,64), Fraction(3,32), Fraction(1,16),
    Fraction(3,64), Fraction(1,32), Fraction(1,64),
}, reverse=True)


def _duration_candidates(event, nominal):
    d0=event["duration"]
    vals={d0}
    for d in _STANDARD_DURS:
        if d <= nominal and d0/Fraction(4,1) <= d <= d0*4:
            vals.add(d)

    def cost(d):
        if d == d0:
            return 0.0
        ratio=float(d/d0)
        c=1.0 + abs(math.log2(ratio))
        if d*2 == d0 or d == d0*2:
            c=1.0
        if d*4 == d0*3 or d*3 == d0*4:
            c=min(c,1.10)
        if d*3 == d0*2 or d*2 == d0*3:
            c=min(c,1.15)
        if d*8 == d0*7 or d*7 == d0*8:
            c=min(c,1.20)
        if event["type"] == "rest-chord":
            c += 0.75
        return c

    return sorted(((cost(d),d) for d in vals), key=lambda q:(q[0],q[1]))


def solve_measure(row, x_tol: float = 12.0):
    nominal=F(row["nominal_whole_units"])
    voices=[]
    for i,v in enumerate(row.get("voices",[])):
        ev=_voice_raw(v)
        span=F(v["span"])
        voices.append({
            "index":i,"part":v["part"],"voice":v["voice"],
            "events":ev,"span":span,"resolved":False,"offset":None,
            "method":None,"reason":None,"repair_cost":None,
        })

    for v in voices:
        if v["events"] and v["span"] == nominal:
            v["resolved"]=True
            v["offset"]=Fraction(0)
            v["method"]="full-span"

    if not any(v["resolved"] for v in voices):
        firsts=[v["events"][0]["x"] for v in voices if v["events"]]
        if firsts:
            left=min(firsts)
            for v in voices:
                if v["events"] and abs(v["events"][0]["x"]-left) <= 5.0 and v["span"] <= nominal:
                    v["resolved"]=True
                    v["offset"]=Fraction(0)
                    v["method"]="left-edge-seed"

    def ref_points():
        pts=[]
        for v in voices:
            if not v["resolved"]:
                continue
            off=v["offset"]
            for e in v["events"]:
                pts.append((e["x"], e["onset"]+off, v["index"]))
        return pts

    def place_partials():
        changed=True
        any_change=False
        while changed:
            changed=False
            refs=ref_points()
            for v in voices:
                if v["resolved"] or not v["events"] or v["span"] > nominal:
                    continue
                candidates=[]
                evidence=[]
                for e in v["events"]:
                    near=[(abs(e["x"]-rx), rt, rvi) for rx,rt,rvi in refs if abs(e["x"]-rx) <= x_tol]
                    if not near:
                        continue
                    dist,rt,rvi=min(near,key=lambda q:(q[0],q[1]))
                    off=rt-e["onset"]
                    if 0 <= off and off+v["span"] <= nominal:
                        candidates.append(off)
                        evidence.append((e["cid"],e["x"],rt,dist,off,rvi))
                if not candidates:
                    continue
                counts=Counter(candidates)
                off,n=counts.most_common(1)[0]
                best_dist=min(ev[3] for ev in evidence if ev[4]==off)
                if n >= 2 or best_dist <= 5.0:
                    v["resolved"]=True
                    v["offset"]=off
                    v["method"]="x-anchor"
                    v["anchor_count"]=n
                    v["anchor_best_px"]=best_dist
                    changed=True
                    any_change=True
        return any_change

    place_partials()

    def anchor_constraints(v, refs):
        anchors={}
        for i,e in enumerate(v["events"]):
            near=[(abs(e["x"]-rx),rt,rvi) for rx,rt,rvi in refs if rvi != v["index"] and abs(e["x"]-rx) <= x_tol]
            if not near:
                continue
            near.sort(key=lambda q:(q[0],q[1]))
            best=near[0][0]
            close=[q for q in near if q[0] <= min(x_tol,best+1.0)]
            times=Counter(q[1] for q in close)
            t,n=times.most_common(1)[0]
            if n >= 2 or best <= 5.0:
                anchors[i]=(t,best,n)
        return anchors

    def repair_voice(v):
        if v["resolved"] or not v["events"]:
            return False
        refs=ref_points()
        if not refs:
            return False
        anchors=anchor_constraints(v,refs)

        offset=None
        if 0 in anchors:
            offset=anchors[0][0]
        else:
            zero_refs=[(abs(v["events"][0]["x"]-rx),rt) for rx,rt,_ in refs if rt == 0]
            if zero_refs and min(zero_refs)[0] <= 5.0:
                offset=Fraction(0)
        if offset is None or offset < 0 or offset >= nominal:
            return False

        must_repair = v["span"] > nominal or offset + v["span"] > nominal
        if not must_repair:
            return False

        anchor_times={i:t for i,(t,_,_) in anchors.items()}
        states={offset:(0.0,[],[])}

        for i,e in enumerate(v["events"]):
            nxt={}
            for t,(cost,chosen,onsets) in states.items():
                if i in anchor_times and t != anchor_times[i]:
                    continue
                if not (Fraction(0) <= t < nominal):
                    continue
                for dc,d in _duration_candidates(e,nominal):
                    nt=t+d
                    if nt > nominal:
                        continue
                    nc=cost+dc
                    prev=nxt.get(nt)
                    if prev is None or nc < prev[0]:
                        nxt[nt]=(nc,chosen+[d],onsets+[t])
            states=nxt
            if not states:
                return False

        if nominal not in states:
            return False
        cost,chosen,onsets=states[nominal]

        changed=sum(1 for e,d in zip(v["events"],chosen) if e["duration"] != d)
        if changed == 0 or changed > max(3, len(v["events"])//2):
            return False

        for e,d,t in zip(v["events"],chosen,onsets):
            e["duration"]=d
            e["onset"]=t-offset
        v["span"]=nominal-offset
        v["resolved"]=True
        v["offset"]=offset
        v["method"]="constraint-repair"
        v["repair_cost"]=round(cost,4)
        v["repair_changes"]=changed
        v["anchor_count"]=len(anchor_times)
        v["anchor_best_px"]=min((q[1] for q in anchors.values()),default=None)
        return True

    changed=True
    while changed:
        changed=False
        for v in voices:
            if repair_voice(v):
                changed=True
                place_partials()
        if place_partials():
            changed=True

    for v in voices:
        if v["resolved"]:
            continue
        if v["span"] > nominal:
            v["reason"]="overfull"
        elif not v["events"]:
            v["reason"]="empty"
        else:
            refs=ref_points()
            anchors=anchor_constraints(v,refs) if refs else {}
            if 0 in anchors and anchors[0][0]+v["span"] > nominal:
                v["reason"]="anchored-overflow"
            else:
                v["reason"]="ambiguous-offset"

    attacks={}
    for v in voices:
        if not v["resolved"]:
            continue
        off=v["offset"]
        for e in v["events"]:
            if e["attack"]:
                t=e["onset"]+off
                if not (Fraction(0) <= t < nominal):
                    v["resolved"]=False
                    v["reason"]="attack-outside-measure"
                    break
                attacks.setdefault(t,[]).append(e["cid"])

    unresolved=[v for v in voices if not v["resolved"]]
    ok=not unresolved
    attack_rows=[
        {
            "onset_whole_units":str(t),
            "onset_quarter_units":str(t*4),
            "chords":sorted(cids),
        }
        for t,cids in sorted(attacks.items())
    ] if ok else []

    return {
        "measure":row["measure"],
        "page":row["page"],
        "system":row["system"],
        "nominal_whole_units":row["nominal_whole_units"],
        "ok":ok,
        "attack_count":len(attack_rows) if ok else None,
        "attacks":attack_rows,
        "voices":[
            {
                "part":v["part"],"voice":v["voice"],
                "span":str(v["span"]),
                "resolved":v["resolved"],
                "offset":str(v["offset"]) if v["offset"] is not None else None,
                "method":v["method"],
                "reason":v["reason"],
                "repair_cost":v.get("repair_cost"),
                "repair_changes":v.get("repair_changes"),
                "anchor_count":v.get("anchor_count"),
                "anchor_best_px":v.get("anchor_best_px"),
            }
            for v in voices
        ],
    }


def solve_all(raw):
    return {str(m):solve_measure(row) for m,row in raw.items()}


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("raw_json")
    ap.add_argument("--out",default="visual_attack_solved.json")
    args=ap.parse_args()
    raw=json.load(open(args.raw_json))
    solved=solve_all(raw)
    Path(args.out).write_text(json.dumps(solved,indent=2))
    ok=[int(m) for m,r in solved.items() if r["ok"]]
    bad=[int(m) for m,r in solved.items() if not r["ok"]]
    print("VISUAL_SOLVER measures=%d ok=%d unresolved=%d"%(len(solved),len(ok),len(bad)))
    print("VISUAL_SOLVER_OK",ok)
    print("VISUAL_SOLVER_UNRESOLVED",bad)
    for m in (6,7,9,17,23,28,34,39,43,45):
        if str(m) in solved:
            print("SOLVER_TARGET",m,json.dumps(solved[str(m)],separators=(",",":")))
