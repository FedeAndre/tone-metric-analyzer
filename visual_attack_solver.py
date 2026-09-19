from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
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


def solve_measure(row, x_tol: float = 12.0):
    nominal=F(row["nominal_whole_units"])
    voices=[]
    for i,v in enumerate(row.get("voices",[])):
        ev=_voice_raw(v)
        span=F(v["span"])
        voices.append({
            "index":i,"part":v["part"],"voice":v["voice"],
            "events":ev,"span":span,"resolved":False,"offset":None,
            "method":None,"reason":None,
        })

    # A voice that exactly fills the measure has an unambiguous zero offset.
    for v in voices:
        if v["events"] and v["span"] == nominal:
            v["resolved"]=True
            v["offset"]=Fraction(0)
            v["method"]="full-span"

    def ref_points():
        pts=[]
        for v in voices:
            if not v["resolved"]:
                continue
            off=v["offset"]
            for e in v["events"]:
                pts.append((e["x"], e["onset"]+off, v["index"]))
        return pts

    # Iteratively place partial voices by geometric coincidence with resolved voices.
    changed=True
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
            # Require either two independent coincidences, or one very close coincidence.
            best_dist=min(ev[3] for ev in evidence if ev[4]==off)
            if n >= 2 or best_dist <= 3.0:
                v["resolved"]=True
                v["offset"]=off
                v["method"]="x-anchor"
                v["anchor_count"]=n
                v["anchor_best_px"]=best_dist
                changed=True

    # Overfull voices are impossible as currently interpreted. Partial voices without
    # a geometric anchor are left unresolved instead of guessed.
    for v in voices:
        if v["resolved"]:
            continue
        if v["span"] > nominal:
            v["reason"]="overfull"
        elif not v["events"]:
            v["reason"]="empty"
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
    for m in (7,28,34,39):
        print("SOLVER_TARGET",m,json.dumps(solved[str(m)],separators=(",",":")))
