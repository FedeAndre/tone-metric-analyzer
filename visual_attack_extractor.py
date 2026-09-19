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

OUT = Path("/work/visual_attack_audit")
OUT.mkdir(parents=True, exist_ok=True)

def sheet_number(name: str) -> int:
    m = re.search(r"sheet#(\d+)", name, re.I)
    return int(m.group(1)) if m else 10**9

def flabel(v):
    return None if v is None else str(v)

def get_duration(by, chord_heads, chord_stem, flags, dots, head_black, rest_child, cid, levels):
    node = by.get(cid)
    if node is None:
        return None
    typ = loc(node.tag)
    if typ == "rest-chord":
        rr = by.get(rest_child.get(cid))
        return _rest_duration(rr.get("shape") if rr is not None else "")
    if typ != "head-chord":
        return None
    hs = chord_heads.get(cid, [])
    if not hs:
        return None
    if any(head_black.get(h, False) for h in hs):
        base = Fraction(1,4)
    else:
        shapes = [(by[h].get("shape") or "").upper() for h in hs if h in by]
        base = Fraction(2) if any("BREVE" in s for s in shapes) else Fraction(1) if any("WHOLE" in s for s in shapes) else Fraction(1,2)
    st = chord_stem.get(cid)
    lev = max(levels.get(st or "", 0), flags.get(st or "", 0))
    d0 = base / (2**lev)
    dn = max([dots.get(h,0) for h in hs] or [0])
    return d0 * sum(Fraction(1,2**i) for i in range(dn+1))

def feasible_starts(gap: Fraction):
    if gap < 0:
        return []
    # 1/128 whole note = 1/32 quarter note; enough for all observed values here.
    step = Fraction(1,128)
    n = int(gap / step)
    vals = [step*i for i in range(n+1)]
    if not vals or vals[-1] != gap:
        vals.append(gap)
    return sorted(set(vals))

def choose_start(seq, gap, anchors, tol_x, nominal):
    # seq: [{x, rel, duration, attack, rest}]
    if gap < 0:
        return None, "overfull"
    if gap == 0:
        return Fraction(0), "full-span"
    if seq and seq[0]["rest"]:
        return Fraction(0), "leading-rest"
    cands = feasible_starts(gap)
    if not anchors:
        return None, "no-anchor"

    scored = []
    for s in cands:
        penalty = 0.0
        exact_matches = 0
        order_viol = 0
        for e in seq:
            t = s + e["rel"]
            # nearest visual anchor
            near = sorted(anchors, key=lambda a: abs(a["x"]-e["x"]))
            if near:
                a = near[0]
                dx = abs(a["x"]-e["x"])
                if dx <= tol_x:
                    exact_matches += 1
                    penalty += 200.0 * float(abs(t-a["t"]))
                # hard-ish visual ordering constraints
                for q in near[:8]:
                    if e["x"] < q["x"]-tol_x and t > q["t"]:
                        order_viol += 1
                    elif e["x"] > q["x"]+tol_x and t < q["t"]:
                        order_viol += 1
            if t < 0 or t >= nominal + Fraction(1,128):
                penalty += 10000
        penalty += 1000.0*order_viol - 25.0*exact_matches
        scored.append((penalty, s, exact_matches, order_viol))
    scored.sort(key=lambda x:(x[0], x[1]))
    if not scored:
        return None, "no-candidate"
    best = scored[0]
    if len(scored) > 1 and abs(scored[1][0]-best[0]) < 1e-9 and scored[1][1] != best[1]:
        return None, "ambiguous"
    # Require at least one close anchor or a zero-violation uniquely best solution.
    if best[2] == 0 and best[3] > 0:
        return None, "weak-anchor"
    return best[1], f"anchored(matches={best[2]},viol={best[3]})"

def main(omr_path: str):
    all_measures = {}
    page_measure_base = 0

    with ZipFile(omr_path) as archive:
        members = sorted(
            [n for n in archive.namelist() if re.search(r"sheet#\d+/sheet#\d+\.xml$", n, re.I)],
            key=lambda n:(sheet_number(n), n.lower())
        )
        for page_index, member in enumerate(members):
            root = etree.fromstring(archive.read(member))
            by = {e.get("id"):e for e in root.iter() if e.get("id")}
            il = _interline(root)
            pic = next(e for e in root.iter() if loc(e.tag)=="picture")
            W,H = float(pic.get("width")), float(pic.get("height"))
            binary_name = member.rsplit("/",1)[0] + "/BINARY.png"
            binary = np.array(Image.open(BytesIO(archive.read(binary_name))).convert("L")) < 128

            chord_heads=defaultdict(list); chord_stem={}; semantic_beams=defaultdict(set)
            flags=defaultdict(int); dots=defaultdict(int); rest_child={}
            slur_heads=defaultdict(dict); slur_ext=defaultdict(set); explicit_ties=set()

            for rel in (e for e in root.iter() if loc(e.tag)=="relation"):
                child=next(iter(rel),None)
                if child is None: continue
                k=loc(child.tag); s,t=rel.get("source"),rel.get("target")
                if k=="containment" and s in by and t in by:
                    if loc(by[s].tag)=="head-chord" and loc(by[t].tag)=="head":
                        chord_heads[s].append(t)
                    elif loc(by[s].tag)=="rest-chord" and loc(by[t].tag)=="rest":
                        rest_child[s]=t
                elif k=="chord-stem": chord_stem[s]=t
                elif k=="beam-stem": semantic_beams[t].add(s)
                elif k=="flag-stem":
                    shape=(by.get(s).get("shape") if by.get(s) is not None else "") or ""
                    m=re.search(r"FLAG_(\d+)",shape)
                    flags[t]=max(flags[t],int(m.group(1)) if m else 1)
                elif k=="augmentation": dots[t]+=1
                elif k=="slur-head":
                    sl=by.get(s); side=(child.get("side") or "").upper()
                    if sl is not None and loc(sl.tag)=="slur":
                        slur_heads[s][side]=t
                        if (sl.get("tie") or "").lower()=="true": explicit_ties.add(s)

            for sid,node in by.items():
                if loc(node.tag)=="slur":
                    for attr in ("left-extension","right-extension"):
                        q=node.get(attr)
                        if q:
                            slur_ext[sid].add(q); slur_ext[q].add(sid)

            tied_right=set(); tie_starts=set(); seen=set()
            for sid in slur_heads:
                if sid in seen: continue
                comp=[]; stack=[sid]; seen.add(sid)
                while stack:
                    q=stack.pop(); comp.append(q)
                    for n in slur_ext.get(q,set()):
                        if n not in seen:
                            seen.add(n); stack.append(n)
                lefts=[slur_heads[s]["LEFT"] for s in comp if "LEFT" in slur_heads.get(s,{})]
                rights=[slur_heads[s]["RIGHT"] for s in comp if "RIGHT" in slur_heads.get(s,{})]
                explicit=any(s in explicit_ties for s in comp)
                inferred=False
                if not explicit and len(comp)>1 and lefts and rights:
                    lp={by[h].get("pitch") for h in lefts if h in by}
                    rp={by[h].get("pitch") for h in rights if h in by}
                    inferred=bool(lp & rp)
                if explicit or inferred:
                    tied_right.update(rights); tie_starts.update(lefts)

            head_black={}
            for hid,node in by.items():
                if loc(node.tag)!="head": continue
                b=get_box(node)
                if b is None: continue
                shape=(node.get("shape") or "").upper()
                ratio=_head_fill_ratio(binary,b)
                val="BLACK" in shape
                if "VOID" in shape and ratio>=.80: val=True
                elif "BLACK" in shape and ratio<=.75: val=False
                head_black[hid]=val

            page_base=page_measure_base
            systems=[e for e in root.iter() if loc(e.tag)=="system"]
            for sy,system in enumerate(systems):
                top,bottom=_system_bounds(system,H)
                stacks=[e for e in system if loc(e.tag)=="stack"]
                parts=[e for e in system if loc(e.tag)=="part"]
                staffids={e.get("id") for e in system.iter() if loc(e.tag)=="staff" and e.get("id")}
                sys_base=page_base
                bounds=[(float(s.get("left")),float(s.get("right"))) for s in stacks]
                nominal=[Fraction(s.get("expected") or s.get("duration") or "1") for s in stacks]

                # Geometry records for note and rest chords.
                geom={}
                by_ms=defaultdict(list)
                for cid,node in by.items():
                    typ=loc(node.tag)
                    if typ not in ("head-chord","rest-chord") or node.get("staff") not in staffids:
                        continue
                    cb=get_box(node)
                    if cb is None: continue
                    mis=[i for i,(l,r) in enumerate(bounds) if l<=cb.cx<=r]
                    if len(mis)!=1: continue
                    mi=mis[0]
                    hs=chord_heads.get(cid,[])
                    hbs=[get_box(by[h]) for h in hs if h in by and get_box(by[h])]
                    x=cb.cx if not hbs else float(median([b.cx for b in hbs]))
                    direction=None
                    st=chord_stem.get(cid)
                    sb=get_box(by.get(st))
                    if sb and hbs:
                        direction="up" if sb.cx>=median([b.cx for b in hbs]) else "down"
                    rec={"id":cid,"type":typ,"staff":node.get("staff") or "","mi":mi,"x":x,
                         "heads":list(hs),"stem":st,"direction":direction}
                    geom[cid]=rec; by_ms[(mi,rec["staff"])].append(rec)

                # Beam levels: semantic recognition plus direct pixel completion.
                levels={s:len(bs) for s,bs in semantic_beams.items()}
                for (mi,staff),arr in by_ms.items():
                    arr=sorted([c for c in arr if c["stem"] and c["direction"]],key=lambda c:c["x"])
                    for a,b in zip(arr,arr[1:]):
                        if a["direction"]!=b["direction"]: continue
                        sa,sb=get_box(by.get(a["stem"])),get_box(by.get(b["stem"]))
                        if not sa or not sb: continue
                        ta=(sa.cx,sa.y if a["direction"]=="up" else sa.y+sa.h,a["direction"])
                        tb=(sb.cx,sb.y if b["direction"]=="up" else sb.y+sb.h,b["direction"])
                        lv=_pixel_beam_levels(binary,ta,tb,il)
                        if lv:
                            levels[a["stem"]]=max(levels.get(a["stem"],0),lv)
                            levels[b["stem"]]=max(levels.get(b["stem"],0),lv)

                durations={}
                for cid in geom:
                    durations[cid]=get_duration(by,chord_heads,chord_stem,flags,dots,head_black,rest_child,cid,levels)

                # Voice membership/order only. Deliberately never reads slot time-offset/x-offset.
                voices_by_measure=defaultdict(list)
                for mi,stack_node in enumerate(stacks):
                    for pi,part in enumerate(parts):
                        measures=[e for e in part if loc(e.tag)=="measure"]
                        if mi>=len(measures): continue
                        for voice in [e for e in measures[mi] if loc(e.tag)=="voice"]:
                            ids=[]; seen_ids=set()
                            for ent in [e for e in voice.iter() if loc(e.tag)=="entry"]:
                                val=next((c for c in ent if loc(c.tag)=="value"),None)
                                if val is None or (val.get("status") or "").upper()!="BEGIN": continue
                                cid=val.get("chord")
                                if not cid or cid in seen_ids or cid not in geom: continue
                                seen_ids.add(cid); ids.append(cid)
                            if ids:
                                voices_by_measure[mi].append({"part":pi,"voice":voice.get("id") or "1","ids":ids})

                for mi,(left,right) in enumerate(bounds):
                    gm=sys_base+mi+1; meter=nominal[mi]
                    voice_recs=[]
                    for v in voices_by_measure.get(mi,[]):
                        rel=Fraction(0); seq=[]; bad=False
                        for cid in v["ids"]:
                            d=durations.get(cid)
                            if d is None:
                                bad=True
                                seq.append({"cid":cid,"x":geom[cid]["x"],"rel":rel,"duration":None,
                                            "rest":geom[cid]["type"]=="rest-chord","attack":False})
                                continue
                            attack=(geom[cid]["type"]=="head-chord" and any(h not in tied_right for h in geom[cid]["heads"]))
                            seq.append({"cid":cid,"x":geom[cid]["x"],"rel":rel,"duration":d,
                                        "rest":geom[cid]["type"]=="rest-chord","attack":attack})
                            rel+=d
                        voice_recs.append({"part":v["part"],"voice":v["voice"],"seq":seq,"span":rel,
                                           "unknown_duration":bad,"start":None,"reason":None})

                    anchors=[]
                    unresolved=[]
                    # Pass 1: deterministic full-span / explicit leading-rest voices.
                    for vr in voice_recs:
                        if vr["unknown_duration"]:
                            vr["reason"]="unknown-duration"; unresolved.append(vr); continue
                        gap=meter-vr["span"]
                        if gap<0:
                            vr["reason"]="overfull"; unresolved.append(vr); continue
                        if gap==0 or (vr["seq"] and vr["seq"][0]["rest"]):
                            vr["start"]=Fraction(0)
                            vr["reason"]="full-span" if gap==0 else "leading-rest"
                            for e in vr["seq"]:
                                if not e["rest"]:
                                    anchors.append({"x":e["x"],"t":e["rel"]})
                    # Pass 2: use exact visual alignments against resolved voices.
                    changed=True
                    while changed:
                        changed=False
                        for vr in voice_recs:
                            if vr["start"] is not None or vr["unknown_duration"] or vr["reason"]=="overfull":
                                continue
                            s,reason=choose_start(vr["seq"],meter-vr["span"],anchors,.45*il,meter)
                            if s is not None:
                                vr["start"]=s; vr["reason"]=reason; changed=True
                                for e in vr["seq"]:
                                    if not e["rest"]:
                                        anchors.append({"x":e["x"],"t":s+e["rel"]})
                    for vr in voice_recs:
                        if vr["start"] is None and vr not in unresolved:
                            vr["reason"]=vr["reason"] or "unresolved-start"; unresolved.append(vr)

                    attacks=defaultdict(list)
                    note_rows=[]
                    for cid,g in geom.items():
                        if g["mi"]!=mi or g["type"]!="head-chord": continue
                        attack_heads=[h for h in g["heads"] if h not in tied_right]
                        note_rows.append({"cid":cid,"x":g["x"],"staff":g["staff"],"duration":flabel(durations.get(cid)),
                                          "heads":g["heads"],"attack_heads":attack_heads})
                    for vr in voice_recs:
                        if vr["start"] is None: continue
                        for e in vr["seq"]:
                            if not e["attack"]: continue
                            t=vr["start"]+e["rel"]
                            if 0<=t<meter:
                                attacks[t].append(e["cid"])

                    attack_rows=[{"onset_whole":str(t),"onset_quarter":str(t*4),"chords":sorted(set(cids))}
                                 for t,cids in sorted(attacks.items())]
                    status="resolved" if not unresolved else "unresolved"

                    # Full-measure auditable overlay.
                    x0=max(0,int(round(left-il))); x1=min(int(W),int(round(right+il)))
                    y0=max(0,int(round(top-2.2*il))); y1=min(int(H),int(round(bottom+2.2*il)))
                    crop=np.where(binary[y0:y1,x0:x1],0,255).astype(np.uint8)
                    canvas=Image.new("RGB",(x1-x0,y1-y0+70),"white")
                    canvas.paste(Image.fromarray(crop,mode="L").convert("RGB"),(0,70))
                    draw=ImageDraw.Draw(canvas)
                    draw.text((8,8),f"m{gm} visual attack extractor: {status}; timing slots NOT used",fill=(0,0,0))
                    draw.text((8,30),"green=attack  blue=same chord  yellow=tied continuation  red=unresolved voice",fill=(0,0,0))
                    draw.text((8,50),f"attacks={len(attack_rows)}  unresolved_voices={len(unresolved)}",fill=(0,0,0))
                    radius=max(5,int(round(.30*il))); width=max(2,int(round(.11*il)))
                    unresolved_ids={e["cid"] for vr in unresolved for e in vr["seq"]}
                    for row in note_rows:
                        hbs=[(h,get_box(by.get(h))) for h in row["heads"]]
                        hbs=[(h,b) for h,b in hbs if b is not None]
                        attack_h=[(h,b) for h,b in hbs if h not in tied_right]
                        rep=attack_h[0][0] if attack_h else None
                        for h,b in hbs:
                            if row["cid"] in unresolved_ids:
                                color=(220,0,0)
                            elif h in tied_right:
                                color=(240,175,0)
                            elif h==rep:
                                color=(0,170,0)
                            else:
                                color=(0,105,230)
                            cx=b.cx-x0; cy=b.cy-y0+70
                            draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=color,width=width)
                    canvas.save(OUT/f"measure_{gm:02d}.png")

                    all_measures[str(gm)]={
                        "page":page_index+1,"system":sy+1,"meter_whole":str(meter),
                        "status":status,"attack_count":len(attack_rows),"attacks":attack_rows,
                        "voices":[{
                            "part":vr["part"],"voice":vr["voice"],"span":str(vr["span"]),
                            "start":flabel(vr["start"]),"reason":vr["reason"],
                            "items":[{"cid":e["cid"],"rel":str(e["rel"]),"duration":flabel(e["duration"]),
                                      "rest":e["rest"],"attack":e["attack"]} for e in vr["seq"]]
                        } for vr in voice_recs],
                        "unresolved_voice_count":len(unresolved)
                    }
                page_base += len(stacks)
            page_measure_base=page_base

    summary={
        "measure_count":len(all_measures),
        "resolved_measures":[int(m) for m,r in all_measures.items() if r["status"]=="resolved"],
        "unresolved_measures":[int(m) for m,r in all_measures.items() if r["status"]!="resolved"],
        "measures":all_measures,
    }
    (OUT/"full_score_visual_attacks.json").write_text(json.dumps(summary,indent=2))
    print("FULL_VISUAL_SUMMARY="+json.dumps({
        "measure_count":summary["measure_count"],
        "resolved_count":len(summary["resolved_measures"]),
        "unresolved_count":len(summary["unresolved_measures"]),
        "unresolved_measures":summary["unresolved_measures"]
    },separators=(",",":")))
    for m in (7,28,34,39):
        r=all_measures.get(str(m))
        print(f"TARGET m{m}: "+json.dumps({
            "status":r["status"],"count":r["attack_count"],
            "q":[x["onset_quarter"] for x in r["attacks"]],
            "unresolved_voices":r["unresolved_voice_count"]
        },separators=(",",":")))

if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("usage: visual_attack_extractor.py SCORE.omr")
    main(sys.argv[1])
