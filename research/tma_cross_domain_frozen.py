#!/usr/bin/env python3
from __future__ import annotations

import io, json, math, os, tarfile, time, zipfile
from fractions import Fraction
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy import signal, stats
from scipy.io import loadmat

from tone_metric.engine import recursive_integer_structure, _next_seq_anchor
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile

SEED=20260924
RNG=np.random.default_rng(SEED)
OUT=Path("research/tma_cross_domain_frozen_results")
OUT.mkdir(parents=True, exist_ok=True)
CACHE=Path("/tmp/tma_cross_domain_cache"); CACHE.mkdir(parents=True,exist_ok=True)

F=8
K=3
GRID=256
N_PER_DOMAIN=20
NULL_B=1500
PERM_B=2000
MAX_LEVEL_FEATURE=12

FEATURES=[
    "mean_H","sd_H","mean_D","sd_D","mean_lambda","sd_lambda","multilevel",
    "pivot_rate","compound_pivot_fraction","pivot_depth",
    "tree_span_events","tree_span_time","tree_root_fraction","tree_ascending_fraction",
]+[f"occ_L{i}" for i in range(1,MAX_LEVEL_FEATURE+1)]

LOG=[]

def log(s):
    s=str(s); print(s,flush=True); LOG.append(s)

def download(url,path,retries=6,timeout=600):
    path=Path(path)
    if path.exists() and path.stat().st_size>1000:
        return path
    last=None
    for a in range(retries):
        try:
            with requests.get(url,stream=True,timeout=timeout,headers={"User-Agent":"Mozilla/5.0 TMA-research"},allow_redirects=True) as r:
                if r.status_code==429:
                    raise RuntimeError("HTTP 429")
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk: f.write(chunk)
            if path.stat().st_size<=1000: raise RuntimeError("implausibly short")
            return path
        except Exception as e:
            last=e
            try:path.unlink()
            except FileNotFoundError:pass
            time.sleep(min(5*(a+1),30))
    raise RuntimeError(f"download failed {url}: {last}")

def quantize_phases(ph):
    x=np.asarray(ph,float)
    x=x[np.isfinite(x)]
    if len(x)<K: raise ValueError("need >=3 phases")
    x=np.clip(np.mod(x,1.0),1/GRID,(GRID-1)/GRID)
    if len(x)>K:
        # preserve the full-frame geometry using deterministic order-statistic sampling
        ids=np.round(np.linspace(0,len(x)-1,K)).astype(int)
        x=np.sort(x)[ids]
    bins=np.clip(np.rint(x*GRID).astype(int),1,GRID-1)
    # Resolve quantization collisions deterministically to nearest free bin.
    used=set(); out=[]
    for b in sorted(bins):
        if b not in used:
            q=b
        else:
            q=None
            for d in range(1,GRID):
                for c in (b-d,b+d):
                    if 1<=c<=GRID-1 and c not in used:
                        q=c;break
                if q is not None:break
        used.add(q);out.append(Fraction(int(q),GRID))
    if len(out)!=K or len(set(out))!=K: raise RuntimeError("phase quantization failure")
    return sorted(out)

def topk_event_phases(score,k=K,min_sep=1):
    score=np.asarray(score,float)
    if len(score)<k: raise ValueError("too short")
    order=np.argsort(np.nan_to_num(score,nan=-np.inf))[::-1]
    chosen=[]
    for idx in order:
        if all(abs(int(idx)-j)>=min_sep for j in chosen):
            chosen.append(int(idx))
            if len(chosen)==k: break
    if len(chosen)<k:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
                if len(chosen)==k:break
    chosen=np.sort(chosen)
    # centers of samples, avoiding exact frame endpoints
    return (chosen+0.5)/len(score)

def analyze_frames(frames):
    if len(frames)!=F: raise ValueError(f"need {F} frames")
    qframes=[quantize_phases(x) for x in frames]
    rows=[]
    for fi,phs in enumerate(qframes):
        for q in phs:
            rows.append((Fraction(fi)+q,fi,q))
    rows.sort(key=lambda z:z[0])
    unique=sorted({t for t,_,_ in rows})
    actual=set(unique)
    levels={t:set() for t in unique}

    max_pos=F+1
    top_end=_next_seq_anchor(2,max_pos)
    point_layers={}; interval_levels={}
    recursive_integer_structure(1,top_end,2,1,point_layers,interval_levels)
    point_layers={p:ls for p,ls in point_layers.items() if p<=max_pos}
    interval_levels={p:l for p,l in interval_levels.items() if p<max_pos}

    for pos,ls in point_layers.items():
        t=Fraction(pos-1)
        if t in actual:
            levels[t].update(int(x) for x in ls)

    by_interval=defaultdict(list)
    for t in unique:
        k=int(math.floor(float(t)))
        if 0<=k<F: by_interval[k].append(t)

    def refine(a,b,ev,parent,guard=0):
        if guard>20: raise RuntimeError("recursive guard")
        interior=[t for t in ev if a<t<b]
        if not interior:return
        m=(a+b)/2; lvl=int(parent)+1
        for t in (a,m,b):
            if t in actual: levels[t].add(lvl)
        left=[t for t in ev if a<=t<=m]
        right=[t for t in ev if m<=t<=b]
        if any(a<t<m for t in left): refine(a,m,left,lvl,guard+1)
        if any(m<t<b for t in right): refine(m,b,right,lvl,guard+1)

    for i in range(F):
        a,b=Fraction(i),Fraction(i+1);ev=by_interval.get(i,[])
        if any(a<t<b for t in ev):
            refine(a,b,ev,int(interval_levels.get(i+1,1)))

    evrows=[];wave=[]
    for onset,fi,q in rows:
        Kset=sorted(int(x) for x in levels[onset] if int(x)>0)
        if not Kset: raise RuntimeError(f"unresolved event {onset}")
        evrows.append((max(Kset),len(Kset),min(Kset),Kset))
        wave.append({
            "segment_index":0,"measure_index":fi,"measure_number":str(fi+1),
            "onset_quarter":str(onset),"offset_in_measure_quarter":str(q),
            "levels":Kset,"height":max(Kset),"density":len(Kset),
            "lowest_level":min(Kset),"attack":True,"parenthetical":False,
        })
    H=np.array([r[0] for r in evrows],float)
    D=np.array([r[1] for r in evrows],float)
    L=np.array([r[2] for r in evrows],float)
    piv=build_pivot_profile(wave)
    tree=build_tree_profile(wave); br=tree.get("branches",[])
    feat={
        "mean_H":H.mean(),"sd_H":H.std(ddof=1),"mean_D":D.mean(),"sd_D":D.std(ddof=1),
        "mean_lambda":L.mean(),"sd_lambda":L.std(ddof=1),"multilevel":np.mean(D>1),
        "pivot_rate":len(piv)/len(wave),
        "compound_pivot_fraction":float(np.mean([bool(p.get("compound")) for p in piv])) if piv else 0.0,
        "pivot_depth":float(np.mean([float(p.get("drop_depth",0)) for p in piv])) if piv else 0.0,
        "tree_span_events":float(np.mean([int(b["target_node_index"])-int(b["source_node_index"]) for b in br])) if br else 0.0,
        "tree_span_time":float(np.mean([float(Fraction(str(b["distance_quarter"]))) for b in br])) if br else 0.0,
        "tree_root_fraction":float(np.mean([bool(n.get("root")) for n in tree.get("nodes",[])])) if tree.get("nodes") else 0.0,
        "tree_ascending_fraction":float(np.mean([bool(b.get("ascending_level")) for b in br])) if br else 0.0,
    }
    for lv in range(1,MAX_LEVEL_FEATURE+1):
        feat[f"occ_L{lv}"]=float(np.mean([lv in r[3] for r in evrows]))
    return feat

def mkrow(domain,entity,view,frames,meta=None):
    f=analyze_frames(frames)
    return {"domain":domain,"entity":str(entity),"view":view,**f,**(meta or {})}

# ---------- adapters ----------
def gait_rows():
    import sys
    sys.path.insert(0,str(Path("research").resolve()))
    import tma_gait_gauge_validation as g
    recs=[*[(x,"control") for x in g.CONTROL_IDS[:10]],*[(x,"pd") for x in g.PD_IDS[:10]]]
    out=[]
    for rec,grp in recs:
        arr=g.download_record(rec);cy,fs,_=g.extract_cycles(arr,rec)
        eligible=[]
        for c in cy:
            ph=[float(Fraction(e["phase_num"],e["phase_den"]))/2.0 for e in c["events"] if e["label"]!="LHS"]
            if len(ph)==3:eligible.append(ph)
        if len(eligible)>=16:
            out += [mkrow("gait",rec,"A",eligible[:8],{"subgroup":grp}),mkrow("gait",rec,"B",eligible[8:16],{"subgroup":grp})]
    log(f"gait pairs={len(out)//2}")
    return out

def cardio_rows():
    import sys
    sys.path.insert(0,str(Path("research").resolve()))
    import tma_physio_fantasia as p
    recs=p.YOUNG[:10]+p.OLD[:10];out=[]
    for j,rec in enumerate(recs,1):
        raw=p.load_subject(rec)
        _,_,phases=p.orientation_features(raw,"peak")
        q=[np.asarray(x,float) for x in phases if len(x)>=3]
        if len(q)>=16:
            out += [mkrow("cardiorespiratory",rec,"A",q[:8]),mkrow("cardiorespiratory",rec,"B",q[8:16])]
        log(f"cardio {j}/{len(recs)}")
    log(f"cardiorespiratory pairs={len(out)//2}")
    return out

def neural_rows():
    import sys
    sys.path.insert(0,str(Path("research").resolve()))
    import tma_neural_theta_spike as n
    tmp=Path("/tmp/tma_neural_cross.nwb"); candidates=[]
    per_sub=defaultdict(list)
    # Frozen balanced subset: process sessions only until each animal contributes
    # five eligible 16-cycle windows. This preserves the prespecified 20-entity
    # design while avoiding unnecessary multi-GB downloads after quota is filled.
    for idx,(sub,ses,url) in enumerate(n.ASSETS,1):
        if len(per_sub[sub])>=5:
            continue
        try:
            _,surr=n.process_session(sub,ses,url,tmp)
            for rid,ss,phase_lists in surr:
                q=[np.asarray(x,float) for x in phase_lists if len(x)>=3]
                if len(q)>=16 and len(per_sub[ss])<5:
                    per_sub[ss].append((rid,q))
        except Exception as e:
            log(f"neural skip {ses}: {e!r}")
        if all(len(per_sub[x])>=5 for x in ("M01","M02","M03","M05")):
            break
    for sub in ("M01","M02","M03","M05"):
        candidates.extend([(sub,rid,q) for rid,q in per_sub[sub][:5]])
    out=[]
    for sub,rid,q in candidates[:N_PER_DOMAIN]:
        ent=f"{sub}:{rid}"
        out += [mkrow("neural_theta_spike",ent,"A",q[:8],{"subject":sub}),
                mkrow("neural_theta_spike",ent,"B",q[8:16],{"subject":sub})]
    log(f"neural pairs={len(out)//2}")
    return out

def oasbud_frames(rf,roi):
    rf=np.asarray(rf,float);roi=np.asarray(roi)>0
    eligible=np.flatnonzero(np.sum(roi,axis=0)>=20)
    if len(eligible)<F:return None
    cols=eligible[np.round(np.linspace(0,len(eligible)-1,F)).astype(int)]
    frames=[]
    for c in cols:
        ids=np.flatnonzero(roi[:,c])
        if len(ids)<20:return None
        lo,hi=int(ids.min()),int(ids.max())
        y=np.abs(signal.hilbert(rf[:,c]))
        seg=y[lo:hi+1].copy()
        mask=roi[lo:hi+1,c]
        seg[~mask]=-np.inf
        minsep=max(2,len(seg)//30)
        ph=topk_event_phases(seg,K,minsep)
        # phases are within lesion ROI bounding interval
        frames.append(ph)
    return frames

def oasbud_rows():
    p=download("https://zenodo.org/record/545928/files/OASBUD.mat?download=1",CACHE/"OASBUD.mat")
    m=loadmat(p,squeeze_me=True,struct_as_record=False); data=m["data"]
    out=[]
    for i,row in enumerate(data):
        if len(out)//2>=N_PER_DOMAIN:break
        try:
            a=oasbud_frames(row.rf1,row.roi1);b=oasbud_frames(row.rf2,row.roi2)
            if a is None or b is None:continue
            out += [mkrow("ultrasound",i,"A",a,{"class":int(row.__dict__["class"])}),
                    mkrow("ultrasound",i,"B",b,{"class":int(row.__dict__["class"])})]
        except Exception as e: log(f"OASBUD skip {i}: {e!r}")
    log(f"ultrasound pairs={len(out)//2}")
    return out

def cmapss_sensor_frames(arr,sensor_cols):
    x=np.asarray(arr,float); n=len(x)
    b=max(10,int(round(.2*n)))
    base=x[:b,sensor_cols]
    mu=np.mean(base,axis=0);sd=np.std(base,axis=0,ddof=1)
    global_sd=np.std(x[:,sensor_cols],axis=0,ddof=1)
    keep=(sd>1e-8)&(global_sd>1e-8)
    if np.sum(keep)<2:return None
    z=(x[:,np.asarray(sensor_cols)[keep]]-mu[keep])/sd[keep]
    score=np.sqrt(np.mean(z*z,axis=1))
    parts=np.array_split(np.arange(n),F)
    frames=[]
    for ids in parts:
        if len(ids)<K:return None
        s=score[ids]
        ph=topk_event_phases(s,K,max(1,len(ids)//10))
        frames.append(ph)
    return frames

def cmapss_rows():
    p=download("https://data.nasa.gov/docs/legacy/CMAPSSData.zip",CACHE/"CMAPSSData.zip")
    with zipfile.ZipFile(p) as z:
        txt=z.read("train_FD001.txt")
    arr=np.loadtxt(io.BytesIO(txt))
    out=[]
    sensors=list(range(5,26))
    va=sensors[::2];vb=sensors[1::2]
    for unit in range(1,101):
        if len(out)//2>=N_PER_DOMAIN:break
        x=arr[arr[:,0]==unit]
        a=cmapss_sensor_frames(x,va);b=cmapss_sensor_frames(x,vb)
        if a is None or b is None:continue
        out += [mkrow("cmapss",unit,"A",a),mkrow("cmapss",unit,"B",b)]
    log(f"cmapss pairs={len(out)//2}")
    return out

def computing_rows():
    out=[]
    # Representative transition from three nested divider stages in each
    # 8-clock frame. Different entities use cyclic phase translations; A/B
    # are separated blocks of the same deterministic counter.
    base=np.array([1/8,2/8,4/8],float)
    for ent in range(N_PER_DOMAIN):
        shift=(ent%8)/64.0
        frames=[]
        for fi in range(16):
            # preserve nested spacings while changing absolute counter phase
            frames.append(np.mod(base+shift,1.0))
        out += [mkrow("binary_counter",ent,"A",frames[:8]),mkrow("binary_counter",ent,"B",frames[8:])]
    log(f"binary_counter pairs={len(out)//2}")
    return out

def hcp_extract_archive(path):
    root=CACHE/"hcp"; root.mkdir(exist_ok=True)
    with tarfile.open(path,"r:gz") as tf:
        members=tf.getmembers()
        reg=[m for m in members if m.name.endswith("/regions.npy") or m.name=="regions.npy"]
        sl=[m for m in members if m.name.endswith("/subjects_list.txt") or m.name=="subjects_list.txt"]
        if not reg or not sl:
            raise RuntimeError("HCP regions.npy or subjects_list.txt not found")
        raw=tf.extractfile(sl[0]).read().decode("utf-8","replace")
        subject_ids=[x.strip() for x in raw.split() if x.strip()][:N_PER_DOMAIN]
        needed=[reg[0],sl[0]]
        for m in members:
            n="/"+m.name.replace("\\","/")
            if any(f"/subjects/{sid}/MOTOR/" in n for sid in subject_ids):
                if n.endswith("/data.npy") or "/EVs/" in n:
                    needed.append(m)
        tf.extractall(root,members=needed)
    regions=list(root.rglob("regions.npy"))
    if not regions: raise RuntimeError("HCP regions.npy not found after extract")
    base=regions[0].parent
    return base,subject_ids

def select_signal_peaks(y):
    y=np.asarray(y,float)
    y=signal.detrend(y)
    if len(y)<K:return None
    return topk_event_phases(y,K,max(1,len(y)//6))

def hcp_rows():
    url="https://files.de-1.osf.io/v1/resources/54w3g/providers/osfstorage/60e80c2bf80fdb01334d9147"
    p=download(url,CACHE/"hcp_task.tgz",retries=8,timeout=1200)
    root,subject_ids=hcp_extract_archive(p)
    regions=np.load(root/"regions.npy",allow_pickle=True).T
    nets=np.asarray(regions[1]).astype(str)
    sm=np.array(["somatomotor" in x.lower() for x in nets])
    if np.sum(sm)<5:
        raise RuntimeError(f"Somatomotor parcels not found; networks={np.unique(nets)}")
    out=[]
    for sid in subject_ids:
        subject=root/"subjects"/str(sid)/"MOTOR"
        views=[]
        for key in ("LR","RL"):
            run=subject/f"tfMRI_MOTOR_{key}"
            tsfile=run/"data.npy"
            evdir=run/"EVs"
            if not tsfile.exists() or not evdir.exists():
                views=[];break
            ts=np.load(tsfile)
            if ts.ndim!=2:
                views=[];break
            y=np.mean(ts[sm],axis=0) if ts.shape[0]==len(sm) else np.mean(ts[:,sm],axis=1)
            blocks=[]
            for cond in ["lf","rf","lh","rh","t"]:
                ev=evdir/f"{cond}.txt"
                if not ev.exists():continue
                a=np.loadtxt(ev,ndmin=2)
                for row in a:
                    onset,dur=float(row[0]),float(row[1])
                    start=int(math.floor((onset+4.0)/.72))
                    n=max(K,int(math.ceil(dur/.72)))
                    seg=y[start:min(len(y),start+n)]
                    ph=select_signal_peaks(seg)
                    if ph is not None: blocks.append((onset,ph))
            blocks=sorted(blocks,key=lambda z:z[0])
            if len(blocks)<F:
                views=[];break
            views.append([b[1] for b in blocks[:F]])
        if len(views)==2:
            out += [mkrow("fmri",sid,"A",views[0]),mkrow("fmri",sid,"B",views[1])]
    log(f"fmri pairs={len(out)//2}")
    return out

# ---------- common comparator ----------
def null_reference():
    vals=[]
    for b in range(NULL_B):
        frames=[]
        for _ in range(F):
            bins=RNG.choice(np.arange(1,GRID),size=K,replace=False)
            frames.append(np.sort(bins)/GRID)
        vals.append(analyze_frames(frames))
        if (b+1)%250==0:log(f"null {b+1}/{NULL_B}")
    d=pd.DataFrame(vals)[FEATURES]
    mu=d.mean();sd=d.std(ddof=1)
    keep=[c for c in FEATURES if np.isfinite(sd[c]) and sd[c]>1e-8]
    return d,mu,sd,keep

def rmsdist(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    return float(np.sqrt(np.mean(d*d)))

def analyze_comparator(df,null_df,mu,sd,keep):
    z=df.copy()
    for c in keep:z[c]=(z[c]-mu[c])/sd[c]
    # exact universal null z-scores: no fitting to real domains.
    z.to_csv(OUT/"sample_signatures_z.csv",index=False)
    df.to_csv(OUT/"sample_features_raw.csv",index=False)
    null_df.to_csv(OUT/"universal_null_features.csv",index=False)

    domains=sorted(z.domain.unique())
    pairs=[]
    for dom in domains:
        a=z[(z.domain==dom)&(z.view=="A")].sort_values("entity")
        b=z[(z.domain==dom)&(z.view=="B")].sort_values("entity")
        common=sorted(set(a.entity)&set(b.entity))
        a=a.set_index("entity").loc[common];b=b.set_index("entity").loc[common]
        A=a[keep].to_numpy();B=b[keep].to_numpy()
        same=np.array([rmsdist(A[i],B[i]) for i in range(len(common))])
        diff=[]
        for i in range(len(common)):
            for j in range(len(common)):
                if i!=j:diff.append(rmsdist(A[i],B[j]))
        diff=np.asarray(diff)
        ratio=float(np.mean(same)/np.mean(diff)) if len(diff) else np.nan
        # permutation: random pairing of B within domain
        rng=np.random.default_rng(SEED+sum(map(ord,dom)))
        pvals=[]
        for _ in range(PERM_B):
            perm=rng.permutation(len(common))
            pvals.append(np.mean([rmsdist(A[i],B[perm[i]]) for i in range(len(common))]))
        p=float((1+np.sum(np.asarray(pvals)<=np.mean(same)))/(PERM_B+1))
        # top-1 identity
        D=np.array([[rmsdist(A[i],B[j]) for j in range(len(common))] for i in range(len(common))])
        top1=float(np.mean(np.argmin(D,axis=1)==np.arange(len(common))))
        pairs.append({"domain":dom,"n":len(common),"same_mean":same.mean(),"different_mean":diff.mean(),
                      "same_over_different":ratio,"paired_permutation_p":p,"top1_identity":top1,
                      "chance_identity":1/len(common)})
    pairdf=pd.DataFrame(pairs);pairdf.to_csv(OUT/"paired_repeat_validation.csv",index=False)

    # Domain centroids = average of A and B entity signatures.
    entity=z.groupby(["domain","entity"],as_index=False)[keep].mean(numeric_only=True)
    cent=entity.groupby("domain")[keep].mean()
    mat=pd.DataFrame(index=domains,columns=domains,dtype=float)
    for d1 in domains:
        for d2 in domains:mat.loc[d1,d2]=rmsdist(cent.loc[d1,keep],cent.loc[d2,keep])
    mat.to_csv(OUT/"domain_distance_matrix.csv")

    # Leave-one-entity-out nearest-centroid domain classification.
    correct=[];predrows=[]
    for idx,row in entity.iterrows():
        train=entity.drop(index=idx)
        cents=train.groupby("domain")[keep].mean()
        ds={d:rmsdist(row[keep].to_numpy(float),cents.loc[d,keep].to_numpy(float)) for d in cents.index}
        pred=min(ds,key=ds.get)
        correct.append(pred==row.domain)
        predrows.append({"domain":row.domain,"entity":row.entity,"predicted_domain":pred,
                         "correct":pred==row.domain,"nearest_distance":ds[pred]})
    acc=float(np.mean(correct));pd.DataFrame(predrows).to_csv(OUT/"domain_predictions.csv",index=False)

    # Entity-label permutation test for domain accuracy.
    rng=np.random.default_rng(SEED+999); labels=entity.domain.to_numpy().copy(); X=entity[keep].to_numpy(float)
    uniq=domains
    nullacc=[]
    for b in range(PERM_B):
        lab=rng.permutation(labels);ok=0
        for i in range(len(entity)):
            tr=np.arange(len(entity))!=i
            best=None;bestd=np.inf
            for d in uniq:
                ids=np.flatnonzero(tr&(lab==d))
                if len(ids)==0:continue
                c=np.mean(X[ids],axis=0);dist=rmsdist(X[i],c)
                if dist<bestd:bestd=dist;best=d
            ok += (best==lab[i])
        nullacc.append(ok/len(entity))
    pacc=float((1+np.sum(np.asarray(nullacc)>=acc))/(PERM_B+1))

    # Are domain centroids farther from universal random-event null than expected?
    nullz=(null_df[keep]-mu[keep])/sd[keep]
    null_centroid_tests=[]
    for dom in domains:
        rows=entity[entity.domain==dom][keep]
        n=len(rows);obs=rmsdist(rows.mean().to_numpy(float),np.zeros(len(keep)))
        vals=[]
        for _ in range(PERM_B):
            ids=RNG.integers(0,len(nullz),size=n)
            vals.append(rmsdist(nullz.iloc[ids].mean().to_numpy(float),np.zeros(len(keep))))
        pn=float((1+np.sum(np.asarray(vals)>=obs))/(PERM_B+1))
        null_centroid_tests.append({"domain":dom,"n":n,"centroid_norm":obs,"null_tail_p":pn})
    nulldf=pd.DataFrame(null_centroid_tests);nulldf.to_csv(OUT/"domain_vs_universal_null.csv",index=False)

    # Stability: bootstrap domain centroids and count nearest-domain relationships.
    boot=defaultdict(lambda:defaultdict(int))
    for b in range(1000):
        bc={}
        for dom in domains:
            q=entity[entity.domain==dom][keep]
            ids=RNG.integers(0,len(q),size=len(q))
            bc[dom]=q.iloc[ids].mean().to_numpy(float)
        for dom in domains:
            others={d:rmsdist(bc[dom],bc[d]) for d in domains if d!=dom}
            boot[dom][min(others,key=others.get)]+=1
    stab=[]
    for dom in domains:
        for other,count in sorted(boot[dom].items(),key=lambda kv:-kv[1]):
            stab.append({"domain":dom,"nearest_domain":other,"bootstrap_fraction":count/1000})
    pd.DataFrame(stab).to_csv(OUT/"nearest_domain_bootstrap.csv",index=False)

    summary={
        "seed":SEED,"frozen_design":{"frames_per_sample":F,"events_per_frame":K,"dyadic_grid":GRID,
            "arity":2,"features":keep,"universal_null_samples":NULL_B},
        "domains":domains,"n_entities_by_domain":entity.groupby("domain").size().to_dict(),
        "domain_classification":{"leave_one_entity_out_accuracy":acc,"chance":1/len(domains),
                                 "permutation_p":pacc,"B":PERM_B},
        "paired_repeat":pairdf.to_dict(orient="records"),
        "domain_vs_null":nulldf.to_dict(orient="records"),
        "distance_matrix":mat.to_dict(),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

def main():
    t=time.time()
    allrows=[]
    adapters=[
        ("gait",gait_rows),
        ("cardiorespiratory",cardio_rows),
        ("neural_theta_spike",neural_rows),
        ("ultrasound",oasbud_rows),
        ("cmapss",cmapss_rows),
        ("binary_counter",computing_rows),
        ("fmri",hcp_rows),
    ]
    failures={}
    for name,fn in adapters:
        try:
            log(f"=== {name} ===")
            rows=fn()
            if len(rows)<10: raise RuntimeError(f"only {len(rows)//2} pairs")
            allrows.extend(rows)
        except Exception as e:
            failures[name]=repr(e);log(f"DOMAIN FAILURE {name}: {e!r}")
    if len(set(r["domain"] for r in allrows))<4:
        raise RuntimeError("fewer than four domains completed")
    log("=== universal null ===")
    null_df,mu,sd,keep=null_reference()
    df=pd.DataFrame(allrows)
    summary=analyze_comparator(df,null_df,mu,sd,keep)
    summary["failures"]=failures;summary["runtime_s"]=time.time()-t
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
    log(json.dumps(summary,indent=2))
    log(f"runtime_s={time.time()-t:.1f}")

if __name__=="__main__":
    main()
