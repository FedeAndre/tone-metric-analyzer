#!/usr/bin/env python3
from __future__ import annotations
import io, json, math, re, zipfile
from fractions import Fraction
from pathlib import Path
import numpy as np, pandas as pd, requests
from scipy import signal, stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tone_metric.engine import analyze
from tone_metric.models import Hit, MeasureInfo
from tone_metric.waves import build_wave_profile
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile
from tone_metric.theory import boundary_sequence,boundary_sequence_recurrence,pascal_binomial_mod,lucas_binomial_mod

SEED=20260925
RNG=np.random.default_rng(SEED)
ZIP_URL="https://physionet.org/content/tremordb/get-zip/1.0.0/"
OUT=Path("research/tma_tremordb_exact_results"); OUT.mkdir(parents=True,exist_ok=True)
FS=100.0; N_CYCLES=64; PROJ_TOL_S=0.005
EVENT_TYPES=("UP0","POSPEAK","DOWN0","NEGPEAK")
PRIMARY=("ren","ref","ron","rof")
AGE={"g1":54,"g2":52,"v3":71,"v4":67,"v5":40,"s6":61,"s7":59,"s8":64,
     "g9":68,"g10":59,"g11":57,"g12":54,"g13":50,"s14":57,"s15":40,"s16":37}

def load_zip():
    r=requests.get(ZIP_URL,timeout=300,headers={"User-Agent":"Mozilla/5.0 TMA-research"}); r.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(r.content))

def parse_units(z):
    n=[x for x in z.namelist() if x.endswith("/file_description.txt")][0]
    txt=z.read(n).decode(errors="replace")
    out={}
    # Data rows contain subject, filename, range, units, laser, rate, samples.
    for line in txt.splitlines():
        m=re.match(r"^\s*([gsv]\d+)\s+([gsv]\d+r(?:e[fn]|o[fn]|(?:15|30|45|60)of)\.(?:let|rit))\s+\S+\s+(mm/s|m/s)\s+",line)
        if m: out[m.group(2)]=m.group(3)
    return out

def list_records(z,units):
    rows=[]
    for n in z.namelist():
        base=Path(n).name
        if not re.match(r"^[gsv]\d+r(?:e[fn]|o[fn]|(?:15|30|45|60)of)\.(?:let|rit)$",base):
            continue
        parent=Path(n).parent.name
        amp=parent[-1].upper() if parent and parent[-1] in "hl" else None
        cond=parent[:-1] if amp else None
        subj=re.match(r"^([gsv]\d+)",base).group(1)
        side=base.split(".")[-1]
        target={"g":"GPi","v":"Vim","s":"STN"}[subj[0]]
        rows.append({"path":n,"file":base,"subject":subj,"condition":cond,"amp_group":amp,
                     "side":side,"target":target,"age":AGE.get(subj,np.nan),"unit":units.get(base)})
    return pd.DataFrame(rows)

def bandpass(x):
    x=np.asarray(x,float)
    x=signal.detrend(x,type="linear")
    sos=signal.butter(4,[2,12],btype="bandpass",fs=FS,output="sos")
    return signal.sosfiltfilt(sos,x)

def crossing_times(y):
    up=np.flatnonzero((y[:-1]<=0)&(y[1:]>0))+1
    down=np.flatnonzero((y[:-1]>=0)&(y[1:]<0))+1
    return up,down

def project(sample,start,end):
    if sample==start:return Fraction(0),0,0.0,0.0
    raw=Fraction(2*(sample-start),end-start); rf=float(raw); cyc=(end-start)/FS
    for depth in range(21):
        den=2**depth; k=int(math.floor(rf*den+.5)); q=Fraction(k,den)
        err=abs(float(q-raw)*cyc/2)
        if err<=PROJ_TOL_S+1e-12:return q,depth,1000*err,rf
    return raw,-1,0.0,rf

def extract_cycles(raw):
    y=bandpass(raw)
    up,down=crossing_times(y)
    # discard first/last 2 s
    up=up[(up>=200)&(up<len(y)-200)]
    cycles=[]
    for a,b in zip(up[:-1],up[1:]):
        dur=(b-a)/FS
        if not (0.10<=dur<=0.50): continue
        di=np.searchsorted(down,a+1)
        if di>=len(down) or down[di]>=b: continue
        d=int(down[di])
        if d-a<2 or b-d<2: continue
        p=a+int(np.argmax(y[a:d]))
        n=d+int(np.argmin(y[d:b]))
        if not (a<p<d<n<b): continue
        events=[]
        for lab,s in (("UP0",a),("POSPEAK",p),("DOWN0",d),("NEGPEAK",n)):
            q,dep,err,rf=project(s,a,b)
            events.append({"label":lab,"sample":int(s),"raw_phase":rf,
                           "phase_num":q.numerator,"phase_den":q.denominator,
                           "projection_depth":dep,"projection_error_ms":err})
        cycles.append({"start_sample":int(a),"end_sample":int(b),"duration_s":dur,"events":events})
    # Select a contiguous 64-cycle block centered in the recording.
    if len(cycles)<N_CYCLES: raise RuntimeError(f"only {len(cycles)} valid cycles")
    k=(len(cycles)-N_CYCLES)//2
    return cycles[k:k+N_CYCLES],y

def build_hits(cycles):
    hits=[];labs=[];cis=[];measures=[]
    for i,c in enumerate(cycles):
        start=Fraction(2*i)
        measures.append(MeasureInfo(index=i,number=str(i+1),start=start,full_duration=Fraction(2),
            actual_duration=Fraction(2),pickup_shift=Fraction(0),numerator=2,denominator=4,
            implicit=False,opening_anacrusis=False))
        for e in c["events"]:
            q=Fraction(e["phase_num"],e["phase_den"])
            hits.append(Hit(onset=start+q,duration=Fraction(0),measure_index=i,measure_number=str(i+1),
                offset_in_measure=q,sources=[],canonical_recovered=False))
            labs.append(e["label"]);cis.append(i)
    order=sorted(range(len(hits)),key=lambda j:(hits[j].onset,EVENT_TYPES.index(labs[j])))
    return [hits[j] for j in order],measures,[labs[j] for j in order],[cis[j] for j in order]

def analyze_cycles(cycles):
    hits,measures,labs,cis=build_hits(cycles)
    res=analyze(hits,measures)
    er=[e for seg in res["segments"] for e in seg["events"]]
    if len(er)!=len(labs): raise RuntimeError("engine event count mismatch")
    ev=pd.DataFrame({"label":labs,"cycle_index":cis,
        "H":[int(r["tone_metric_height"]) for r in er],
        "D":[int(r["tone_metric_density"]) for r in er],
        "lambda":[int(r["lowest_tone_metric_level"]) for r in er],
        "levels":[";".join(map(str,r["tone_metric_levels"])) for r in er]})
    wave=build_wave_profile(res); piv=build_pivot_profile(wave); tree=build_tree_profile(wave); br=tree.get("branches",[])
    f={"mean_H":float(ev.H.mean()),"mean_D":float(ev.D.mean()),"mean_lambda":float(ev["lambda"].mean()),
       "multilevel":float((ev.D>1).mean()),"pivot_rate":float(len(piv)/len(ev)),
       "compound_pivot_fraction":float(np.mean([p["compound"] for p in piv])) if piv else np.nan,
       "pivot_depth":float(np.mean([p["drop_depth"] for p in piv])) if piv else np.nan,
       "tree_span_events":float(np.mean([int(b["target_node_index"])-int(b["source_node_index"]) for b in br])) if br else np.nan,
       "tree_span_time":float(np.mean([float(Fraction(str(b["distance_quarter"]))) for b in br])) if br else np.nan,
       "tree_root_fraction":float(np.mean([bool(n["root"]) for n in tree.get("nodes",[])])) if tree.get("nodes") else np.nan}
    for lab in EVENT_TYPES:
        q=ev[ev.label==lab]
        f[f"{lab}_mean_H"]=float(q.H.mean());f[f"{lab}_mean_D"]=float(q.D.mean())
        f[f"{lab}_mean_lambda"]=float(q["lambda"].mean());f[f"{lab}_multilevel"]=float((q.D>1).mean())
    dep=[e["projection_depth"] for c in cycles for e in c["events"][1:]]
    err=[e["projection_error_ms"] for c in cycles for e in c["events"][1:]]
    f["projection_depth_mean"]=float(np.mean(dep));f["projection_error_ms_mean"]=float(np.mean(err));f["projection_error_ms_max"]=float(np.max(err))
    return f,ev,piv,tree

def conventional(raw,unit,cycles):
    scale=1000.0 if unit=="m/s" else 1.0
    xmm=np.asarray(raw,float)*scale
    x=signal.detrend(xmm)
    f,p=signal.welch(x,fs=FS,nperseg=min(1024,len(x)))
    mask=(f>=2)&(f<=12); fp=f[mask]; pp=p[mask]
    dom=float(fp[np.argmax(pp)]) if len(fp) else np.nan
    pn=pp/pp.sum() if pp.sum()>0 else np.ones_like(pp)/len(pp)
    ent=float(-np.sum(pn*np.log(pn+1e-15))/np.log(len(pn))) if len(pn)>1 else np.nan
    dur=np.array([c["duration_s"] for c in cycles])
    return {"rms_mm_s":float(np.sqrt(np.mean(xmm**2))),"dominant_freq_hz":dom,
            "spectral_entropy_2_12":ent,"cycle_period_mean":float(dur.mean()),
            "cycle_period_cv":float(dur.std(ddof=1)/dur.mean())}

def dz(diff):
    d=np.asarray(diff,float)
    return float(np.mean(d)/np.std(d,ddof=1)) if len(d)>1 and np.std(d,ddof=1)>0 else np.nan

def paired_table(df,a,b,features):
    A=df[df.condition==a].set_index("subject");B=df[df.condition==b].set_index("subject")
    subs=sorted(set(A.index)&set(B.index)); rows=[]
    for feat in features:
        x=A.loc[subs,feat].to_numpy(float);y=B.loc[subs,feat].to_numpy(float);d=x-y
        if np.allclose(d,0): p=1.0
        else: p=float(stats.ttest_rel(x,y).pvalue)
        try: wp=float(stats.wilcoxon(d,zero_method="wilcox",alternative="two-sided").pvalue)
        except: wp=np.nan
        rows.append({"contrast":f"{a}-minus-{b}","feature":feat,"n_pairs":len(subs),
                     f"{a}_mean":float(np.mean(x)),f"{b}_mean":float(np.mean(y)),
                     "mean_difference":float(np.mean(d)),"paired_t_p":p,"wilcoxon_p":wp,"dz":dz(d)})
    return pd.DataFrame(rows)

def bh(p):
    p=np.asarray(p,float);n=len(p);o=np.argsort(p);q=np.empty(n);last=1.
    for rank,idx in reversed(list(enumerate(o,start=1))):
        last=min(last,p[idx]*n/rank);q[idx]=last
    return q

def loso_auc(df,features,a="ref",b="rof"):
    z=df[df.condition.isin([a,b])].dropna(subset=features).copy()
    z["y"]=(z.condition==a).astype(int)
    pred=np.full(len(z),np.nan)
    for sub in z.subject.unique():
        te=z.subject==sub;tr=~te
        if z.loc[tr,"y"].nunique()<2:continue
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000))
        m.fit(z.loc[tr,features],z.loc[tr,"y"])
        pred[te]=m.predict_proba(z.loc[te,features])[:,1]
    ok=np.isfinite(pred)
    return float(roc_auc_score(z.loc[ok,"y"],pred[ok])),int(ok.sum())

def main():
    z=load_zip(); units=parse_units(z); recs=list_records(z,units)
    audit={"boundary_match":boundary_sequence(2,12)==boundary_sequence_recurrence(2,12),"lucas_pascal_mismatches":0}
    for n in range(64):
        for k in range(n+1):
            audit["lucas_pascal_mismatches"]+=pascal_binomial_mod(n,k,2)!=lucas_binomial_mod(n,k,2)
    rows=[]; failures={}; worked={}
    for i,r in recs.iterrows():
        raw=np.fromstring(z.read(r.path).decode(errors="replace"),sep="\n")
        try:
            cycles,y=extract_cycles(raw); feat,ev,piv,tree=analyze_cycles(cycles); conv=conventional(raw,r.unit,cycles)
            rows.append({**r.to_dict(),**conv,**feat})
            if r.file in ("g2rof.rit","g2ref.rit","g9rof.rit","g9ref.rit"):
                worked[r.file]={"first_cycle":cycles[0],"features":feat,"conventional":conv,
                                "first_events":ev.head(12).to_dict(orient="records")}
        except Exception as e: failures[r.file]=repr(e)
    df=pd.DataFrame(rows);df.to_csv(OUT/"record_features.csv",index=False)
    (OUT/"worked_examples.json").write_text(json.dumps(worked,indent=2,default=str))
    (OUT/"paper_math_audit.json").write_text(json.dumps(audit,indent=2))

    diagnostics={"projection_depth_mean","projection_error_ms_mean","projection_error_ms_max"}
    meta={"path","file","subject","condition","amp_group","side","target","age","unit",
          "rms_mm_s","dominant_freq_hz","spectral_entropy_2_12","cycle_period_mean","cycle_period_cv"}|diagnostics
    tma=[c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    # remove constants/nonfinite
    tma=[c for c in tma if df[c].notna().all() and np.nanstd(df[c])>1e-10]

    contrasts=[]
    for a,b in [("ref","rof"),("ren","ron"),("ron","rof"),("ren","ref")]:
        q=paired_table(df,a,b,tma)
        if len(q):
            q["q_BH"]=bh(q.paired_t_p.values);contrasts.append(q)
    cdf=pd.concat(contrasts,ignore_index=True) if contrasts else pd.DataFrame()
    cdf.to_csv(OUT/"paired_tma_contrasts.csv",index=False)

    # Conventional treatment effects as sanity/context.
    conventional_feats=["rms_mm_s","dominant_freq_hz","spectral_entropy_2_12","cycle_period_mean","cycle_period_cv"]
    cc=[]
    for a,b in [("ref","rof"),("ren","ron"),("ron","rof"),("ren","ref")]:
        cc.append(paired_table(df,a,b,conventional_feats))
    convdf=pd.concat(cc,ignore_index=True);convdf.to_csv(OUT/"paired_conventional_contrasts.csv",index=False)

    # HAT vs LAT treatment-response differences: difference score ref-rof.
    resp=[]
    A=df[df.condition=="ref"].set_index("subject");B=df[df.condition=="rof"].set_index("subject")
    subs=sorted(set(A.index)&set(B.index))
    for feat in tma:
        tmp=[]
        for s in subs:
            tmp.append({"subject":s,"amp_group":A.loc[s,"amp_group"],"delta":float(A.loc[s,feat]-B.loc[s,feat])})
        q=pd.DataFrame(tmp);h=q[q.amp_group=="H"].delta.values;l=q[q.amp_group=="L"].delta.values
        if len(h)>=2 and len(l)>=2:
            tt=stats.ttest_ind(h,l,equal_var=False)
            resp.append({"feature":feat,"n_H":len(h),"n_L":len(l),"H_delta":float(np.mean(h)),"L_delta":float(np.mean(l)),"welch_p":float(tt.pvalue)})
    respdf=pd.DataFrame(resp)
    if len(respdf): respdf["q_BH"]=bh(respdf.welch_p.values);respdf=respdf.sort_values("welch_p")
    respdf.to_csv(OUT/"dbs_response_hat_vs_lat.csv",index=False)

    # 15-60 min washout: within-subject centered correlation with time.
    long=df[df.condition.isin(["r15of","r30of","r45of","r60of"])].copy()
    time_map={"r15of":15,"r30of":30,"r45of":45,"r60of":60};long["minutes"]=long.condition.map(time_map)
    wash=[]
    for feat in tma+conventional_feats:
        q=long[["subject","minutes",feat]].dropna()
        if len(q)<12:continue
        q["x"]=q.minutes-q.groupby("subject").minutes.transform("mean")
        q["y"]=q[feat]-q.groupby("subject")[feat].transform("mean")
        if np.std(q.x)>0 and np.std(q.y)>0:
            rr=stats.pearsonr(q.x,q.y)
            wash.append({"feature":feat,"n_obs":len(q),"n_subjects":q.subject.nunique(),"within_subject_r":float(rr.statistic),"p":float(rr.pvalue)})
    washdf=pd.DataFrame(wash)
    if len(washdf): washdf["q_BH"]=bh(washdf.p.values);washdf=washdf.sort_values("p")
    washdf.to_csv(OUT/"dbs_washout_trends.csv",index=False)

    # Does TMA improve discrimination of DBS-on vs DBS-off, both medication off?
    conv=["rms_mm_s","dominant_freq_hz","spectral_entropy_2_12","cycle_period_mean","cycle_period_cv"]
    compact=[x for x in ["mean_H","mean_D","mean_lambda","multilevel","pivot_rate","compound_pivot_fraction",
                          "pivot_depth","tree_span_events","tree_span_time","POSPEAK_mean_D","NEGPEAK_mean_D"] if x in df]
    auc0,n0=loso_auc(df,conv);auc1,n1=loso_auc(df,conv+compact)
    inc={"contrast":"ref(DBS on, med off) vs rof(DBS off, med off)","conventional":conv,"tma":compact,
         "n_predictions_base":n0,"n_predictions_plus_tma":n1,"auc_base":auc0,"auc_plus_tma":auc1,"delta_auc":auc1-auc0}
    (OUT/"incremental_classification.json").write_text(json.dumps(inc,indent=2))

    primary=cdf[cdf.contrast=="ref-minus-rof"].sort_values("paired_t_p") if len(cdf) else pd.DataFrame()
    summary={"randomly_selected_dataset":"Effect of Deep Brain Stimulation on Parkinsonian Tremor",
             "paper_math_audit":audit,"n_records_analyzed":len(df),"n_failures":len(failures),"failures":failures,
             "records_by_condition":df.groupby("condition").size().to_dict(),
             "primary_dbs_off_med_off_pairs":int(len(set(df[df.condition=="ref"].subject)&set(df[df.condition=="rof"].subject))),
             "top_primary_tma":primary.head(12).to_dict(orient="records"),
             "top_washout":washdf.head(12).to_dict(orient="records"),
             "incremental":inc,
             "max_projection_error_ms":float(df.projection_error_ms_max.max())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__": main()
