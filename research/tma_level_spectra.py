#!/usr/bin/env python3
"""Frozen-engine Tone-Metric Level x frequency matrix:
check whether level-specific Fourier oscillations add real physiology beyond
exact gait event phases, phase histograms, stride-time spectra, and the TMA
expected fractal baseline. Exploratory, predetermined 2-band summaries.
"""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import tma_gait_gauge_validation as g
from tma_gait_fast_exact import fast_level_sets,assert_equivalent

OUT=Path("research/tma_level_spectral_results")
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261008
N_SHUFFLE=19
BANDS={"slow_8to32":(1/32,1/8),"fast_2to8":(1/8,1/2+.001)}
LEVEL_CLASSES={"L1to4":(1,4),"L5to8":(5,8),"L9plus":(9,100)}
CORE=["Age","Gender","stride_duration_mean","stride_duration_cv"]
CORE += [f"ph_{lab}_{x}" for lab in ("RTO","RHS","LTO") for x in ("mean","sd")]
FULL=["Age","Gender"]+[f"conv_{x}" for x in ("stride_mean","stride_cv","phase_sd","phase_acf","stance_mean","phase_mean")]
FULL += [f"conv_{lab}_{freq}" for lab in ("stride","RTO","RHS","LTO") for freq in
 ("16to32_strides","8to16_strides","4to8_strides","2to4_strides")]
EXPECTED=[f"null_{v}_{f}" for v in ("H","D") for f in
 ("16to32_strides","8to16_strides","4to8_strides","2to4_strides")]
PREDICTORS=[f"{lev}_{f}" for lev in LEVEL_CLASSES for f in BANDS]
PREDICTORS += [f"phasegap_{f}" for f in BANDS]
# samples may have deterministic Level 1-4 with little variance - assess explicitly
def describe(c):
    rows,levels=fast_level_sets(c)
    matrix=np.zeros((3,len(c)))
    for onset,j,q,label in rows:
        s=levels[onset]
        for k,(lo,hi) in enumerate(LEVEL_CLASSES.values()):
            matrix[k,j] += sum(lo<=lv<=hi for lv in s)/4.
    return matrix
def signals(c):
    mat=describe(c)
    freqs=np.fft.rfftfreq(len(c),1)
    ffts=np.fft.rfft(mat-mat.mean(axis=1,keepdims=True),axis=1)
    # avoid artificial spectra from long unbounded trends: detrend same as raw
    detr=np.vstack([x-np.polyval(np.polyfit(np.arange(len(x)),x,1),np.arange(len(x)))
                       for x in mat])
    ffts=np.fft.rfft(detr,axis=1)
    power=np.abs(ffts)**2
    spec={}
    for i,lv in enumerate(LEVEL_CLASSES):
        sig=power[i]
        avg=float(np.mean(sig[1:]))
        for band,(lo,hi) in BANDS.items():
            mask=(freqs>=lo-1e-12)&(freqs<hi-1e-12)
            if band=="fast_2to8":mask=(freqs>=lo-1e-12)&(freqs<=.5+1e-12)
            spec[f"{lv}_{band}"]=float(np.log((sig[mask].mean()+1e-8)/(avg+1e-8)))
    for band,(lo,hi) in BANDS.items():
        mask=(freqs>=lo-1e-12)&(freqs<hi-1e-12)
        if band=="fast_2to8":mask=(freqs>=lo-1e-12)&(freqs<=.5+1e-12)
        v=ffts[0,mask]*np.conjugate(ffts[2,mask])
        # weighted mean phase cosine, signal-power normalized to avoid very
        # weak phases giving false apparent coherence
        num=np.sum(v)
        denom=np.sqrt(np.sum(abs(ffts[0,mask])**2)*np.sum(abs(ffts[2,mask])**2))
        spec[f"phasegap_{band}"]=float(np.real(num/(denom+1e-8)))
    return spec,mat
def load(item):
    name,idx=item
    arr=g.download_record(name)
    cycles,*_=g.extract_cycles(arr,name)
    if idx in (0,27,61):
        for n in (16,32,64):assert_equivalent(cycles[:n],name+":"+str(n))
    original,mat=signals(cycles)
    rng=np.random.default_rng(SEED+idx*1931)
    null=[signals(g.cycle_order_scramble(cycles,rng))[0] for _ in range(N_SHUFFLE)]
    out={"record":name}
    for f in PREDICTORS:
        expected=float(np.mean([v[f] for v in null]))
        out["level_full_"+f]=original[f]
        out["level_order_"+f]=original[f]-expected
        out["level_expected_"+f]=expected
    for j,n in enumerate(LEVEL_CLASSES):
        out["level_variance_"+n]=float(np.var(mat[j]))
        out["level_occupancy_"+n]=float(mat[j].mean())
    return out
def idtest(df,group,mode):
    first=df[(df.group==group)&(df.trial==1)].set_index("subject")
    second=df[(df.group==group)&(df.trial==2)].set_index("subject")
    ids=sorted(set(first.index)&set(second.index));n=len(ids)
    if n<3:return{}
    X=first.loc[ids,mode].to_numpy(float);Y=second.loc[ids,mode].to_numpy(float)
    mu=X.mean(axis=0);sd=np.maximum(X.std(axis=0),1e-7)
    X=(X-mu)/sd;Y=(Y-mu)/sd
    D=np.sqrt(np.mean((Y[:,None,:]-X[None,:,:])**2,axis=2))
    ranked=np.argmin(D,axis=1)
    rng=np.random.default_rng(SEED+221)
    score=float(np.mean(np.diag(D)))
    null=np.array([np.mean(D[np.arange(n),rng.permutation(n)]) for i in range(4999)])
    return {"n":n,"top1_count":int(np.sum(ranked==np.arange(n))),"chance":1/n,
      "within_distance":score,"different_distance":float(np.mean(D[~np.eye(n,dtype=bool)])),
      "permutation_p":float((1+sum(null<=score))/(1+len(null)))}
def predict(df,cols,outcome,alpha,trial):
    training=df[(df.group=="pd")&(df.trial==1)&np.isfinite(df[outcome])]
    testing=df[(df.group=="pd")&(df.trial==trial)&np.isfinite(df[outcome])]
    Y=[];P=[]
    for _,r in testing.iterrows():
        tr=training[training.subject!=r.subject]
        mdl=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        mdl.fit(tr[cols],tr[outcome])
        p=mdl.predict(pd.DataFrame([r])[cols])[0]
        Y.append(float(r[outcome]));P.append(float(p))
    a=np.asarray(Y);b=np.asarray(P)
    return {"n":len(Y),"MAE":float(np.mean(abs(a-b))),
      "R2":float(1-sum((a-b)**2)/sum((a-a.mean())**2)),"errors":list(abs(a-b))}
def main():
    tick=time.monotonic()
    df=pd.read_csv("research/tma_individual_fingerprint_results/phase_control_features.csv")
    assert len(df)==86 and df.subject.nunique()==47
    with ThreadPoolExecutor(max_workers=10) as ex:
        jobs={ex.submit(load,(name,i)):name for i,name in enumerate(df.record)}
        results=[]
        for j,f in enumerate(as_completed(jobs),1):
            results.append(f.result())
            if j%10==0:print("Level spectra",j,"of",len(jobs),flush=True)
    augmented=df.merge(pd.DataFrame(results),on="record",validate="one_to_one")
    augmented.to_csv(OUT/"features.csv",index=False)
    reps={
      "basic_phases":CORE,
      "traditional":FULL,
      "traditional_plus_expected":FULL+EXPECTED,
      "expected_TMA":EXPECTED,
      "level_original":[f"level_full_{k}" for k in PREDICTORS],
      "level_order":[f"level_order_{k}" for k in PREDICTORS],
      "level_expected":[f"level_expected_{k}" for k in PREDICTORS],
      "basic_plus_level":CORE+[f"level_full_{k}" for k in PREDICTORS],
      "traditional_plus_level":FULL+[f"level_full_{k}" for k in PREDICTORS],
      "traditional_plus_level_order":FULL+[f"level_order_{k}" for k in PREDICTORS],
      "traditional_plus_level_expected":FULL+[f"level_expected_{k}" for k in PREDICTORS],
      "traditional_plus_TMAexpected_level":FULL+EXPECTED+[f"level_full_{k}" for k in PREDICTORS]
    }
    ident={group:{n:idtest(augmented,group,cols) for n,cols in reps.items()}
            for group in ("pd","control")}
    scores={}
    for outcome in ("UPDRSM","UPDRS","HoehnYahr","TUAG","Speed_01"):
        scores[outcome]={}
        for trial in (1,2):
            scores[outcome][f"trial{trial}"]={}
            for alpha in (20,80):
                vals={name:predict(augmented,cols,outcome,alpha,trial) for name,cols in reps.items()}
                scores[outcome][f"trial{trial}"][str(alpha)]={
                    n:{k:v for k,v in d.items() if k!="errors"} for n,d in vals.items()}
            print("clinical",outcome,trial,flush=True)
    # pre-specified 8-component Level-by-frequency null deviation
    discr=[]
    for col in [f"level_order_{k}" for k in PREDICTORS]:
        # unadjusted Spearman with timed up-and-go; FDR across 8 is needed.
        sample=augmented[(augmented.group=="pd")&(augmented.trial==1)&np.isfinite(augmented.TUAG)]
        rr,p=stats.spearmanr(sample[col],sample.TUAG)
        discr.append({"feature":col,"rho":float(rr),"p":float(p)})
    out={"dataset":"PhysioNet Ga gait 86 recordings, 47 people; 29 PD, 18 HC",
         "n_subj":47,"n_walks":86,"n_PD_repeats":25,"nnull":N_SHUFFLE,
         "n_level_spectral_features":len(PREDICTORS),"feature_keys":PREDICTORS,
         "per_level_occupancy_summary":{
           v:{"mean":float(augmented["level_occupancy_"+v].mean()),
              "mean_variance":float(augmented["level_variance_"+v].mean())}
           for v in LEVEL_CLASSES},
         "individual_identity":ident,"clinical_predictions":scores,
         "individual_band_correlations":discr,"runtime_s":time.monotonic()-tick,
         "limitations":["Per-stride frequency description of 4 foot contacts and occupied fractal Levels",
            "Static 64-stride windows and three coarse Level strata; no full Level by Level continuous Fourier transform",
            "Motor-severity models tiny, exploratory, multiple outcomes and alpha settings",
            "Repeat-trial conditions not proved strictly identical",
            "First-walk data have 39 expected nulls in preexisting features while new Level nulls use 19",
            "Exact model engine tested on selected first few participants and dates"],
         }
    (OUT/"results.json").write_text(json.dumps(out,indent=2))
    print("RESULT",json.dumps({"identity":ident,"occupancy":out["per_level_occupancy_summary"],
         "predictions":scores,"corr":discr},indent=2),flush=True)
if __name__=="__main__":main()
