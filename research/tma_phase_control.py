#!/usr/bin/env python3
"""Test whether cycle-order-expected TMA Fourier features convey a unique
individual/clinical profile beyond basic foot-contact phase distributions.

Original 86 Ga gait trial files: 64 gait cycles each; features computed without
TMA to benchmark already archived TMA features. Clinical outcomes are patient-
level from same archive. Independent trial _02 tested against subjects excluded
from fitting on _01. No tuning on outcomes; only two fixed ridge alpha values.
"""
import json,time,warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import tma_gait_gauge_validation as gait

warnings.filterwarnings("ignore",category=RuntimeWarning)
OUT=Path("research/tma_individual_fingerprint_results")
OUT.mkdir(exist_ok=True,parents=True)
SEED=20261008
BANDS=["16to32_strides","8to16_strides","4to8_strides","2to4_strides"]
TMA=[f"null_{m}_{b}" for m in ("H","D") for b in BANDS]
TMARES=[f"tma_{m}_{b}" for m in ("H","D") for b in BANDS]
BASE=["Age","Gender"]
CONV=["conv_"+x for x in ("stride_mean","stride_cv","phase_sd","phase_acf","stance_mean","phase_mean")]+[
f"conv_{p}_{band}" for p in ("stride","RTO","RHS","LTO") for band in BANDS]
Q=["RTO","RHS","LTO"]
def features(record):
    arr=gait.download_record(record)
    c,fs,integrity=gait.extract_cycles(arr,record)
    phase=np.array([[float(e["raw_phase"])/2 for e in x["events"][1:]] for x in c])
    # phase proportion of each cycle, not absolute seconds
    dur=np.array([x["duration_s"] for x in c],float)
    segments=np.diff(np.concatenate([np.zeros((len(phase),1)),phase,np.ones((len(phase),1))],axis=1),axis=1)
    o={"record":record}
    for j,n in enumerate(Q):
        x=phase[:,j]
        o[f"ph_{n}_mean"]=float(x.mean())
        o[f"ph_{n}_sd"]=float(x.std(ddof=1))
        for p in (.1,.5,.9):
            o[f"ph_{n}_q{int(p*100)}"]=float(np.quantile(x,p))
        bins=np.histogram(x,bins=np.linspace(0,1,9))[0]/len(x)
        for i,v in enumerate(bins):o[f"hist_{n}_{i}"]=float(v)
        # per-event metrical occupancy on dyadic 1/8 grid.
        for depth in (2,3,4):
            o[f"phase_{n}_near_{depth}"]=float(np.mean(np.abs(x*2**depth-np.round(x*2**depth))<.09))
    for j,n in enumerate(("LHS_to_RTO","RTO_to_RHS","RHS_to_LTO","LTO_to_LHS")):
        o[f"gap_{n}_mean"]=float(segments[:,j].mean())
        o[f"gap_{n}_sd"]=float(segments[:,j].std(ddof=1))
    for a,b in ((0,1),(0,2),(1,2)):
        o[f"phase_crosscov_{a}{b}"]=float(np.cov(phase[:,a],phase[:,b])[0,1])
    o["stride_duration_mean"]=float(dur.mean())
    o["stride_duration_cv"]=float(dur.std(ddof=1)/dur.mean())
    o["LHS_stance_seconds"]=float(np.mean(segments[:,0]*dur))
    return o
def matrix_pair(df,rep,participants):
    first=df[(df.trial==1)&df.subject.isin(participants)].set_index("subject").loc[participants]
    other=df[(df.trial==2)&df.subject.isin(participants)].set_index("subject").loc[participants]
    features=SETS[rep]
    a=first[features].to_numpy(float);b=other[features].to_numpy(float)
    mu=a.mean(axis=0);sd=np.maximum(a.std(axis=0),1e-8)
    a=(a-mu)/sd;b=(b-mu)/sd
    dist=np.sqrt(np.mean((b[:,None,:]-a[None,:,:])**2,axis=2))
    return dist
def identity(df,rep,participants):
    D=matrix_pair(df,rep,participants)
    ranks=[int(np.sum(D[i,:]<D[i,i])+1) for i in range(len(participants))]
    rng=np.random.default_rng(SEED)
    sim=np.array([float(np.mean(D[np.arange(len(participants)),rng.permutation(len(participants))])) for _ in range(9999)])
    obs=float(np.mean(np.diag(D)))
    return {"n":len(participants),"top1":int(sum(x==1 for x in ranks)),
      "top1_rate":float(np.mean(np.array(ranks)==1)),
      "median_rank":float(np.median(ranks)),
      "within_distance":obs,"between_distance":float(np.mean(D[~np.eye(len(D),dtype=bool)])),
      "permutation_p":float((1+sum(sim<=obs))/(1+len(sim)))}
def fit_predict(tr,te,cols,y,alpha):
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    m.fit(tr[cols],tr[y].to_numpy(float))
    return float(m.predict(te[cols])[0])
def evaluate(df,target,rep,alpha,trial):
    candidates=df[(df.group=="pd")&(df.trial==trial)&np.isfinite(df[target])].copy()
    train=df[(df.group=="pd")&(df.trial==1)&np.isfinite(df[target])].copy()
    score=[];ref=[];names=[]
    for _,te in candidates.iterrows():
        training=train[train.subject!=te.subject]
        assert te.subject not in training.subject.values
        p=fit_predict(training,pd.DataFrame([te]),SETS[rep],target,alpha)
        score.append(p);ref.append(float(te[target]));names.append(te.subject)
    a=np.asarray(ref);b=np.asarray(score)
    return {"n":len(ref),"mae":float(np.mean(np.abs(a-b))),
      "r2":float(1-np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)),
      "pred":score,"true":ref,"subjects":names}
def compare(modelA,modelB):
    assert modelA["subjects"]==modelB["subjects"]
    lossA=np.abs(np.array(modelA["pred"])-np.array(modelA["true"]))
    lossB=np.abs(np.array(modelB["pred"])-np.array(modelB["true"]))
    gains=lossA-lossB
    rng=np.random.default_rng(SEED+333)
    sample=np.array([float(np.mean(gains[rng.integers(0,len(gains),len(gains))])) for _ in range(4999)])
    return {"mean_MAE_improvement":float(np.mean(gains)),
            "boot95":np.quantile(sample,[.025,.975]).tolist(),
            "bootstrap_fraction_improved":float(np.mean(sample>0))}
def expected_feature_reconstruction(df,features,alpha,testing_trial):
    # Predict 8 expected TMA bands from footprint phase descriptors; subjects
    # missing in _02 are excluded from out-of-trial testing.
    train=df[df.trial==1]
    target=df[df.trial==testing_trial]
    X=target[features].to_numpy(float)
    Y=target[TMA].to_numpy(float)
    pred=[]
    for _,r in target.iterrows():
        tr=train[train.subject!=r.subject]
        mdl=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        mdl.fit(tr[features],tr[TMA])
        pred.append(mdl.predict(pd.DataFrame([r])[features])[0])
    pred=np.asarray(pred)
    sd=np.maximum(Y.std(axis=0),1e-8)
    mse=np.mean((Y-pred)**2,axis=0)
    r2=1-mse/(sd**2)
    return {"n":len(Y),"mean_R2":float(np.mean(r2)),
      "median_R2":float(np.median(r2)),
      "by_TMA_band_R2":dict(zip(TMA,[float(x) for x in r2]))}
def main():
    t=time.monotonic()
    old=pd.read_csv(OUT/"subject_spectral_fingerprints.csv")
    assert len(old)==86
    names=old.record.tolist()
    feats=[];errors=[]
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures={executor.submit(features,name):name for name in names}
        for i,f in enumerate(as_completed(futures),1):
            try:feats.append(f.result())
            except Exception as e:errors.append({"record":futures[f],"error":str(e)})
            if i%10==0:print("Raw walking file",i,"/",len(names),flush=True)
    if errors:raise RuntimeError(f"Missing clinical gait phases: {errors}")
    f=pd.DataFrame(feats)
    df=old.merge(f,on="record",validate="one_to_one")
    df.to_csv(OUT/"phase_control_features.csv",index=False)
    pdids=sorted(set(df[(df.group=="pd")&(df.trial==1)].subject)&set(df[(df.group=="pd")&(df.trial==2)].subject))
    assert len(pdids)==25
    global SETS
    CORE=["stride_duration_mean","stride_duration_cv"]+[f"ph_{n}_{stat}" for n in Q for stat in ("mean","sd")]
    RICH=CORE+[f"ph_{n}_q{p}" for n in Q for p in (10,50,90)]+[
        f"hist_{n}_{i}" for n in Q for i in range(8)]+[
        f"phase_{n}_near_{i}" for n in Q for i in (2,3,4)]+[
        f"phase_crosscov_{a}{b}" for a,b in ((0,1),(0,2),(1,2))]
    GAP=[f"gap_{n}_{stat}" for n in ("LHS_to_RTO","RTO_to_RHS","RHS_to_LTO","LTO_to_LHS") for stat in ("mean","sd")]
    RICH+=GAP
    SETS={"expected":TMA,"order_residual":TMARES,
      "basic_phases":CORE,"rich_phases":RICH,
      "conventional":BASE+CONV,
      "basic_phases_plus_expected":CORE+TMA,
      "rich_phases_plus_expected":RICH+TMA,
      "conventional_plus_expected":BASE+CONV+TMA,
      "basic_clinical":BASE+CORE,
      "rich_clinical":BASE+RICH,
      "basic_clinical_plus_expected":BASE+CORE+TMA,
      "rich_clinical_plus_expected":BASE+RICH+TMA}
    identification={k:identity(df,k,pdids) for k in ("expected","order_residual","basic_phases","rich_phases","conventional","basic_phases_plus_expected","rich_phases_plus_expected")}
    models={}
    targets=("UPDRSM","UPDRS","HoehnYahr","TUAG","Speed_01")
    reps=("expected","basic_clinical","rich_clinical","conventional",
          "basic_clinical_plus_expected","rich_clinical_plus_expected","conventional_plus_expected")
    for target in targets:
        models[target]={}
        for alpha in (20,80):
            models[target][str(alpha)]={}
            for trial in (1,2):
                predictions={p:evaluate(df,target,p,alpha,trial) for p in reps}
                base="basic_clinical";rich="rich_clinical";conv="conventional"
                modelvals={p:{k:v for k,v in z.items() if k in ("n","mae","r2")} for p,z in predictions.items()}
                contrasts={f"{p}_vs_{p}_plus_expected":compare(predictions[p],predictions[p+"_plus_expected"])
                           for p in (base,rich,conv)}
                models[target][str(alpha)][f"trial_{trial}"]={"models":modelvals,"comparisons":contrasts}
                print("Models",target,alpha,"trial",trial,flush=True)
    reconstruction={}
    for alpha in (20,80,300):
        reconstruction[str(alpha)]={}
        for trial in (1,2):
            reconstruction[str(alpha)][f"trial_{trial}"]={
              "core":expected_feature_reconstruction(df,CORE,alpha,trial),
              "rich":expected_feature_reconstruction(df,RICH,alpha,trial)}
    result={"n_subjects":df.subject.nunique(),"n_pd":29,"n_recordings":len(df),"n_PD_pairs":len(pdids),
            "methods":"8-band shuffled-cycle-expected spectra versus phase histograms and phase quantiles from original gait waveforms",
            "sets":{k:v for k,v in SETS.items()},
            "individual_identification":identification,
            "clinical_models":models,
            "reconstruct_expected_from_phases":reconstruction,
            "seconds":time.monotonic()-t,
            "limitations":["Case-control archival data, no prospective severity trajectory",
             "_02 walking-task and speed equivalence not documented as definitely identical to _01",
             "Data contain 64 normalized walking cycles per trial",
             "Models exploratory; five severity/performance endpoints and several feature sets, two ridge alphas",
             "Expected TMA null uses 39 shuffles first walk and 19 shuffles second walk",
             "High-dimensional rich phase histograms potentially overfit n=29 despite penalization"]}
    (OUT/"phase_control_results.json").write_text(json.dumps(result,indent=2))
    print("RESULT_SUMMARY",json.dumps({"identity":identification,
            "reconstruction":reconstruction,
            "clinical":{tar:{a:{tr:{
                "scores":z["models"],
                "clinical_rich_gain":z["comparisons"]["rich_clinical_vs_rich_clinical_plus_expected"]}
                for tr,z in trials.items()} for a,trials in levels.items()} for tar,levels in models.items()}}),flush=True)
if __name__=="__main__":main()
