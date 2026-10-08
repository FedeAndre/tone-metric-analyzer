#!/usr/bin/env python3
"""Exploratory Level-specific Fourier description of richly variable drum rhythms.
Frozen TMA Level placements, preserved origin, 97 performances from Groove MIDI.

Independent, withheld drum timbre/orchestration changes; leave-drummer-out models
against conventional phrase timing/onset power and full wave H/D FFT. Crucially
the full onset FFT benchmark controls original event periodicities.
"""
import json,io,time
from pathlib import Path
from fractions import Fraction
import numpy as np,pandas as pd,mido
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import tma_variable_drums as base
from scipy import stats

OUT=Path("research/tma_level_music_results")
OUT.mkdir(parents=True,exist_ok=True)
WINDOW=16
GRID=base.GRID
N=WINDOW*GRID
FREQ=np.fft.rfftfreq(N,1/GRID)
LEVELS={"shallow":(1,4),"middle":(5,8),"deep":(9,100)}
BANDS={"very_slow":(.0625,.25),"slow":(.25,.5),"medium":(.5,2),"fast":(2,8)}
KEYS=[f"{lev}_{key}" for lev in LEVELS for key in BANDS]
KEYS+=[f"crossphase_{str(f).replace('.','p')}" for f in (.25,.5,1,2)]
KEYS += ["deep_occupancy_fraction","level_cross_coherence","levelfreq_entropy"]
def one_window(times,lev,w):
    selected=[t for t in times if 16*w<=float(t)<16*(w+1)]
    M=np.zeros((3,N),float)
    for t in selected:
        idx=int((t-Fraction(w*16))*GRID)
        for j,(name,(lo,hi)) in enumerate(LEVELS.items()):
            M[j,idx]=sum(lo<=k<=hi for k in lev[t])
    spectrum=np.fft.rfft(M-M.mean(axis=1,keepdims=True),axis=1)
    power=np.abs(spectrum)**2
    vals={}
    for j,name in enumerate(LEVELS):
        ps=power[j]
        denom=float(np.sum(ps[1:]))+1e-12
        for band,(l,h) in BANDS.items():
            mask=(FREQ>=l-1e-9)&(FREQ<h-1e-9)
            vals[f"{name}_{band}"]=float(np.sum(ps[mask])/denom)
    low=spectrum[0];hi=spectrum[2]
    for f in (.25,.5,1,2):
        k=int(np.argmin(np.abs(FREQ-f)))
        z=low[k]*hi[k].conjugate()
        vals[f"crossphase_{str(f).replace('.','p')}"]=float(np.real(z)/(abs(z)+1e-9))
    vals["deep_occupancy_fraction"]=float(np.mean([any(k>=9 for k in lev[t]) for t in selected]))
    num=np.abs(np.sum(low[1:]*hi[1:].conjugate()))**2
    denom=np.sum(abs(low[1:])**2)*np.sum(abs(hi[1:])**2)+1e-9
    vals["level_cross_coherence"]=float(np.clip(num/denom,0,1))
    p=power/(power.sum()+1e-12)
    v=p.ravel()
    vals["levelfreq_entropy"]=float(-np.sum(v*np.log(v+1e-12))/np.log(len(v)))
    return vals
def model(df,cols,target,alpha):
    y=df[target].to_numpy(float);pred=np.zeros(len(y))
    for drummer in df.drummer.unique():
        test=(df.drummer==drummer).to_numpy()
        mdl=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        mdl.fit(df.loc[~test,cols],y[~test])
        pred[test]=mdl.predict(df.loc[test,cols])
    return pred
def measure(df,actual,pred):
    y=df[actual].to_numpy(float)
    def m(sel):
        q=np.ones(len(df),bool) if sel=="all" else (df.selection==sel).to_numpy()
        return {"n":int(q.sum()),"MAE":float(np.mean(abs(y[q]-pred[q]))),
            "R2":float(1-np.sum((y[q]-pred[q])**2)/np.sum((y[q]-np.mean(y[q]))**2))}
    return {s:m(s) for s in ("all","regular","variable")}
def boost(df,target,baseline,added):
    y=df[target].to_numpy()
    delta=abs(y-baseline)-abs(y-added)
    ids=sorted(df.drummer.unique())
    index={id:np.where(df.drummer==id)[0] for id in ids}
    rng=np.random.default_rng(20261008)
    sampled=[]
    for _ in range(2999):
        ix=np.concatenate([index[id] for id in rng.choice(ids,len(ids),replace=True)])
        sampled.append(float(np.mean(delta[ix])))
    return {"gain":float(np.mean(delta)),"cluster_boot95":np.quantile(sampled,[.025,.975]).tolist()}
def main():
    t=time.monotonic()
    df=pd.read_csv("research/tma_fourier_results/window_features.csv")
    assert len(df)==679 and df.track.nunique()==97
    archive,meta,_=base.load_data()
    lookup={row["id"]:row for row in meta}
    out=[]
    for i,rec in enumerate(sorted(df.track.unique())):
        item=lookup[rec]
        members=[n for n in archive.namelist() if n.endswith(item["midi_filename"])]
        assert len(members)==1
        midi=mido.MidiFile(file=io.BytesIO(archive.read(members[0])))
        notes=[]
        for track in midi.tracks:
            tick=0
            for m in track:
                tick+=m.time
                if m.type=="note_on" and m.velocity>0:
                    notes.append((tick/midi.ticks_per_beat,int(m.note),int(m.velocity)))
        prepared=base.prepare(notes);assert prepared is not None
        times=prepared[0]
        lv=base.levels_full(times,32)
        if i<3:base.check_engine(times,32)
        for win in range(1,8):
            values=one_window(times,lv,win)
            out.append({"track":rec,"window":win,**{"LS_"+k:values[k] for k in KEYS}})
        if (i+1)%15==0:print("Music level",i+1,"of 97",flush=True)
    f=pd.DataFrame(out)
    df=df.merge(f,on=["track","window"],validate="one_to_one")
    assert len(df)==679
    df.to_csv(OUT/"music_level_windows.csv",index=False)
    cols=[f"LS_{key}" for key in KEYS]
    baseline=["window","bpm"]+[x for x in df.columns if x.startswith("timing_")]+[
      x for x in df.columns if x.startswith("onset_") or x.startswith("ioi_ordinal_")]
    fullfft=[x for x in df.columns if x.startswith("H_") or x.startswith("D_")]
    original_time=[x for x in df.columns if x.startswith("origin_")]
    models={
      "ordinary_timing_and_FFT":baseline,
      "ordinary_plus_H_D_spectra":baseline+fullfft,
      "ordinary_plus_Level_spectra":baseline+cols,
      "ordinary_plus_H_D_and_Level":baseline+fullfft+cols,
      "ordinary_plus_time_TMA":baseline+original_time,
      "ordinary_plus_time_TMA_and_Level":baseline+original_time+cols,
      "Level_only":["window","bpm"]+cols}
    assert all(len(m)==len(set(m)) for m in models.values())
    scores={}
    for outcome in ("change","drift"):
        scores[outcome]={}
        for alpha in (100,300,1000):
            pred={k:model(df,c,outcome,alpha) for k,c in models.items()}
            scores[outcome][str(alpha)]={
              "metrics":{k:measure(df,outcome,p) for k,p in pred.items()},
              "gains":{
                "Level_plus_ordinary":boost(df,outcome,pred["ordinary_timing_and_FFT"],pred["ordinary_plus_Level_spectra"]),
                "Level_beyond_H_D":boost(df,outcome,pred["ordinary_plus_H_D_spectra"],pred["ordinary_plus_H_D_and_Level"]),
                "Level_beyond_time_TMA":boost(df,outcome,pred["ordinary_plus_time_TMA"],pred["ordinary_plus_time_TMA_and_Level"])
              }
            }
        print(outcome,"models done",flush=True)
    statsout={}
    for col in cols:
        a=df[col]
        statsout[col]={"mean":float(a.mean()),"std":float(a.std()),"nonzero_fraction":float(np.mean(abs(a)>1e-8))}
    output={"dataset":"Groove MIDI 1.0 97 long performances","n_windows":len(df),
            "n_drummers":df.drummer.nunique(),"features":KEYS,
            "feature_moments":statsout,"model_results":scores,
            "seconds":time.monotonic()-t,
            "caveats":["Per 4-bar four-beat continuous metronome windows, beat-grid 1/64 quarter note",
            "No alpha selected on holdout; all three reported",
            "Fixed-origin full TMA Level sets; level power fractions and phase correlations",
            "Phase cosine undefined at vanishing component amplitude and approximated as zero",
            "Cross-level spectra not guaranteed to be uniquely fractal; no causal inference",
            "Outcome timbre mixture uses instrument identities from same rhythmic performance"]}
    (OUT/"results.json").write_text(json.dumps(output,indent=2))
    print("DONE",json.dumps({"n":len(df),"results":scores,"features":statsout},indent=2),flush=True)
if __name__=="__main__":main()
