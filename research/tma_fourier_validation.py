#!/usr/bin/env python3
"""Fourier decomposition of TMA H/D waves on real variable drumming.

Unweighted attack spectrum and singleton-event static metrical-template spectra
are controlled. TMA FFT features describe same-window data; no prediction is
claimed. Drum pitch-family changes are withheld targets; leave-drummer-out
generalization.
"""
import io,math,json,collections,statistics,time,zipfile
from functools import lru_cache
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy import stats
import mido
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tma_variable_drums as base

OUT=Path("research/tma_fourier_results")
OUT.mkdir(parents=True,exist_ok=True)
N=base.N_BARS
WIN=base.WIN_BARS
GRID=base.GRID
SAMPLES=WIN*4*GRID # 1024 bins / four-bar analysis window
FS=GRID
SEED=20261008

def robustmean(x):return float(np.mean(x)) if len(x) else 0.

@lru_cache(maxsize=8192)
def singleton_structure(t):
    t=Fraction(t)
    return tuple(sorted(base.levels_full([t],N)[t]))

# Frequency expressed as cycles per notated quarter-note metronome beat.
FREQ=np.fft.rfftfreq(SAMPLES,d=1/FS)
BANDS=((.0625,.25),(.25,.5),(.5,1),(1,2),(2,4),(4,8),(8,16),(16,32.0001))
HARM=(.25,.5,1,2,4,8,16)

def spectral(x):
    x=np.asarray(x,float)
    assert len(x)==SAMPLES
    power=np.abs(np.fft.rfft(x))**2
    power[0]=0
    total=float(np.sum(power))
    ps=power/(total+1e-14)
    features={}
    for i,(a,b) in enumerate(BANDS):
        mask=(FREQ>=a)&(FREQ<b)
        features[f'band{i}']=float(ps[mask].sum())
    pp=ps[1:]
    features['entropy']=float(-np.sum(pp*np.log(pp+1e-14))/np.log(len(pp)))
    features['centroid']=float(np.sum(ps*FREQ))
    features['logpower']=float(np.log1p(total))
    # Slope is not called a fractal dimension: binning and sparse event trains
    # induce spectral structure even without fractal dynamics.
    freqs=np.geomspace(.25,16,9)
    fcent=[];pw=[]
    for a,b in zip(freqs[:-1],freqs[1:]):
        mask=(FREQ>=a)&(FREQ<b)
        if not mask.any():continue
        fcent.append(np.sqrt(a*b));pw.append(float(power[mask].mean()))
    features['logslope']=float(np.polyfit(np.log(fcent),np.log(np.array(pw)+1e-10),1)[0])
    for f in HARM:
        j=int(np.argmin(abs(FREQ-f)))
        features[f'peak_{str(f).replace(".","p")}']=float(ps[j])
    return features

def as_event_index(values):
    values=np.asarray(values,float)
    if len(values)<4:return np.zeros(SAMPLES)
    times=np.linspace(0,1,len(values))
    return np.interp(np.linspace(0,1,SAMPLES),times,values)

def feature_window(times,levels,index):
    lo=WIN*4*index
    hi=lo+WIN*4
    ts=[t for t in times if lo<=t<hi]
    q=np.asarray([float(t) for t in ts])
    grid_idx=np.asarray([int((t-Fraction(lo))*GRID) for t in ts],int)
    assert (grid_idx>=0).all() and (grid_idx<SAMPLES).all()
    A=np.zeros(SAMPLES);H=np.zeros(SAMPLES);D=np.zeros(SAMPLES)
    SH=np.zeros(SAMPLES);SD=np.zeros(SAMPLES)
    xH=np.array([max(levels[t]) for t in ts],float)
    xD=np.array([len(levels[t]) for t in ts],float)
    single=[singleton_structure(t) for t in ts]
    h0=np.array([max(a) for a in single],float)
    d0=np.array([len(a) for a in single],float)
    # Center per-event weights, avoid mistaking counts for structural variation.
    A[grid_idx]=1.0
    H[grid_idx]=xH-xH.mean()
    D[grid_idx]=xD-xD.mean()
    SH[grid_idx]=h0-h0.mean()
    SD[grid_idx]=d0-d0.mean()
    onset=np.diff(q)
    ordinal=np.arange(SAMPLES)
    ioi_ord=as_event_index(onset-np.mean(onset))
    H_ord=as_event_index(xH-xH.mean())
    D_ord=as_event_index(xD-xD.mean())
    specs={}
    for name,x in [('onset',A),('ioi_ordinal',ioi_ord),
                   ('singleH',SH),('singleD',SD),
                   ('H',H),('D',D),('H_ordinal',H_ord),('D_ordinal',D_ord),
                   ('contextH',H-SH),('contextD',D-SD)]:
        for k,v in spectral(x).items():specs[f'{name}_{k}']=v
    # Include relative phases wrt raw onset event train, rather than treating
    # power alone as a complete representation of the narrative.
    AF=np.fft.rfft(A)
    for name,x in [('singleH',SH),('singleD',SD),('H',H),('D',D),
                   ('contextH',H-SH),('contextD',D-SD)]:
        XF=np.fft.rfft(x)
        for f in (0.25,0.5,1,2):
            j=int(np.argmin(abs(FREQ-f)))
            z=XF[j]*np.conjugate(AF[j])
            z=z/(abs(z)+1e-12)
            label=str(f).replace(".","p")
            specs[f'{name}_relphase_{label}_cos']=float(z.real)
            specs[f'{name}_relphase_{label}_sin']=float(z.imag)
    specs.update({'fraction_contextual_stack':robustmean([levels[t]!=set(single[i]) for i,t in enumerate(ts)]),
                  'fraction_contextual_H':robustmean(xH!=h0),
                  'fraction_contextual_D':robustmean(xD!=d0),
                  'context_H_mean_abs':robustmean(abs(xH-h0)),
                  'context_D_mean_abs':robustmean(abs(xD-d0)),
                  'event_count':len(ts)})
    return specs

def summarize_feature_groups(names):
    def prefix(name):return sorted([f for f in names if f.startswith(name+"_")])
    b={
        'onset_FFT':prefix('onset')+prefix('ioi_ordinal'),
        'single_event_FFT':prefix('singleH')+prefix('singleD'),
        'TMA_FFT':prefix('H')+prefix('D'),
        'context_FFT':prefix('contextH')+prefix('contextD')}
    assert len(b['onset_FFT'])>30
    assert len(b['TMA_FFT'])>60
    return b

def leave_drummer_out(df,features,target,alpha):
    y=df[target].to_numpy(float)
    pred=np.zeros(len(df))
    for player in df.drummer.unique():
        test=(df.drummer==player).to_numpy()
        m=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        m.fit(df.loc[~test,features],y[~test])
        pred[test]=m.predict(df.loc[test,features])
    return pred

def group_mae(df,y,p):
    out={}
    for lab in ('all','regular','variable'):
        a=np.ones(len(df),bool) if lab=='all' else (df.selection==lab).to_numpy()
        out[lab]={'MAE':float(np.mean(abs(y[a]-p[a]))),
                  'R2':float(1-sum((y[a]-p[a])**2)/sum((y[a]-y[a].mean())**2))}
    return out

def paired_boot(df,baseline,other,target,B=2999):
    ids=list(df.drummer.unique())
    group={key:np.where(df.drummer.to_numpy()==key)[0] for key in ids}
    y=df[target].to_numpy(float)
    gain=abs(y-baseline)-abs(y-other)
    rng=np.random.default_rng(SEED+445)
    out={}
    for s in ('all','regular','variable'):
        selected=np.ones(len(df),bool) if s=='all' else (df.selection==s).to_numpy()
        avg=float(gain[selected].mean())
        samples=[]
        for i in range(B):
            ind=np.concatenate([group[x] for x in rng.choice(ids,len(ids),replace=True)])
            ii=ind[selected[ind]]
            if len(ii):samples.append(float(gain[ii].mean()))
        out[s]={'gain_MAE_positive_if_Fourier_better':avg,
                'drummer_bootstrap_95CI':np.quantile(samples,[.025,.975]).tolist(),
                'fraction_bootstrap_positive':float(np.mean(np.array(samples)>0))}
    return out

def main():
    start=time.monotonic()
    archive,metadata,_=base.load_data()
    in_df=pd.read_csv('research/tma_variable_drums_results/variable_drum_windows.csv')
    assert len(in_df)==679 and in_df.track.nunique()==97 and in_df.drummer.nunique()==10
    lookup={m['id']:m for m in metadata}
    specs=[];records={}
    for i,track in enumerate(sorted(in_df.track.unique())):
        meta=lookup[track]
        names=[n for n in archive.namelist() if n.endswith(meta['midi_filename'])]
        assert len(names)==1,track
        midi=mido.MidiFile(file=io.BytesIO(archive.read(names[0])))
        notes=[]
        for tr in midi.tracks:
            tick=0
            for m in tr:
                tick+=m.time
                if m.type=='note_on' and m.velocity>0:notes.append((tick/midi.ticks_per_beat,int(m.note),int(m.velocity)))
        prepared=base.prepare(notes)
        assert prepared is not None
        times,notes,grouped,quant_error,counts=prepared
        levels=base.levels_full(times,N)
        # Verify exact frozen engine (not a generic hierarchy) on representatives.
        if i<7:
            for bars in (4,8,16,32):base.check_engine(times,bars)
            for bars in (4,8,16,24):
                prefix=[t for t in times if t<bars*4]
                old=base.levels_full(prefix,bars)
                assert all(levels[t]==old[t] for t in prefix)
        for w in range(8):
            f=feature_window(times,levels,w)
            f.update({'track':track,'window':w})
            specs.append(f)
        if (i+1)%10==0:print('Analyzed',i+1,'of 97',flush=True)
    F=pd.DataFrame(specs)
    result=in_df.merge(F,on=['track','window'],how='left',validate='many_to_one')
    assert len(result)==679
    # Mean variation in spectrum relative to TMA's isolated-event metric template.
    contextcols=['fraction_contextual_stack','fraction_contextual_H','fraction_contextual_D',
                 'context_H_mean_abs','context_D_mean_abs']
    contextSummary={c:{'mean':float(F[c].mean()),'median':float(F[c].median()),
                    'p90':float(F[c].quantile(.9))} for c in contextcols}
    groups=summarize_feature_groups(F.columns)
    old_timing=[k for k in in_df.columns if k.startswith('timing_')]
    old_tma=[k for k in in_df.columns if k.startswith('origin_')]
    basecols=['window','bpm']+old_timing
    configs={
      'timing':basecols,
      'timing_onsetFFT':basecols+groups['onset_FFT'],
      'timing_onset_singleFFT':basecols+groups['onset_FFT']+groups['single_event_FFT'],
      'timing_onset_TMAFFT':basecols+groups['onset_FFT']+groups['TMA_FFT'],
      'timing_onset_single_TMAFFT':basecols+groups['onset_FFT']+groups['single_event_FFT']+groups['TMA_FFT'],
      'timing_onset_contextFFT':basecols+groups['onset_FFT']+groups['context_FFT'],
      'timing_onset_timeTMA':basecols+groups['onset_FFT']+old_tma,
      'timing_onset_timeTMA_TMAFFT':basecols+groups['onset_FFT']+old_tma+groups['TMA_FFT'],
      'TMA_FFT_only':['window','bpm']+groups['TMA_FFT'],
      'onset_FFT_only':['window','bpm']+groups['onset_FFT'],
      'static_template_FFT_only':['window','bpm']+groups['single_event_FFT'],
    }
    for key,col in configs.items():
        assert len(col)==len(set(col)),(key,len(col),len(set(col)))
        assert all(k in result.columns for k in col)
    alphas=(30,100,300,1000)
    output={}
    for outcome in ('change','drift'):
        output[outcome]={}
        Y=result[outcome].to_numpy(float)
        for alpha in alphas:
            preds={name:leave_drummer_out(result,cols,outcome,alpha) for name,cols in configs.items()}
            diagnostics={name:group_mae(result,Y,yhat) for name,yhat in preds.items()}
            paired={
                'TMA_FFT_beyond_onset_FFT':paired_boot(result,preds['timing_onsetFFT'],preds['timing_onset_TMAFFT'],outcome,B=999),
                'TMA_FFT_beyond_time_TMA':paired_boot(result,preds['timing_onset_timeTMA'],preds['timing_onset_timeTMA_TMAFFT'],outcome,B=999),
                'TMA_FFT_beyond_static_template':paired_boot(result,preds['timing_onset_singleFFT'],preds['timing_onset_single_TMAFFT'],outcome,B=999),
                'context_FFT_beyond_onset_FFT':paired_boot(result,preds['timing_onsetFFT'],preds['timing_onset_contextFFT'],outcome,B=999),
            }
            output[outcome][str(alpha)]={'metrics':diagnostics,'comparisons':paired}
            print(outcome,'alpha',alpha,'done',flush=True)
    path=OUT/'window_features.csv'
    result.to_csv(path,index=False)
    final={'dataset':'Groove MIDI 1.0; 97 performances, 10 drummers, 32 bars each',
           'n_tracks':int(result.track.nunique()),'n_drummers':int(result.drummer.nunique()),
           'n_windows':len(result),'FFT_grid_per_quarter_note':GRID,
           'FFT_timebase':'metronome quarter-note beat; 1024 slots per 4-bar window',
           'spectral_model':'mean-centered TMA H and D at attack times, zero elsewhere; extra event-index H/D and onset spectral features',
           'engine_gate':'28 engine-equivalence checks; 28 origin-prefix equivalence checks',
           'sample_context_ablation':contextSummary,
           'n_features':{k:len(v) for k,v in configs.items()},
           'model_results':output,'runtime_s':time.monotonic()-start,
           'limitations':[
            'Music TMA H(t) and D(t) are only defined at attacks; zero-filled impulse encoding is a chosen Fourier representation',
            'Four-bar spectra have limited low-frequency resolution, and power spectra discard temporal phase unless separately included',
            'Fourier features deterministically derive from note events and cannot add Shannon information to complete raw timing',
            'Pitches used only as independent orchestration-change outcome, held out from all timing/TMA descriptors',
            'Endogenous same-performance outcome, not prospective prediction, and only ten independent drummer groups',
            'Dyadic 1/64-quarter metrical projection does not represent non-dyadic tuplets exactly',
            'Alpha grid and many spectral features exploratory; evidence requires replication'
           ]}
    (OUT/'results.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print('RESULT_SUMMARY',json.dumps({'n_tracks':final['n_tracks'],'n_windows':final['n_windows'],
        'sample_context_ablation':contextSummary,'n_features':final['n_features'],
        'model_results':{k:{a:{'metrics':{z:v['all'] for z,v in d['metrics'].items()},
                                 'comparisons':{z:c['all'] for z,c in d['comparisons'].items()}}
                             for a,d in res.items()} for k,res in output.items()}},indent=2),flush=True)

if __name__=='__main__':main()
