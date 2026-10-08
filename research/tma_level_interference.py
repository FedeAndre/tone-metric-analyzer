#!/usr/bin/env python3
"""Identity and empirical Fourier interference of Tone-Metric Level occupancy.

D(t) == sum_k I_k(t) exactly. P_D(f) =
sum P_k(f) + 2 sum_{k<l} Re(I_k(f) * conj(I_l(f))).
Test on 97 Groove MIDI recordings, 679 4-bar windows.

Null: shuffle entire Level-stack vectors among the SAME observed attack times
within each 4-bar window, retaining rhythmic onsets and exact Level-stack
distributions. This specifically tests Level/metrical-position alignment, not
biological significance. Do not claim novel information beyond event times.
"""
import json,io,math
from fractions import Fraction
from pathlib import Path
import numpy as np,pandas as pd,mido
import tma_variable_drums as v

OUT=Path('research/tma_level_interference_results')
OUT.mkdir(exist_ok=True,parents=True)
N=1024;FS=64
F=np.fft.rfftfreq(N,1/FS)
SELECTED=(.25,.5,1,2)
B=59
SEED=20261008

def metric(t,stack,maxlevel):
    T=np.zeros((maxlevel,N))
    q=np.array([int(float((a-t*16)*FS)+1e-7) for a in stack[0]],int)
    if not np.all((q>=0)&(q<N)):raise RuntimeError('bin')
    for j,s in enumerate(stack[1]):
        if s:
            for k in s:T[k-1,q[j]]+=1.
    Z=np.fft.rfft(T-T.mean(axis=1,keepdims=True),axis=1)
    D=np.fft.rfft(T.sum(axis=0)-np.mean(T.sum(axis=0)))
    assert np.allclose(Z.sum(axis=0),D,atol=1e-8)
    power=np.sum(abs(Z)**2,axis=0)
    dd=abs(D)**2
    out={}
    for f in SELECTED:
        k=int(round(f*N/FS))
        x=float(power[k])
        y=float(dd[k])
        # interference ratio corrected for each Level's sum of powers.
        out[f'power_ratio_{str(f).replace(".","p")}']=float(y/(x+1e-10))
        out[f'cross_term_{str(f).replace(".","p")}']=float((y-x)/(x+1e-10))
    L=np.array([len(x) for x in stack[1]],float)
    out['n_hits']=len(q)
    out['n_level_depths']=maxlevel
    out['mean_density']=float(L.mean())
    return out

def main():
    z,meta,_=v.load_data()
    a=pd.read_csv('research/tma_fourier_results/window_features.csv')
    lookup={row['id']:row for row in meta}
    ids=sorted(a.track.unique());rows=[]
    for i,rec in enumerate(ids):
        item=lookup[rec]
        fn=next(n for n in z.namelist() if n.endswith(item['midi_filename']))
        mf=mido.MidiFile(file=io.BytesIO(z.read(fn)))
        notes=[]
        for tr in mf.tracks:
            tick=0
            for m in tr:
                tick+=m.time
                if m.type=='note_on' and m.velocity>0:
                    notes.append((tick/mf.ticks_per_beat,int(m.note),int(m.velocity)))
        pre=v.prepare(notes);assert pre is not None
        ts=pre[0];lv=v.levels_full(ts,32)
        maxl=max(max(x) for x in lv.values())
        for w in range(1,8):
            event=[x for x in ts if 16*w<=x<(w+1)*16]
            stacks=[sorted(lv[x]) for x in event]
            obs=metric(w,(event,stacks),maxl)
            rng=np.random.default_rng(SEED+i*37+w*193)
            null=[]
            for bi in range(B):
                perm=rng.permutation(len(stacks))
                shuffled=[stacks[j] for j in perm]
                null.append(metric(w,(event,shuffled),maxl))
            row={'track':rec,'drummer':a[a.track==rec].drummer.iloc[0],'window':w}
            for name,vv in obs.items():
                row[name]=vv
                if name.startswith('power_ratio_'):
                    nv=np.array([n[name] for n in null])
                    row[name+'_minus_null']=float(vv-np.mean(nv))
                    row[name+'_null_p']=float((1+np.sum(abs(nv-np.mean(nv))>=abs(vv-np.mean(nv))))/(B+1))
            rows.append(row)
        if (i+1)%20==0:print('Checked',i+1,'of',len(ids),flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==679
    df.to_csv(OUT/'interference_windows.csv',index=False)
    summary={'tracks':len(ids),'windows':len(df),'nulls_per_window':B,'frequencies':SELECTED}
    frequencies={}
    for f in SELECTED:
        col='power_ratio_'+str(f).replace('.','p')
        dat=df[col].to_numpy(float)
        residual=df[col+'_minus_null'].to_numpy(float)
        per=df[col+'_null_p'].to_numpy(float)
        frequencies[str(f)]={
          'median_power_ratio':float(np.median(dat)),
          'mean_power_ratio':float(np.mean(dat)),
          'fraction_negative_interference':float(np.mean(dat<1.)),
          'fraction_positive_interference':float(np.mean(dat>1.)),
          'median_difference_from_permuted_Level_stacks':float(np.median(residual)),
          'mean_difference_from_permuted_Level_stacks':float(np.mean(residual)),
          'n_per_window_p_le_0p05':int(np.sum(per<=.05))}
    rng=np.random.default_rng(SEED)
    dgs=list(df.drummer.unique())
    for f in SELECTED:
        key=str(f)
        col='power_ratio_'+key.replace('.','p')+'_minus_null'
        estimates=[]
        for _ in range(3999):
            draws=rng.choice(dgs,len(dgs),replace=True)
            xx=np.concatenate([df.loc[df.drummer==d,col].values for d in draws])
            estimates.append(float(np.mean(xx)))
        frequencies[key]['drummer_cluster_boot95']=np.quantile(estimates,[.025,.975]).tolist()
    summary['results']=frequencies
    summary['limitations']=[
     'Level occupancy is generated by TMA engine from original timings. Fourier adds no new source information',
     'Surrogate controls alignment of Level stacks with observed onset positions, not general physiological mechanisms',
     'Frequency-specific phase and power ratio is undefined if little or no Level spectral energy, stabilized numerically',
     'All 97 recordings are drumming under metronome, not physiological disease data',
     'Descriptive effect of Level-frequency decomposition is separate from clinical predictive utility']
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    print('PASS exact Fourier decomposition, all 679 windows')
    print('FINAL',json.dumps(summary,indent=2))
if __name__=='__main__':main()
