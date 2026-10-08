#!/usr/bin/env python3
"""Within-performance fractal TMA narrative of expressive human drumming.
Dataset: Gillick et al. Groove MIDI; metronome supplies actual beat reference.
Independent outcome: drum instrument/orchestration changes (pitch classes not fed
to TMA, which sees only attacks and the 4/4 metrical grid).

Frozen repository engine Level rules; explicit mathematical equivalence gates.
Comparison: tempo/onsets, cumulative nonfractal timing history, origin-based
TMA, window-restarted TMA; leave-drummer-out evaluation.
"""
from __future__ import annotations
import collections,csv,hashlib,io,json,math,statistics,time,zipfile
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import requests,mido
from scipy.spatial.distance import jensenshannon
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error

from tone_metric.engine import recursive_integer_structure,_next_seq_anchor,analyze
from tone_metric.models import Hit,MeasureInfo
from tone_metric.pivots import build_pivot_profile
from tone_metric.trees import build_tree_profile

OUT=Path('research/tma_variable_drums_results')
OUT.mkdir(parents=True,exist_ok=True)
URL='https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip'
SHA='651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e'
SEED=20261008
N_BARS=32
WIN_BARS=4
N_WINDOWS=8
GRID=64 # dyadic subdivisions of one quarter-note metronome pulse
ALPHA=100
SURROGATE_B=29
TMA_FIELDS=['mean_H','std_H','mean_D','std_D','mean_L','multilevel','H_jumps',
 'level_turnover','pivot_rate','pivot_depth','tree_span','tree_long','tree_cross']
BASE_FIELDS=['events_per_bar','attacks_per_bar','events_cv','ioi_mean','ioi_std',
 'ioi_madiff','phase_entropy8','phase_entropy16','phase_entropy32',
 'offbeat_fraction','microtiming_mean','microtiming_std','phase_autocorr1',
 'bar_count_autocorr1','bar_count_madiff','event_quarter_R',
 'cumulative_events_cv','cumulative_phase_entropy16','cumulative_offbeat',
 'cumulative_bar_count_acf1']

def mean(x):return float(np.mean(x)) if len(x) else 0.
def sd(x):return float(np.std(x,ddof=1)) if len(x)>1 else 0.
def acf(x):return float(np.corrcoef(x[:-1],x[1:])[0,1]) if len(x)>3 and np.std(x[:-1])>1e-9 and np.std(x[1:])>1e-9 else 0.
def entropy(q,k):
    if not len(q):return 0.
    x=np.bincount(np.minimum(k-1,(np.asarray(q)%1*k).astype(int)),minlength=k)
    p=x[x>0]/np.sum(x)
    return float(-(p*np.log(p)).sum()/np.log(k))

def load_data():
    r=requests.get(URL,timeout=120);r.raise_for_status()
    raw=r.content
    assert hashlib.sha256(raw).hexdigest()==SHA,'Canonical MIDI checksum mismatch'
    z=zipfile.ZipFile(io.BytesIO(raw))
    infos=[n for n in z.namelist() if n.endswith('info.csv')]
    meta=list(csv.DictReader(io.StringIO(z.read(infos[0]).decode('utf-8-sig'))))
    name_map={n.split('/')[-1]:n for n in z.namelist() if n.endswith('.mid')}
    return z,meta,name_map

def get_midi(z,meta,name_map):
    p=meta['midi_filename']
    fn=p.split('/')[-1]
    member=name_map[fn]
    mf=mido.MidiFile(file=io.BytesIO(z.read(member)))
    notes=[]
    for track in mf.tracks:
        tick=0
        for msg in track:
            tick+=msg.time
            if msg.type=='note_on' and msg.velocity>0:
                notes.append((tick/mf.ticks_per_beat,int(msg.note),int(msg.velocity)))
    notes.sort()
    return notes

def groups(pitch):
    if pitch==36:return 0 # kick
    if pitch in (37,38,40):return 1 # snare
    if pitch in (42,44,46,22,26):return 2 # hi-hat
    if pitch in (43,45,47,48,50,58):return 3 # toms
    if pitch in (49,52,55,57):return 4 # crash
    if pitch in (51,53,59):return 5 # ride
    return 6

def prepare(notes):
    # First N_BARS only, no reference to future events.
    notes=[(t,p,v) for t,p,v in notes if 0<=t<4*N_BARS]
    times=set()
    note_groups=collections.defaultdict(list)
    errors=[]
    for t,p,v in notes:
        q=Fraction(round(t*GRID),GRID)
        if q>=4*N_BARS:continue
        times.add(q)
        note_groups[int(q//(4*WIN_BARS))].append((q,groups(p),v))
        errors.append(abs(float(q)-t))
    ts=sorted(times)
    if len(ts)<180:return None
    counts=[sum(4*i<=float(t)<4*(i+1) for t in ts) for i in range(N_BARS)]
    if min(sum(counts[i:i+4]) for i in range(0,N_BARS,4))<14:return None
    return ts,notes,note_groups,errors,counts

def levels_full(times,n_bars=N_BARS):
    """Binary 4/4 specialization of frozen engine for dyadically projected MIDI.
    All recursive boundaries are identical to engine._refine_span; same level rules.
    """
    nbeats=n_bars*4
    intlayers={}
    parent={}
    recursive_integer_structure(1,_next_seq_anchor(2,nbeats+1),2,1,intlayers,parent)
    levels={t:set() for t in times}
    seen=set(times)
    for p,ll in intlayers.items():
        if p<=nbeats+1:
            t=Fraction(p-1)
            if t in seen:levels[t].update(ll)
    bybeat=collections.defaultdict(list)
    for t in times:bybeat[int(t)].append(t)
    def refine(a,b,ev,p):
        interior=[t for t in ev if a<t<b]
        if not interior:return
        m=(a+b)/2
        level=p+1
        for t in (a,m,b):
            if t in seen:levels[t].add(level)
        left=[t for t in ev if a<=t<=m]
        right=[t for t in ev if m<=t<=b]
        if any(a<t<m for t in left):refine(a,m,left,level)
        if any(m<t<b for t in right):refine(m,b,right,level)
    for j,ts in bybeat.items():
        a=Fraction(j)
        b=a+1
        refine(a,b,ts,int(parent.get(j+1,1)))
    if not all(levels.values()):raise RuntimeError('Unmapped event in frozen binary engine')
    return levels

def check_engine(times,nbar):
    prefix=[t for t in times if t<nbar*4]
    ours=levels_full(prefix,nbar)
    hits=[]
    for t in prefix:
        mi=int(t)//4
        hits.append(Hit(onset=t,duration=Fraction(0),measure_index=mi,measure_number=str(mi+1),
                        offset_in_measure=t-4*mi,sources=[],canonical_recovered=False))
    ms=[MeasureInfo(index=i,number=str(i+1),start=Fraction(4*i),full_duration=Fraction(4),
        actual_duration=Fraction(4),pickup_shift=Fraction(0),numerator=4,denominator=4) for i in range(nbar)]
    r=analyze(hits,ms)
    actual={Fraction(q['onset_quarter']):set(q['tone_metric_levels']) for s in r['segments'] for q in s['events']}
    assert ours==actual,(len(ours),len(actual),next(((t,ours[t],actual.get(t)) for t in ours if ours[t]!=actual.get(t)),None))

def wave_features(times,nbar,reset=False):
    if reset:
        # Each 4-bar phrase locally reset to zero; no long-range TMA history.
        result=[]
        for idx in range(0,nbar,WIN_BARS):
            local=[t-Fraction(idx*4) for t in times if idx*4<=t<(idx+WIN_BARS)*4]
            result.append(wave_one(local,WIN_BARS,0,WIN_BARS))
        return np.asarray(result,float)
    return wave_one(times,nbar,None,None)

def wave_one(times,nbar,ignore0,ignore1):
    levels=levels_full(times,nbar)
    wave=[]
    for t in times:
        ll=sorted(levels[t]);mi=int(t)//4
        wave.append({'segment_index':0,'measure_index':mi,'measure_number':str(mi+1),
          'onset_quarter':str(t),'offset_in_measure_quarter':str(t-mi*4),
          'levels':ll,'height':max(ll),'density':len(ll),
          'lowest_level':min(ll),'attack':True,'parenthetical':False})
    piv=build_pivot_profile(wave)
    tree=build_tree_profile(wave)['branches']
    results=[]
    segments=[(i*WIN_BARS,(i+1)*WIN_BARS) for i in range(nbar//WIN_BARS)] if ignore0 is None else [(ignore0,ignore1)]
    for lo,hi in segments:
        a=[e for e in wave if lo<=e['measure_index']<hi]
        H=np.asarray([e['height'] for e in a],float)
        D=np.asarray([e['density'] for e in a],float)
        L=np.asarray([e['lowest_level'] for e in a],float)
        turnover=[1-len(set(p['levels'])&set(q['levels']))/len(set(p['levels'])|set(q['levels'])) for p,q in zip(a[:-1],a[1:])]
        # The pivot becomes part of the narrative at its confirmed recovery,
        # never retrospectively at its crest.
        confirmed=[p for p in piv if lo<=int(Fraction(p['recovery_onset_quarter']))//4<hi]
        br=[b for b in tree if lo<=b['target_measure_index']<hi]
        distances=[b['target_node_index']-b['source_node_index'] for b in br]
        vals=[mean(H),sd(H),mean(D),sd(D),mean(L),mean(D>1),
              mean(abs(np.diff(H))),mean(turnover),
              len(confirmed)/max(len(a),1),mean([p['drop_depth'] for p in confirmed]),
              mean(distances),mean(np.asarray(distances)>=8) if distances else 0.,
              mean([b['source_measure_index']<lo for b in br])]
        results.append(vals)
    return np.asarray(results,float) if ignore0 is None else np.array(results[0],float)

def plain_features(times,notes,counts):
    time=np.array([float(t) for t in times],float)
    window=[]
    for i in range(N_WINDOWS):
        lo=i*WIN_BARS*4;hi=(i+1)*WIN_BARS*4
        e=time[(time>=lo)&(time<hi)]
        past=time[time<hi]
        within=np.diff(e)
        prev=np.diff(past)
        phase=e%1
        sh=counts[i*WIN_BARS:(i+1)*WIN_BARS]
        cp=counts[:(i+1)*WIN_BARS]
        jitter=np.abs(e*GRID-np.rint(e*GRID))/GRID # always zero after projection; use original note onset times instead.
        orig=np.array([t for t,p,v in notes if lo<=t<hi],float)
        actual_jitter=np.abs(orig*GRID-np.rint(orig*GRID))/GRID
        vals=[len(e)/WIN_BARS,
          len(orig)/WIN_BARS,sd(sh)/(mean(sh)+1e-9),
          mean(within),sd(within),mean(abs(np.diff(within))),
          entropy(phase,8),entropy(phase,16),entropy(phase,32),
          mean(np.abs(phase-np.round(phase*2)/2)>.1),
          mean(actual_jitter),sd(actual_jitter),
          acf(within),acf(sh),mean(abs(np.diff(sh))),
          float(abs(np.mean(np.exp(2j*np.pi*phase)))) if len(phase) else 0.,
          sd(cp)/(mean(cp)+1e-9),entropy(past%1,16),
          mean(np.abs((past%1)-np.round((past%1)*2)/2)>.1),
          acf(cp)]
        window.append(vals)
    return np.asarray(window,float)

def orchestration(groups_by_window):
    hist=[]
    for i in range(N_WINDOWS):
        rows=groups_by_window[i]
        v=np.ones(7)*.5 # Dirichlet-smoothed distribution of withheld MIDI pitch families
        for _,c,vel in rows:v[c]+=1
        hist.append(v/v.sum())
    deltas=[]
    drifts=[]
    for i,v in enumerate(hist):
        deltas.append(float(jensenshannon(v,hist[i-1],base=2)) if i else 0.)
        drifts.append(float(jensenshannon(v,hist[0],base=2)))
    return np.array(deltas),np.array(drifts)

def analyze_track(item,raw):
    meta,selection_class=item
    names=[n for n in raw.namelist() if n.endswith(meta['midi_filename'])]
    if not names:return None
    mf=mido.MidiFile(file=io.BytesIO(raw.read(names[0])))
    notes=[]
    for track in mf.tracks:
        tick=0
        for m in track:
            tick+=m.time
            if m.type=='note_on' and m.velocity>0:notes.append((tick/mf.ticks_per_beat,int(m.note),int(m.velocity)))
    prepared=prepare(notes)
    if prepared is None:return None
    times,notes,gmap,quant_err,counts=prepared
    orig=wave_features(times,N_BARS)
    reset=wave_features(times,N_BARS,True)
    plain=plain_features(times,notes,counts)
    trans,drift=orchestration(gmap)
    variability=float(np.std(counts,ddof=1)/(np.mean(counts)+1e-9))
    # no driver labels supplied to TMA. All learning performed across drummers.
    return {
      'track':meta['id'],'drummer':meta['drummer'],'style':meta['style'],
      'selection':selection_class,'bpm':float(meta['bpm']),
      'variable_cv':variability,'n_unique_events':len(times),
      'hits_per_bar':len(notes)/N_BARS,
      'quant_error_ms_median':float(np.median(quant_err)*60e3/float(meta['bpm'])),
      'quant_error_ms_p95':float(np.quantile(quant_err,.95)*60e3/float(meta['bpm'])),
      'original':orig.tolist(),'reset':reset.tolist(),
      'conventional':plain.tolist(),'outcome_change':trans.tolist(),
      'outcome_drift':drift.tolist(),
      'attack_times':[str(t) for t in times]
    }

def build_frame(dataset):
    rows=[]
    for r in dataset:
        x=np.array(r['original']);y=np.array(r['reset']);z=np.array(r['conventional'])
        for i in range(1,N_WINDOWS):
            row={k:r[k] for k in ('track','drummer','selection','bpm','variable_cv','n_unique_events','hits_per_bar')}
            row.update({'window':i,'change':r['outcome_change'][i],
                        'drift':r['outcome_drift'][i]})
            for name,arr,columns in [('origin',x,TMA_FIELDS),('reset',y,TMA_FIELDS),('timing',z,BASE_FIELDS)]:
                for j,key in enumerate(columns):
                    row[f'{name}_{key}']=arr[i,j]
                    row[f'{name}_{key}_dprev']=arr[i,j]-arr[i-1,j]
                    row[f'{name}_{key}_dstart']=arr[i,j]-arr[0,j]
            rows.append(row)
    return pd.DataFrame(rows)

def heldout(df,features,ykey,alpha=ALPHA):
    res=np.zeros(len(df))
    y=df[ykey].to_numpy(float)
    for drummer in sorted(df.drummer.unique()):
        te=(df.drummer==drummer).to_numpy()
        tr=~te
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(df.loc[tr,features].to_numpy(float),y[tr])
        res[te]=model.predict(df.loc[te,features].to_numpy(float))
    return res

def group_boot_gain(df,baseline,extended,target,n_boot=2999):
    u=df.drummer.unique()
    group={k:np.where((df.drummer==k).to_numpy())[0] for k in u}
    y=df[target].to_numpy()
    d=np.abs(y-baseline)-np.abs(y-extended)
    rng=np.random.default_rng(SEED+333)
    boot=np.array([mean(d[np.concatenate([group[x] for x in rng.choice(u,len(u),replace=True)])]) for i in range(n_boot)])
    return {'mean_MAE_gain':float(mean(d)),'drummer_bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist(),
           'boot_frac_gt0':float(np.mean(boot>0))}

def validation_select(meta,scout):
    # Stratify on first 32 bars' note-count CV based on stable official metadata;
    # selection deliberately includes highly variable and relatively steady
    # performances for a direct variability interaction.
    allow={m['midi_filename']:m for m in meta}
    records=[s for s in scout if s['nbar']>=32 and s['time_signature']=='4-4' and s['path'] in allow]
    grouped=collections.defaultdict(list)
    for r in records:grouped[r['drummer']].append(r)
    selected=[]
    for drummer,items in sorted(grouped.items()):
        items=sorted(items,key=lambda x:(x['cv_count'],x['path']))
        cap=min(18,len(items))
        bottom=items[:cap//2]
        top=items[-(cap-cap//2):]
        seen=set()
        for kind,subset in [('regular',bottom),('variable',top)]:
            for s in subset:
                if s['path'] in seen:continue
                selected.append((allow[s['path']],kind));seen.add(s['path'])
    assert len(selected)>=90
    return selected

def narrative_surrogates(tracks,B=SURROGATE_B):
    # Detect genuine long-range ordering in the full TMA narrative. Morphological
    # structure is compared with all 4-bar phrases permuted as intact blocks,
    # preserving each phrase's internal event structure and overall event histogram.
    out=[]
    for ix,row in enumerate(tracks):
        times=[Fraction(s) for s in row['attack_times']]
        orig=np.asarray(row['original'],float)
        rng=np.random.default_rng(SEED+711*ix)
        ns=[]
        for bi in range(B):
            perm=rng.permutation(N_WINDOWS)
            swapped=sorted(t-Fraction(int(t)//(4*WIN_BARS)*(4*WIN_BARS))
                +Fraction(int(j)*(4*WIN_BARS)) for j,k in enumerate(perm)
                for t in times if int(t)//(4*WIN_BARS)==k)
            ns.append(wave_features(swapped,N_BARS))
        null=np.asarray(ns)
        m=null.mean(axis=0);std=null.std(axis=0,ddof=1)
        # Only features with cross-null variance. Avoid contrast due to constant
        # absolute Level drift and deterministic architecture.
        active=std>1e-5
        if not np.any(active):continue
        z=np.zeros_like(orig);zn=np.zeros_like(null)
        z[active]=(orig[active]-m[active])/std[active]
        for j in range(len(null)):zn[j][active]=(null[j][active]-m[active])/std[active]
        obs=float(np.mean(np.square(np.clip(z[active],-20,20))))
        vals=np.array([np.mean(np.square(np.clip(zn[j][active],-20,20))) for j in range(len(null))])
        p=(1+np.sum(vals>=obs))/(1+len(vals))
        out.append({'track':row['track'],'selection':row['selection'],'variability_cv':row['variable_cv'],
                    'order_p':float(p),'order_score':obs,'null_mean_score':float(vals.mean())})
    return out

def main():
    clock=time.monotonic()
    archive,metadata,mnames=load_data()
    scout_url=('https://raw.githubusercontent.com/FedeAndre/tone-metric-analyzer/'
          'research-tma-variable-drums-20261008/groove_scout.json')
    scout=requests.get(scout_url,timeout=90).json()
    chosen=validation_select(metadata,scout)
    print('Selected',len(chosen),'tracks',flush=True)
    tracks=[]
    for i,item in enumerate(chosen):
        row=analyze_track(item,archive)
        if row:tracks.append(row)
        if (i+1)%10==0:print('Analyzed',i+1,'of',len(chosen),flush=True)
    assert len(tracks)>=80,'Not enough correctly parsed tracks'
    # Gate frozen paper fidelity and fixed-prefix causal computation.
    for row in tracks[:5]:
        times=[Fraction(s) for s in row['attack_times']]
        for n in (4,8,16,32):check_engine(times,n)
        whole=levels_full(times,N_BARS)
        for n in (4,8,16,24):
            small=[t for t in times if t<4*n]
            pref=levels_full(small,n)
            assert all(pref[t]==whole[t] for t in pref)
    print('PASS exact-engine levels and prefix invariance 20 + 20 checks',flush=True)
    df=build_frame(tracks)
    df.to_csv(OUT/'variable_drum_windows.csv',index=False)
    features=lambda name,ks:[f'{name}_{k}{suffix}' for k in ks for suffix in ('','_dprev','_dstart')]
    normal=features('timing',BASE_FIELDS)
    core=['mean_D','mean_H','H_jumps','level_turnover','pivot_rate','tree_span']
    orig=features('origin',TMA_FIELDS)
    reset=features('reset',TMA_FIELDS)
    models={'conventional_timing':normal,
       'origin_TMA_only':orig,
       'reset_TMA_only':reset,
       'timing_plus_origin_TMA':normal+orig,
       'timing_plus_reset_TMA':normal+reset,
       'timing_plus_core_origin':normal+features('origin',core),
       'timing_plus_core_reset':normal+features('reset',core)}
    results={}
    predfile=df[['track','drummer','selection','variable_cv','window']].copy()
    for target in ('change','drift'):
        y=df[target].to_numpy(float)
        results[target]={}
        preds={}
        for k,keys in models.items():
            pred=heldout(df,keys,target)
            preds[k]=pred
            predfile[f'{target}_{k}']=pred
            results[target][k]={'MAE':mean(abs(y-pred)),
                'R2':float(1-sum((y-pred)**2)/sum((y-y.mean())**2))}
        for newkey in ['timing_plus_origin_TMA','timing_plus_core_origin','timing_plus_reset_TMA','timing_plus_core_reset']:
            results[target][f'gain_{newkey}']=group_boot_gain(df,preds['conventional_timing'],preds[newkey],target)
        for selection in ('regular','variable'):
            sub=(df.selection==selection).to_numpy()
            results[target][selection+'_only_metrics']={
              k:{'MAE':mean(abs(y[sub]-pred[sub])),
                 'R2':float(1-sum((y[sub]-pred[sub])**2)/sum((y[sub]-y[sub].mean())**2))}
              for k,pred in preds.items()}
        predfile[target]=y
    predfile.to_csv(OUT/'drummer_holdout_predictions.csv',index=False)
    # Surrogate all included performances; class-specific false-positive rates
    # reveal whether *structured* variability matters.
    order=narrative_surrogates(tracks)
    (OUT/'narrative_surrogates.json').write_text(json.dumps(order,indent=2))
    surstats={}
    for sel in ('regular','variable'):
        tmp=[x for x in order if x['selection']==sel]
        surstats[sel]={'n':len(tmp),'n_p_le_0p05':sum(x['order_p']<=.05 for x in tmp),
             'n_p_le_0p10':sum(x['order_p']<=.1 for x in tmp),
             'median_p':float(np.median([x['order_p'] for x in tmp]))}
    out={'dataset':'Gillick et al. Groove MIDI 1.0 MIDI-only, CC BY 4.0',
         'dataset_source':URL,'data_sha256':SHA,'sample_n_tracks':len(tracks),
         'n_drummer_groups':int(df.drummer.nunique()),'n_windows':len(df),
         'selected_label_counts':df.drop_duplicates('track').selection.value_counts().to_dict(),
         'criterion':'changes in drum orchestration families withheld entirely from TMA and timing features',
         'quantization':'1/64 quarter-note metronome beat; nearest snapping, simultaneous attacks merged',
         'median_quantization_error_ms':float(np.median([r['quant_error_ms_median'] for r in tracks])),
         'median_p95_quant_error_ms':float(np.median([r['quant_error_ms_p95'] for r in tracks])),
         'median_note_count_CV':float(np.median([r['variable_cv'] for r in tracks])),
         'median_hits_per_bar':float(np.median([r['hits_per_bar'] for r in tracks])),
         'median_unique_attacks_per_32bars':float(np.median([r['n_unique_events'] for r in tracks])),
         'models':results,'narrative_order_surrogate':surstats,
         'limitations':['Labels of high/low variability selected from note-count CV on full recordings (not causal); descriptors themselves strictly fixed-origin',
          'Only selected long 4/4 grooves, no within-track human-annotated transition labels',
          'Independent orchestration metric uses withheld instrument pitches, but events originate in same MIDI recording',
          '1/64-beat dyadic snap loses exact non-dyadic tuplets and some human microtiming',
          'Statistical comparison descriptive and exploratory; no clinical data'],
         'elapsed_seconds':time.monotonic()-clock}
    (OUT/'result.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print('FINAL_RESULT '+json.dumps({k:out[k] for k in ('sample_n_tracks','n_drummer_groups','n_windows','median_note_count_CV','median_hits_per_bar','median_unique_attacks_per_32bars','models','narrative_order_surrogate')},indent=2),flush=True)

if __name__=='__main__':main()
