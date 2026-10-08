#!/usr/bin/env python3
"""TMA versus conventional sound-rhythm features, independent head micromotion.

Original University of Oslo Standstill2019, version 2.0.5 ZIP and all 6 WAV
stimuli, 16 participant groups, self-report 98 published eligible participants.

- Four-beat quarter-note tactus at nominal audio BPM, separate irregular drum attacks.
- Frozen TMA binary engine, verified against production analyze() on every sound.
- 8-beat audio windows, behavioral outcome is 0.2-5Hz filtered head speed
  relative to preceding silence; beat-locking is secondary.
- Compare conventional audio spectrum+tempo+loudness with static singleton and
  event-conditioned TMA Level Fourier features, 16-group held-out CV and
  six-stimulus-out CV; no outcome-driven selection.
"""
import hashlib,io,json,math,time,zipfile
from fractions import Fraction
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy import signal
from scipy.io import wavfile
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
import tma_variable_drums as tma

URL='https://zenodo.org/api/records/21986395/files/Standstill2019.zip/content'
MD5='00ad3f608238feb7e43468bc7a78808e'
OUT=Path('research/tma_standstill_auditory_results');OUT.mkdir(parents=True,exist_ok=True)
FS=120
SEED=20261008
SPEED_BAND=(.2,5)
TMA_LEVELS=('low','middle','high')
FREQUENCIES=(.25,.5,1.,2.)
F_LABELS=('0p25','0p5','1','2')
def get():
 p=Path('/tmp/Standstill2019.zip')
 if not p.exists() or p.stat().st_size<100000000:
  with requests.get(URL,stream=True,timeout=(40,180)) as r:
   print('ARCHIVE_DOWNLOAD',r.status_code,r.headers.get('content-length'),flush=True)
   r.raise_for_status()
   with p.open('wb') as w:
    for block in r.iter_content(2**20):
     if block:w.write(block)
 h=hashlib.md5()
 with p.open('rb') as f:
  while chunk:=f.read(2**20):h.update(chunk)
 assert h.hexdigest()==MD5,('unverified dataset',h.hexdigest())
 return zipfile.ZipFile(p)
def table(z,name):
 fn=next(s for s in z.namelist() if s.endswith(name))
 return pd.read_csv(io.BytesIO(z.read(fn)),sep='\t',comment='#')
def extract_audio(z,filename,tempo,kind):
 raw=z.read(next(q for q in z.namelist() if q.endswith('/sound_data/'+filename)))
 sr,y=wavfile.read(io.BytesIO(raw))
 assert sr==44100
 # Predefined audio-only correction: high-band spectral flux, validated
 # against exact 48,64,72 beat-aligned metronome attack counts before
 # examining any head-motion target. See onset_detection_audit.json.
 y=np.mean(y.astype(np.float64)/(2**31),axis=1)
 hop=256
 _,stamps,Z=signal.stft(y,fs=sr,nperseg=1024,noverlap=1024-hop,
                         boundary=None,padded=False)
 hz=np.fft.rfftfreq(1024,1/sr)
 magnitude=np.log1p(30.0*np.abs(Z))
 delta=np.maximum(0.0,np.diff(magnitude,axis=1,prepend=magnitude[:,:1]))
 novelty=np.mean(delta[hz>=3000],axis=0)
 cut=float(np.quantile(novelty,.98))*.40
 peaks,_=signal.find_peaks(novelty,height=cut,prominence=cut,
                           distance=int(.085*sr/hop))
 ts=stamps[peaks]
 beat=60/tempo
 expected=round((len(y)/sr)*tempo/60)
 if kind=="metronome":
  assert abs(len(ts)-expected)<=1,('metronome_audio_onset_QC_failed',filename,len(ts),expected)
 else:
  assert len(ts)>=1.2*expected,('drum_audio_onset_QC_failed',filename,len(ts),expected)
 # Quantize to maximum 1/64 quarter beat, use onset start as audio time origin.
 times=sorted(set(Fraction(int(round(t/beat*64)),64) for t in ts if t/beat>=0))
 times=[q for q in times if q>=0]
 bars=math.ceil((len(y)/sr/beat)/4)
 levels=tma.levels_full(times,bars)
 singleton={q:tma.levels_full([q],bars)[q] for q in times}
 tma.check_engine(times,bars)
 print('SOURCE_AUDIO',filename,'beats',round(len(y)/sr/beat,2),'events',len(times),'bars',bars,
       'complex',kind=='drums','mean_levels',round(float(np.mean([len(levels[q]) for q in times])),2),flush=True)
 # fixed sequential eight-beat windows (event indices not uniformly spaced)
 rows=[]
 for wi in range(int((len(y)/sr/beat)//8)):
  lo=wi*8;hi=(wi+1)*8
  chosen=[q for q in times if lo<=q<hi]
  if len(chosen)<4:continue
  feature={'tempo':tempo,'drums':int(kind=='drums'),'window_index':wi,
           'window_start_s':wi*8*beat,'window_end_s':(wi+1)*8*beat,
           'attacks':len(chosen),'onset_rate':len(chosen)/(8*beat)}
  v=np.array([float(q)-lo for q in chosen])
  intervals=np.diff(v)
  feature['ioi_cv']=float(np.std(intervals)/(np.mean(intervals)+1e-8))
  feature['ioi_mad']=float(np.mean(abs(np.diff(intervals)))) if len(intervals)>1 else 0.
  feature['phase_resultant']=float(abs(np.mean(np.exp(2j*np.pi*(v%1)))))
  feature['offbeat_frac']=float(np.mean(abs(v-np.rint(v))>.11))
  feature['onset_phase_entropy']=float(-sum((p*np.log(p)) for p in (np.histogram(v%1,bins=8,range=(0,1))[0]/len(v)) if p>0)/np.log(8))
  # measured loudness of ORIGINAL audio in matching time span, no future
  a=int(wi*8*beat*sr);b=min(len(y),int((wi+1)*8*beat*sr))
  local=y[a:b];feature['audio_rms']=float(np.sqrt(np.mean(local**2)))
  feature['audio_mean_abs']=float(np.mean(abs(local)))
  # 8 beats at 64 bins per beat.
  N=512;ON=np.zeros(N);H=np.zeros(N);D=np.zeros(N);sH=np.zeros(N);sD=np.zeros(N)
  occupancy=np.zeros((3,N))
  for q in chosen:
   j=int((q-lo)*64)
   if j>=N:continue
   S=levels[q];S0=singleton[q]
   ON[j]=1;H[j]=max(S);D[j]=len(S);sH[j]=max(S0);sD[j]=len(S0)
   occupancy[0,j]=sum(1<=k<=4 for k in S)
   occupancy[1,j]=sum(5<=k<=8 for k in S)
   occupancy[2,j]=sum(k>=9 for k in S)
  feature['height_mean']=float(np.mean([max(levels[q]) for q in chosen]))
  feature['density_mean']=float(np.mean([len(levels[q]) for q in chosen]))
  feature['context_density_mean']=float(np.mean([len(levels[q])-len(singleton[q]) for q in chosen]))
  feature['static_density_mean']=float(np.mean([len(singleton[q]) for q in chosen]))
  F=np.fft.rfftfreq(N,d=1/64)
  spectra={key:np.abs(np.fft.rfft(vv))**2 for key,vv in [
   ('onset',ON),('height',H),('density',D),('static_height',sH),('static_density',sD),
   ('context_height',H-sH),('context_density',D-sD)]}
  L=np.fft.rfft(occupancy,axis=1)
  for f,label in zip(FREQUENCIES,F_LABELS):
   k=int(np.argmin(abs(F-f)))
   # relative spectral power; no DC
   for key,P in spectra.items():
    feature[key+'_p'+label]=float(P[k]/(np.sum(P[1:])+1e-10))
   numerator=float(abs(np.sum(L[:,k]))**2)
   independent=float(np.sum(abs(L[:,k])**2))
   feature['crosslevel_ratio_'+label]=float(numerator/(independent+1e-10))
  rows.append(feature)
 return rows
def motion(z,groups,participants,segments,sounds):
 pass

def main():
 tick=time.monotonic();z=get()
 S=table(z,'/sound_data/segments.tsv')
 P=table(z,'/demographics/participant_marker_key.tsv')
 D=table(z,'/demographics/self_report.tsv')
 print('DATA SHAPE',S.shape,P.shape,D.shape,'DEMOG COLUMNS',list(D.columns)[:15],flush=True)
 assert {'group','condition','onset_s','offset_s','stimulus_id','stimulus_label'}.issubset(S.columns)
 assert len(P)==116 and len(S[S.condition=='music'])==16*6
 ID_TO_AUDIO={1:('DRUMS90bpm.wav',90,'drums'),2:('DRUMS120bpm.wav',120,'drums'),
    3:('DRUMS140bpm.wav',140,'drums'),4:('METRONOME90bpm.wav',90,'metronome'),
    5:('METRONOME120bpm.wav',120,'metronome'),6:('METRONOME140bpm.wav',140,'metronome')}
 sound={}
 for k,(name,bpm,typ) in ID_TO_AUDIO.items():sound[k]=extract_audio(z,name,bpm,typ)
 # Analyze ONLY published-included adults; no demographic leakage.
 col='Code' if 'Code' in D else 'code' if 'code' in D else None
 assert col is not None,('unknown questionnaire key',list(D.columns))
 eligible=set(D.loc[D.included_in_analysis==1,col].dropna().astype(str).str.lower())
 print('PUBLISHED ELIGIBLE',len(eligible),flush=True)
 keys=P[P.questionnaire_key.astype(str).str.lower().isin(eligible)].copy()
 assert 85<=len(keys)<=100,('unexpected study-level eligibility',len(keys))
 # Group-by-group signal filtering, stimulus and preceding-silence normalization.
 sos=signal.butter(3,SPEED_BAND,btype='bandpass',fs=120,output='sos')
 rows=[]
 for ig,g in enumerate(sorted(S.group.unique())):
  path=next(x for x in z.namelist() if x.endswith('/mocap_data/'+g+'.tsv'))
  data=pd.read_csv(io.BytesIO(z.read(path)),sep='\t',skiprows=10,dtype=float)
  group_m=S[(S.group==g)&(S.condition=='music')].sort_values('onset_s')
  people=keys[keys.group==g]
  for _,person in people.iterrows():
   marker=person.marker
   cs=[f'{marker} X',f'{marker} Y',f'{marker} Z']
   if not all(c in data.columns for c in cs):raise ValueError(('missing marker',g,marker,data.columns[:12].tolist()))
   xyz=data[cs].to_numpy(dtype=float)
   if np.any(~np.isfinite(xyz)):continue
   filtered=signal.sosfiltfilt(sos,xyz,axis=0)
   velocity=np.gradient(filtered,axis=0)*FS
   speed=np.sqrt(np.sum(velocity**2,axis=1))
   trname=str(person.questionnaire_key).lower()
   for _,entry in group_m.iterrows():
    sid=int(entry.stimulus_id)
    start=float(entry.onset_s)
    # previous 12 seconds are silent (protocol), not response period
    ia=int(max(0,start-15)*FS);ib=int(max(0,start-1)*FS)
    past=speed[ia:ib]
    baseline=float(np.mean(past))
    assert baseline>.02
    for f in sound[sid]:
     a=start+f['window_start_s']+.12;b=start+f['window_end_s']+.12
     aa=int(a*FS);bb=int(b*FS)
     if bb>len(speed) or (b>entry.offset_s):continue
     v=speed[aa:bb]
     qom=float(np.mean(v))
     ratio=float(np.log((qom+.25)/(baseline+.25)))
     # primary beat-locked motion envelope, independent accelerometry.
     bpm=ID_TO_AUDIO[sid][1];bsecs=60/bpm
     axes=velocity[aa:bb]
     phase=np.exp(-2j*np.pi*(np.arange(len(v))/FS)/bsecs)
     amp=float(np.sqrt(np.sum(abs((axes*phase[:,None]).mean(axis=0))**2))/(np.sqrt(np.mean(np.sum(axes**2,axis=1)))+1e-8))
     rows.append({'group':g,'participant':trname,'stimulus_id':sid,'stimulus_start_s':start,
       'baseline_motion':baseline,'motion_mean_mm_s':qom,'motion_log_change':ratio,
       'movement_beat_lock':amp,**f})
  print('GROUP',g,'persons',len(people),'rows',len(rows),flush=True)
 df=pd.DataFrame(rows)
 assert len(df)>1500 and df.participant.nunique()>=85,('too few observations',len(df))
 df.to_csv(OUT/'tma_mocap_window_features.csv',index=False)
 base=['tempo','drums','window_index','attacks','onset_rate','ioi_cv','ioi_mad',
       'phase_resultant','offbeat_frac','onset_phase_entropy','audio_rms','audio_mean_abs']+[f'onset_p{x}' for x in F_LABELS]
 static=['static_density_mean']+[f'static_{v}_p{x}' for v in ('height','density') for x in F_LABELS]
 full=['height_mean','density_mean']+[f'{v}_p{x}' for v in ('height','density') for x in F_LABELS]+[f'crosslevel_ratio_{x}' for x in F_LABELS]
 contextual=['context_density_mean']+[f'context_{v}_p{x}' for v in ('height','density') for x in F_LABELS]
 reps={'sound_design':['tempo','drums','window_index','audio_rms','audio_mean_abs'],
       'ordinary':base,'ordinary_plus_static':base+static,
       'ordinary_plus_TMA':base+full,
       'ordinary_plus_context':base+contextual,
       'ordinary_plus_static_and_context':base+static+contextual}
 targets=('motion_log_change','movement_beat_lock')
 def predict(yname,cols,alpha,heldout):
  X=df[cols];y=df[yname].to_numpy(float)
  blocks=df[heldout].to_numpy()
  loo=LeaveOneGroupOut()
  predictions=np.full(len(df),np.nan)
  for train,test in loo.split(X,y,blocks):
   mdl=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
   mdl.fit(X.iloc[train],y[train]);predictions[test]=mdl.predict(X.iloc[test])
  assert np.isfinite(predictions).all()
  return predictions
 rng=np.random.default_rng(SEED)
 models={}
 for tar in targets:
  models[tar]={}
  for heldout in ('group','stimulus_id'):
   models[tar][heldout]={}
   for alpha in (100,300):
    preds={n:predict(tar,cols,alpha,heldout) for n,cols in reps.items()}
    y=df[tar].to_numpy(float)
    scored={n:{'mae':float(np.mean(abs(y-p))),
       'R2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))} for n,p in preds.items()}
    controls={}
    if heldout=='group':
     units=sorted(df.group.unique())
     unidx={x:np.where(df.group.to_numpy()==x)[0] for x in units}
     for label in ('ordinary_plus_static','ordinary_plus_TMA','ordinary_plus_context','ordinary_plus_static_and_context'):
      diff=abs(y-preds['ordinary'])-abs(y-preds[label])
      draw=np.array([np.mean(diff[np.concatenate([unidx[x] for x in rng.choice(units,len(units),replace=True)])])
        for _ in range(2999)])
      controls[label]={'MAE_improvement':float(np.mean(diff)),
             'group_cluster_boot95':np.quantile(draw,[.025,.975]).tolist()}
    models[tar][heldout][str(alpha)]={'scores':scored,'incremental_effects':controls}
    print('CV',tar,heldout,alpha,'ordinary_R2',scored['ordinary']['R2'],
          'TMA_R2',scored['ordinary_plus_TMA']['R2'],flush=True)
 out={'dataset':'2019 University of Oslo Standstill 2026 data deposit 2.0.5',
   'n_participants':int(df.participant.nunique()),'n_groups':int(df.group.nunique()),
   'n_segments':int(df[['participant','stimulus_id']].drop_duplicates().shape[0]),
   'n_mocap_audio_windows':len(df),
   'stimulant_onsets':{str(k):len(sound[k]) for k in sound},
   'onset_detector':'audio-only frozen high-frequency spectral flux Q98 x0.40, exact click-count QC',
   'num_features':{k:len(v) for k,v in reps.items()},'models':models,
   'seconds':time.monotonic()-tick,
   'limitations':[
    'Observational stimulus responses to six non-homometric stimuli, not newly constructed matched rhythms',
    'Only six acoustically distinct stimuli; group CV generalizes to new participants not new audio exemplars',
    'Audio onset detection is automated from original WAV and TMA metrical anchoring is approximate onset-sampling',
    'Head motion synchronized to music playback, physiological onset alignment assumed protocol timeline with 120-ms evaluation lag',
    'Publication-eligible n=98, demographic exclusions archived',
    'Six stimuli at three tempos; all variance across stimuli may reflect timbre, loudness or rhythm',
    'Stimulus-heldout sixfold CV is essential control, only six folds',
    'Repeated windows within participants correlated; 16 group cluster bootstrap for group heldout gains']}
 (OUT/'tma_mocap_results.json').write_text(json.dumps(out,indent=2))
 print('FINAL_RESULT',json.dumps({k:v for k,v in out.items() if k!='limitations'},indent=2),flush=True)
if __name__=='__main__':main()
