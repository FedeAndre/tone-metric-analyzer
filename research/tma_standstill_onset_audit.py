#!/usr/bin/env python3
"""Audit attack extraction from six original Standstill2019 WAV stimuli.
Compare spectral-flux low/mid/high broadband onset detection and strict
metronome beat count; do not change clinical outcome model in this QC run.
"""
import io,json,hashlib
import requests,numpy as np
from scipy.io import wavfile
from scipy import signal
from pathlib import Path
import zipfile
URL='https://zenodo.org/api/records/21986395/files/Standstill2019.zip/content'
NAMES=['DRUMS90bpm.wav','DRUMS120bpm.wav','DRUMS140bpm.wav',
       'METRONOME90bpm.wav','METRONOME120bpm.wav','METRONOME140bpm.wav']
OUT=Path('research/tma_standstill_auditory_results');OUT.mkdir(exist_ok=True,parents=True)
def get():
 p=Path('/tmp/Standstill2019.zip')
 if not p.exists():
  with requests.get(URL,stream=True,timeout=(35,160)) as r:
   r.raise_for_status()
   with p.open('wb') as w:
    for z in r.iter_content(2**20):w.write(z)
 h=hashlib.md5(p.read_bytes()).hexdigest()
 assert h=='00ad3f608238feb7e43468bc7a78808e'
 return zipfile.ZipFile(p)
def analyze(y,sr,bpm):
 hop=256
 _,t,z=signal.stft(y,fs=sr,nperseg=1024,noverlap=1024-hop,
    boundary=None,padded=False)
 ff=np.fft.rfftfreq(1024,1/sr)
 magnitude=np.log1p(30*abs(z))
 delta=np.maximum(0,np.diff(magnitude,axis=1,prepend=magnitude[:,:1]))
 out={}
 for group,mask in [('low',ff<250),('mid',(ff>=250)&(ff<3000)),('high',ff>=3000),('all',np.ones(len(ff),bool))]:
  novelty=np.mean(delta[mask],axis=0)
  nl=novelty
  configs={}
  for frac in [.08,.15,.25,.40,.60,.80]:
   p,_=signal.find_peaks(nl,prominence=np.quantile(nl,.98)*frac,
       height=np.quantile(nl,.98)*frac,distance=int(.085*sr/hop))
   stamps=t[p]
   phase=(stamps*bpm/60)%1
   pc=float(abs(np.mean(np.exp(2j*np.pi*phase)))) if len(phase) else 0
   gaps=np.diff(stamps)
   configs[str(frac)]={'n':len(stamps),'resultant_phase':pc,
       'median_gap':float(np.median(gaps)) if len(gaps) else None,
       'head_seconds':stamps[:12].round(3).tolist(),'low':float(np.quantile(nl,.5)),
       'p90':float(np.quantile(nl,.90)),'p98':float(np.quantile(nl,.98))}
  out[group]=configs
 return out
def main():
 z=get();out={}
 for name in NAMES:
  sr,data=wavfile.read(io.BytesIO(z.read(next(p for p in z.namelist() if p.endswith(name)))))
  y=np.mean(data.astype(float)/2**31,axis=1)
  bpm=int(name.split('bpm')[0].replace('DRUMS','').replace('METRONOME',''))
  out[name]={'duration_s':len(y)/sr,'expected_nominal_clicks':round(len(y)/sr*bpm/60),
             'onsets':analyze(y,sr,bpm)}
  print('QC',name,'counts_all',{k:v['n'] for k,v in out[name]['onsets']['all'].items()},
     'low',{k:v['n'] for k,v in out[name]['onsets']['low'].items()},flush=True)
 (OUT/'onset_detection_audit.json').write_text(json.dumps(out,indent=2))
 print('FINAL',json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
