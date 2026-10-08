#!/usr/bin/env python3
"""Inspect original University of Oslo Standstill 2019 audio and MoCap files,
SHA/MD5 verified, no data or models invented.
"""
import hashlib,io,json,time,zipfile
from pathlib import Path
import requests,numpy as np
from scipy.io import wavfile
URL='https://zenodo.org/api/records/21986395/files/Standstill2019.zip/content'
MD5='00ad3f608238feb7e43468bc7a78808e'
OUT=Path('research/tma_standstill_auditory_results');OUT.mkdir(parents=True,exist_ok=True)
def get():
 p=Path('/tmp/Standstill2019.zip')
 if not p.exists():
  with requests.get(URL,stream=True,timeout=(45,180)) as r:
   print('HTTP',r.status_code,'Size',r.headers.get('content-length'),flush=True);r.raise_for_status()
   with p.open('wb') as f:
    for block in r.iter_content(2**20):f.write(block)
 m=hashlib.md5(p.read_bytes()).hexdigest();assert m==MD5,('MD5 mismatch',m)
 print('ZIP VERIFIED',p.stat().st_size,m,flush=True)
 return zipfile.ZipFile(p)
def main():
 st=time.monotonic();z=get()
 names=z.namelist()
 counts={'motion':len([x for x in names if '/mocap_data/' in x and x.endswith('.tsv')]),
 'wav':len([x for x in names if x.endswith('.wav')]),
 'tables':len([x for x in names if x.endswith('.tsv')])}
 samples={}
 for name in names:
  if name.endswith('.wav'):
   b=z.read(name)
   fs,signal=wavfile.read(io.BytesIO(b))
   samples[name]={'fs':fs,'shape':list(signal.shape),'seconds':len(signal)/fs,'dtype':str(signal.dtype)}
 keys=['/sound_data/stimulus_order.tsv','/sound_data/segments.tsv','/metadata/segments_dictionary.tsv','/demographics/participant_marker_key.tsv','/demographics/qom.tsv']
 tables={}
 for k in keys:
  file=next((x for x in names if x.endswith(k)),None)
  if file:tables[k]={'file':file,'head':z.read(file).decode('utf8','replace').splitlines()[:12]}
 mocap=next((x for x in names if '/mocap_data/' in x and x.endswith('.tsv')),None)
 if mocap:tables['mocap_example']={'file':mocap,'head':z.read(mocap).decode('utf8','replace').splitlines()[:18]}
 out={'counts':counts,'wav':samples,'table_examples':tables,'all_paths':names,'duration_s':time.monotonic()-st}
 (OUT/'inspection.json').write_text(json.dumps(out,indent=2))
 print('INSPECTION',json.dumps({k:v for k,v in out.items() if k!='all_paths'},indent=2),flush=True)
if __name__=='__main__':main()
