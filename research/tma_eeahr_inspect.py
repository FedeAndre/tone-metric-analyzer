#!/usr/bin/env python3
import requests,json
url='https://zenodo.org/api/records/15412761'
r=requests.get(url,timeout=70);print('STATUS',r.status_code,'bytes',len(r.content),flush=True);r.raise_for_status()
d=r.json()
print(json.dumps([{'key':f.get('key'),'size':f.get('size'),'links':f.get('links')} for f in d.get('files',[])],indent=2),flush=True)
for f in d.get('files',[]):
 if 'Measurements_IDs_14_15_16.rar' in f.get('key',''):
  u=f['links']['self']
  try:
   rr=requests.get(u,headers={'Range':'bytes=0-1023'},stream=True,timeout=70)
   print('FILE_URL',u,'RANGE_STATUS',rr.status_code,'RANGE_HEADERS',dict(rr.headers),flush=True)
   print('FIRST_BYTES',next(rr.iter_content(chunk_size=128)).hex()[:128],flush=True)
   rr.close()
  except Exception as ex: print('FIRST_BYTES_ERROR',str(ex),flush=True)
