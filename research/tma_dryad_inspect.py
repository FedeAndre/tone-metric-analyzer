#!/usr/bin/env python3
"""Dryad tapping evidence alternative if full 190s EEG archive is unavailable."""
import requests,json,io
import scipy.io as sio
url_options=['https://datadryad.org/downloads/file_stream/4563902',
 'https://datadryad.org/api/v2/files/4563902/download']
for u in url_options:
 try:
  r=requests.get(u,timeout=45)
  print('URL',u,'STATUS',r.status_code,'MIME',r.headers.get('content-type'),'BYTES',len(r.content),flush=True)
  if r.status_code!=200 or len(r.content)<500000:continue
  try:
   d=sio.loadmat(io.BytesIO(r.content),simplify_cells=True)
   print('MAT SUCCESS',json.dumps({k:{'class':str(type(v)),'shape':str(getattr(v,'shape',None)),'keys':list(v.keys()) if isinstance(v,dict) else []} for k,v in d.items() if not k.startswith('__')},indent=2),flush=True)
  except Exception as e:print('MAT ERROR',type(e).__name__,str(e)[:300],flush=True)
  break
 except Exception as e:print('ERROR',u,str(e)[:200],flush=True)
