#!/usr/bin/env python3
import requests,numpy as np
from scipy.io import loadmat,whosmat
from io import BytesIO
from urllib.parse import quote
for id in ['Ano 01','Nor 01']:
 url='https://gitlab.com/liorg/anosmics-breathe-differently/-/raw/main/Data/'+quote(id)+'.mat'
 try:
  r=requests.get(url,timeout=120)
  print('URL',id,'status',r.status_code,'bytes',len(r.content),'mime',r.headers.get('content-type'),flush=True)
  print('MAT variables',whosmat(BytesIO(r.content)),flush=True)
  d=loadmat(BytesIO(r.content))
  for k,x in d.items():
   if k.startswith('__'):continue
   print(id,'key',k,'shape',getattr(x,'shape',None),'dtype',getattr(x,'dtype',None),
      'preview',str(x.flat[:6] if hasattr(x,'flat') else x)[:150],flush=True)
   if k=='Data':
    a=np.asarray(x,dtype=float)
    print('range',np.percentile(a,[0,1,10,25,50,75,90,99,100],axis=0).tolist(),'nan',np.isnan(a).mean(),'len',len(a),flush=True)
 except Exception as e:print('ERROR',id,repr(e),flush=True)
