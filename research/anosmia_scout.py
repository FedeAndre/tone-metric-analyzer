#!/usr/bin/env python3
import requests,json,csv,io
root='https://gitlab.com/api/v4/projects/47366628/repository/tree'
url='https://gitlab.com/liorg/anosmics-breathe-differently/-/raw/main/Data/Participants.csv'
for name,link in [('gitlab_file_index',root+'?path=Data&per_page=100'),('participants',url)]:
 try:
  r=requests.get(link,timeout=30)
  print(name,'status',r.status_code,'size',len(r.content),'preview',r.text[:2000],flush=True)
  if r.ok:
   if name=='gitlab_file_index':
    d=r.json()
    print('paths',[(x['path'],x['type']) for x in d[:80]],flush=True)
    with open('research/anosmia_index.json','w') as f:json.dump(d,f,indent=2)
   else:
    with open('research/anosmia_participants_preview.json','w') as f:json.dump({'header':r.text.splitlines()[0],'sample':r.text.splitlines()[1:5],'nrows':len(r.text.splitlines())},f,indent=2)
 except Exception as e:print(name,'EXC',repr(e),flush=True)
