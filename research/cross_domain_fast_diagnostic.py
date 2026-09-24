#!/usr/bin/env python3
import requests, json
from pathlib import Path
OUT=Path('research/tma_cross_domain_fast_diagnostic');OUT.mkdir(parents=True,exist_ok=True)
lines=[]
def add(x):
    s=str(x);print(s,flush=True);lines.append(s)
try:
    q=['https://api.osf.io/v2/nodes/7bta5/files/osfstorage/'];seen=set()
    while q:
        u=q.pop(0)
        if u in seen:continue
        seen.add(u)
        r=requests.get(u,timeout=60);add(f'OSF_REQ {u} {r.status_code}')
        j=r.json()
        for it in j.get('data',[]):
            a=it.get('attributes',{});l=it.get('links',{})
            add(json.dumps({'kind':a.get('kind'),'name':a.get('name'),'size':a.get('size'),'download':l.get('download'),'related':l.get('related')},ensure_ascii=False))
            if a.get('kind')=='folder':
                rel=l.get('related')
                if isinstance(rel,dict):rel=rel.get('href')
                if rel:q.append(rel)
        nxt=j.get('links',{}).get('next')
        if nxt:q.append(nxt)
except Exception as e:add('OSF_ERR '+repr(e))
for u in ['https://data.nasa.gov/api/views/ff5v-kuh6','https://data.nasa.gov/api/3/action/package_show?id=cmapss-jet-engine-simulated-data','https://data.nasa.gov/data.json']:
    try:
        r=requests.get(u,timeout=60);add(f'NASA_REQ {u} {r.status_code} {len(r.content)}')
        txt=r.text
        if 'data.json' in u:
            j=r.json();matches=[]
            for d in j.get('dataset',[]):
                title=str(d.get('title','')).lower()
                if 'cmapss' in title or 'c-mapss' in title: matches.append(d)
            add(json.dumps(matches,ensure_ascii=False)[:30000])
        else:add(txt[:30000])
    except Exception as e:add('NASA_ERR '+repr(e))
(OUT/'REPORT.txt').write_text('\n'.join(lines),encoding='utf-8')
