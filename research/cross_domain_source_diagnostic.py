#!/usr/bin/env python3
import json, os, tarfile, zipfile, io
from pathlib import Path
import requests

OUT=Path('research/tma_cross_domain_diagnostic')
OUT.mkdir(parents=True, exist_ok=True)
report=[]

def add(s):
    print(s, flush=True); report.append(str(s))

# OSF speech node recursive file inventory
try:
    url='https://api.osf.io/v2/nodes/7bta5/files/osfstorage/'
    seen=[]
    q=[url]
    visited=set()
    while q:
        u=q.pop(0)
        if u in visited: continue
        visited.add(u)
        j=requests.get(u,timeout=60).json()
        for item in j.get('data',[]):
            a=item.get('attributes',{}); l=item.get('links',{})
            seen.append((a.get('kind'),a.get('name'),l.get('download')))
            if a.get('kind')=='folder':
                rel=l.get('related')
                if isinstance(rel,dict): rel=rel.get('href')
                if rel: q.append(rel)
        nxt=j.get('links',{}).get('next')
        if nxt: q.append(nxt)
    add('OSF 7bta5 files:')
    for row in seen: add(row)
except Exception as e: add('OSF_ERROR '+repr(e))

# NASA metadata
for u in [
    'https://data.nasa.gov/api/3/action/package_show?id=cmapss-jet-engine-simulated-data',
    'https://data.nasa.gov/api/views/ff5v-kuh6',
]:
    try:
        r=requests.get(u,timeout=60)
        add(f'NASA {u} status={r.status_code} ct={r.headers.get("content-type")} len={len(r.content)}')
        add(r.text[:12000])
    except Exception as e: add('NASA_ERROR '+repr(e))

# HCP archive structure
try:
    u='https://osf.io/2y3fw/download'
    r=requests.get(u,timeout=240)
    add(f'HCP status={r.status_code} len={len(r.content)} final={r.url}')
    if r.ok and len(r.content)>1000:
        p=OUT/'hcp_task.tgz'; p.write_bytes(r.content)
        with tarfile.open(p) as tf:
            names=tf.getnames()
        add('HCP first entries: '+json.dumps(names[:200]))
        p.unlink()
except Exception as e: add('HCP_ERROR '+repr(e))

# OASBUD structure
try:
    import scipy.io
    u='https://zenodo.org/record/545928/files/OASBUD.mat?download=1'
    p=OUT/'OASBUD.mat'
    with requests.get(u,stream=True,timeout=240) as r:
        r.raise_for_status()
        with p.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    add(f'OASBUD size={p.stat().st_size}')
    m=scipy.io.loadmat(p,squeeze_me=True,struct_as_record=False)
    add('OASBUD keys='+repr([k for k in m if not k.startswith('__')]))
    d=m.get('data')
    add('data type='+repr(type(d))+' shape='+repr(getattr(d,'shape',None)))
    if d is not None:
        first=d.flat[0] if hasattr(d,'flat') else d[0]
        add('first fields='+repr(getattr(first,'_fieldnames',None)))
        for nm in getattr(first,'_fieldnames',[]) or []:
            v=getattr(first,nm)
            add(f' field {nm}: type={type(v)} shape={getattr(v,"shape",None)} dtype={getattr(v,"dtype",None)} sample={repr(v)[:200]}')
    p.unlink()
except Exception as e: add('OASBUD_ERROR '+repr(e))

(OUT/'REPORT.txt').write_text('\n'.join(report),encoding='utf-8')
