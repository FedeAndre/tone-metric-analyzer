#!/usr/bin/env python3
import requests, pandas as pd, io, json, tarfile, zipfile, os, re
from pathlib import Path
OUT=Path('research/tma_cross_domain_schema');OUT.mkdir(exist_ok=True)
log=[]
def add(x): s=str(x); print(s,flush=True); log.append(s)
# Speech tables
for name,url in [('individual','https://osf.io/download/n48hp/'),('cues','https://osf.io/download/35ukt/')]:
    try:
        r=requests.get(url,timeout=120);r.raise_for_status();add(f'{name} size {len(r.content)} final {r.url}')
        if name=='individual': df=pd.read_csv(io.BytesIO(r.content))
        else: df=pd.read_excel(io.BytesIO(r.content))
        add(f'{name} shape={df.shape} columns={list(df.columns)}')
        add(df.head(8).to_string())
    except Exception as e:add(f'{name} ERR {e!r}')
# NASA ZIP listing
try:
    r=requests.get('https://data.nasa.gov/docs/legacy/CMAPSSData.zip',timeout=180);r.raise_for_status()
    add(f'NASA zip size={len(r.content)}')
    z=zipfile.ZipFile(io.BytesIO(r.content));add('NASA files='+repr(z.namelist()))
    for n in z.namelist():
        if 'train_FD001' in n:
            raw=z.read(n).decode('utf-8','replace').splitlines()[:3]; add(n+' first='+repr(raw))
except Exception as e:add('NASA ZIP ERR '+repr(e))
# HCP HEAD/partial
for url in ['https://osf.io/s4h8j/download/','https://osf.io/2y3fw/download/','https://files.de-1.osf.io/v1/resources/54w3g/providers/osfstorage/60e80c2bf80fdb01334d9147']:
    try:
        r=requests.get(url,headers={'User-Agent':'Mozilla/5.0','Range':'bytes=0-1023'},timeout=90,allow_redirects=True)
        add(f'HCP {url} status={r.status_code} len={len(r.content)} final={r.url} headers='+repr(dict(r.headers)))
    except Exception as e:add('HCP ERR '+repr(e))
(OUT/'REPORT.txt').write_text('\n'.join(log),encoding='utf-8')
