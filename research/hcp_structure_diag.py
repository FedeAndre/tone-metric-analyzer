#!/usr/bin/env python3
import tarfile, requests
from pathlib import Path
p=Path('/tmp/hcp_task.tgz')
u='https://files.de-1.osf.io/v1/resources/54w3g/providers/osfstorage/60e80c2bf80fdb01334d9147'
with requests.get(u,stream=True,timeout=1200,allow_redirects=True) as r:
    r.raise_for_status()
    with p.open('wb') as f:
        for c in r.iter_content(1024*1024):
            if c:f.write(c)
with tarfile.open(p,'r:gz') as tf:
    names=tf.getnames()
    sl=[n for n in names if n.endswith('subjects_list.txt')][0]
    ids=tf.extractfile(sl).read().decode().split()
    print('SUBJECTS_LIST_PATH',sl)
    print('FIRST_IDS',ids[:5])
    sid=ids[0].strip()
    matches=[n for n in names if f'/subjects/{sid}/' in '/'+n]
    print('FIRST_SUBJECT_MATCHES',len(matches))
    for n in matches[:120]: print(n)
