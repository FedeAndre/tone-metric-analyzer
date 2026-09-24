#!/usr/bin/env python3
import tarfile, requests, time
from pathlib import Path

urls=[
 'https://osf.io/s4h8j/download/',
 'https://files.de-1.osf.io/v1/resources/54w3g/providers/osfstorage/60e80c2bf80fdb01334d9147',
]
p=Path('/tmp/hcp_task.tgz')
last=None
for u in urls:
    for attempt in range(6):
        try:
            with requests.get(u,stream=True,timeout=1200,allow_redirects=True,
                              headers={'User-Agent':'Mozilla/5.0 TMA-research'}) as r:
                if r.status_code==429:
                    raise RuntimeError('HTTP 429')
                r.raise_for_status()
                with p.open('wb') as f:
                    for c in r.iter_content(1024*1024):
                        if c:f.write(c)
            if p.stat().st_size<1000000: raise RuntimeError('short download')
            print('DOWNLOADED',u,p.stat().st_size,flush=True)
            last=None
            break
        except Exception as e:
            last=e
            print('RETRY',u,attempt+1,repr(e),flush=True)
            time.sleep(5*(attempt+1))
    if last is None: break
if last is not None: raise last

with tarfile.open(p,'r:gz') as tf:
    names=tf.getnames()
    print('NAMES_TOTAL',len(names))
    sl=[n for n in names if n.endswith('subjects_list.txt')][0]
    ids=tf.extractfile(sl).read().decode().split()
    sid=ids[0].strip()
    print('SUBJECTS_LIST_PATH',sl)
    print('FIRST_IDS',ids[:5])
    print('FIRST_120_PATHS')
    for n in names[:120]: print(n)
    print('SUBJECT_ID_MATCHES',sid)
    matches=[n for n in names if sid in n]
    print('COUNT',len(matches))
    for n in matches[:200]: print(n)
    print('MOTOR_MATCHES')
    motor=[n for n in names if 'MOTOR' in n.upper()]
    print('COUNT',len(motor))
    for n in motor[:200]: print(n)
    print('BOLD_MATCHES')
    bold=[n for n in names if 'bold5' in n.lower() or 'bold6' in n.lower()]
    print('COUNT',len(bold))
    for n in bold[:120]: print(n)
