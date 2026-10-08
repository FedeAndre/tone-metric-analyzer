#!/usr/bin/env python3
"""One-group EEAHR public dataset pilot: download MD5-verified original archive,
extract only event MAT files, inspect structures and clean timestamps.
Never upload raw EEG or raw participant files.
"""
import hashlib,json,os,time
from pathlib import Path
import requests,rarfile
import scipy.io as sio
import numpy as np
ROOT=Path('/tmp/eeahr');ROOT.mkdir(parents=True,exist_ok=True)
OUT=Path('research/tma_eeahr_results');OUT.mkdir(parents=True,exist_ok=True)
filename='Measurements_IDs_14_15_16.rar'
url='https://zenodo.org/api/records/15412761/files/'+filename+'/content'
EXPECTED=1129184219
EXPECTED_MD5='de601d3b0407d2bafb658bcfa2e25d45'
def get_archive():
    path=ROOT/filename
    if not path.exists() or path.stat().st_size!=EXPECTED:
        session=requests.Session()
        with session.get(url,stream=True,timeout=(45,130)) as r:
            print('DOWNLOADING',r.status_code,r.headers.get('content-length'),flush=True)
            r.raise_for_status();cnt=0
            with path.open('wb') as fh:
                for block in r.iter_content(chunk_size=2**20):
                    if block:
                        fh.write(block);cnt+=len(block)
                        if cnt%(100*2**20)<2**20:print('MEGABYTES',round(cnt/2**20),flush=True)
    assert path.stat().st_size==EXPECTED,('incomplete_download',path.stat().st_size)
    md5=hashlib.md5(path.read_bytes()).hexdigest()
    assert md5==EXPECTED_MD5,('checksum failed',md5)
    print('DOWNLOAD VERIFIED',path.stat().st_size,md5,flush=True)
    return path
def flatten(x,depth=0):
    if depth>=3:return {'type':str(type(x)), 'shape':list(np.shape(x))}
    if isinstance(x,np.ndarray):
        if x.dtype.names:return {'shape':x.shape,'fields':x.dtype.names,'first':flatten(x.reshape(-1)[0],depth+1)}
        if x.size==0:return {'shape':x.shape,'dtype':str(x.dtype)}
        if x.dtype==object:return {'shape':x.shape,'example':flatten(x.reshape(-1)[0],depth+1)}
        return {'shape':x.shape,'dtype':str(x.dtype),'head':np.asarray(x).reshape(-1)[:8].astype(str).tolist()}
    if hasattr(x,'_fieldnames'):return {k:flatten(getattr(x,k),depth+1) for k in x._fieldnames}
    return {'type':str(type(x)),'str':str(x)[:150]}
def main():
    st=time.monotonic()
    arc=get_archive()
    rf=rarfile.RarFile(str(arc))
    members=rf.infolist()
    print('ARCHIVE MEMBERS',len(members),flush=True)
    ev=[x for x in members if '_EVENTS.mat' in x.filename and not x.isdir()]
    print('EVENT FILES',len(ev),'MB',round(sum(x.file_size for x in ev)/2**20,2),flush=True)
    print('SAMPLE EVENTS PATHS',[x.filename for x in ev[:15]],flush=True)
    previews=[]
    for x in ev[:5]:
        print('OPEN EVENT',x.filename,x.file_size,flush=True)
        try:
            import io
            with rf.open(x) as f:
                dat=sio.loadmat(io.BytesIO(f.read()),simplify_cells=True)
            previews.append({'name':x.filename,'mat':{k:flatten(v) for k,v in dat.items() if not k.startswith('__')}})
        except Exception as e:previews.append({'name':x.filename,'error':str(e)})
    results={'archive':filename,'MD5':EXPECTED_MD5,'member_count':len(members),
      'event_files':len(ev),'event_files_total_MiB':sum(x.file_size for x in ev)/2**20,
      'event_paths':[x.filename for x in ev],
      'samples':previews,'seconds':time.monotonic()-st}
    (OUT/'event_archive_inspection.json').write_text(json.dumps(results,indent=2))
    print('INSPECTION',json.dumps({k:v for k,v in results.items() if k!='event_paths'},indent=2),flush=True)
if __name__=='__main__':main()
