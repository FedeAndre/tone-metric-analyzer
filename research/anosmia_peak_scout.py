#!/usr/bin/env python3
import requests, numpy as np, scipy.signal as ss
from scipy.io import loadmat
from io import BytesIO
from urllib.parse import quote
def load(id):
 url='https://gitlab.com/liorg/anosmics-breathe-differently/-/raw/main/Data/'+quote(id)+'.mat'
 r=requests.get(url,timeout=90);r.raise_for_status()
 d=loadmat(BytesIO(r.content))
 x=np.asarray(d['Data'],float).mean(axis=1)
 samp=float(np.asarray(d['SampleLength']).item()); fs=1/samp
 start=int(d['SleepIndex'].item());wake=int(d['WakeUpIndex'].item())
 ends=[(0,start,'before_sleep'),(wake,len(x),'after_wake')]
 a,b,seg=max(ends,key=lambda t:t[1]-t[0])
 a+=int(120*fs);b-=int(60*fs)
 if b-a<5000:a-=int(60*fs)
 x=x[a:min(b,a+int(30*60*fs))]
 slow=ss.sosfiltfilt(ss.butter(3,[.08,.65],btype='bandpass',fs=fs,output='sos'),x)
 fast=ss.sosfiltfilt(ss.butter(3,[.08,1.8],btype='bandpass',fs=fs,output='sos'),x)
 scale=np.median(abs(slow-np.median(slow)))*1.4826
 major,pp=ss.find_peaks(slow,prominence=.6*scale,distance=int(1.8*fs))
 peaks,qp=ss.find_peaks(fast,prominence=.22*scale,distance=int(.45*fs))
 ma=major[np.r_[True,np.diff(major)>0]]
 durations=np.diff(ma)/fs
 within=[np.sum((peaks>u)&(peaks<v)&(fast[peaks]>.05*scale)) for u,v in zip(ma[:-1],ma[1:])]
 print(id,'segment',seg,'durationmin',len(x)/fs/60,'medianMajorPeriod',np.median(durations),
       'major',len(ma),'sniffExtraCountsPercentiles',np.quantile(within,[0,.25,.5,.75,.9,.99,1]),
       'fractionGe2',np.mean(np.array(within)>=2),'fastSignAtMajor',np.mean(fast[major]),'scale',scale,flush=True)
for code in ["Ano 01","Nor 01","Ano 02","Nor 02","Ano 03","Nor 03","Ano 10","Nor 10"]:
 try:load(code)
 except Exception as e:print('FAIL',code,repr(e),flush=True)
