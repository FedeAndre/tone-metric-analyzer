import io,zipfile,requests,json,re,tarfile
from pathlib import Path
BASEP="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
BASEQ="https://physionet.org/files/nqmitcsxpd/1.0.0/"
BASEG="https://physionet.org/files/gaitdb/1.0.0/"
print("PADS patient 001")
for p in ["patients/patient_001.json","movement/observation_001.json","movement/timeseries/001_TouchNose_LeftWrist.txt"]:
    r=requests.get(BASEP+p,timeout=120); print(p,r.status_code,len(r.content)); print(r.text[:5000])
print("\nNEUROQWERTY ZIP")
r=requests.get(BASEQ+"neuroQWERTY.zip",timeout=120); print(r.status_code,len(r.content))
z=zipfile.ZipFile(io.BytesIO(r.content))
for n in z.namelist()[:80]: print(n)
for n in z.namelist():
    if n.endswith("GT_DataPD_MIT-CSXPD.csv"):
        print("\nGT",n);print(z.read(n).decode(errors="replace")[:5000]);break
for n in z.namelist():
    if n.lower().endswith(".csv") and "GT_Data" not in n:
        print("\nDATA",n);print(z.read(n).decode(errors="replace")[:3000]);break
print("\nGAITDB")
r=requests.get(BASEG,timeout=60); print(r.status_code); print(r.text[:8000])
for f in ["pd1.txt","pd01.txt","pd1.str","old1.txt","y23.txt"]:
    rr=requests.get(BASEG+f,timeout=60); print(f,rr.status_code,len(rr.content),rr.text[:500] if rr.status_code==200 else "")
print("\nDAPHNET")
u="https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
r=requests.get(u,timeout=180); print(r.status_code,len(r.content))
z2=zipfile.ZipFile(io.BytesIO(r.content)); print(z2.namelist()[:50])
for n in z2.namelist():
    if n.lower().endswith(".txt"):
        print(n);print(z2.read(n).decode(errors="replace")[:2500]);break
