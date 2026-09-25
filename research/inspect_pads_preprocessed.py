import requests,re
base="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
for p in ["preprocessed/movement/","scripts/"]:
 r=requests.get(base+p,timeout=60);print("\n###",p,r.status_code,len(r.text));print(r.text[:12000])
