import requests, json
base="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
paths=[
 "preprocessed/file_list.csv",
 "patients/patient_001.json",
 "movement/observation_001.json",
 "preprocessed/movement/file_list.csv"
]
for p in paths:
    r=requests.get(base+p,timeout=120,headers={"User-Agent":"Mozilla/5.0 TMA-research"})
    print("\n###",p,r.status_code,r.headers.get("content-type"),len(r.content))
    if r.ok: print(r.text[:12000])
