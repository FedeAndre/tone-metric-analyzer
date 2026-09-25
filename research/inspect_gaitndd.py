import requests
base="https://physionet.org/files/gaitndd/1.0.0/"
for name in ["park1.hea","park1.ts","subject-description.txt","RECORDS"]:
    r=requests.get(base+name,timeout=60); r.raise_for_status()
    print("\n###",name,"bytes",len(r.content))
    if name.endswith((".hea",".txt")) or name=="RECORDS":
        print(r.text[:5000])
    elif name.endswith(".ts"):
        print(r.text[:2000])
