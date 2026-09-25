import requests, pathlib, numpy as np, wfdb
BASE="https://physionet.org/files/gaitndd/1.0.0/"
root=pathlib.Path("/tmp/gaitndd"); root.mkdir(exist_ok=True)
for rec in ["park1","control1","hunt1","als1"]:
    for ext in ["hea","let","rit","ts"]:
        p=root/f"{rec}.{ext}"
        r=requests.get(BASE+p.name,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
    rr=wfdb.rdrecord(str(root/rec), physical=True)
    print("\nREC",rec,"fs",rr.fs,"sig",rr.sig_name,"units",rr.units,"shape",rr.p_signal.shape)
    for k,name in enumerate(rr.sig_name):
        x=rr.p_signal[:,k]
        print(name,"minmax",float(np.nanmin(x)),float(np.nanmax(x)),"pct",np.nanpercentile(x,[0,1,5,25,50,75,95,99,100]).tolist())
    ts=np.loadtxt(root/f"{rec}.ts")
    print("TS first row",ts[0].tolist(),"last",ts[-1].tolist(),"n",len(ts))
