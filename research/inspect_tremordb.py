import io, zipfile, requests, pathlib, numpy as np, re
url="https://physionet.org/content/tremordb/get-zip/1.0.0/"
r=requests.get(url,timeout=180,headers={"User-Agent":"Mozilla/5.0 TMA-research"}); r.raise_for_status()
z=zipfile.ZipFile(io.BytesIO(r.content))
print("files",len(z.namelist()))
for name in z.namelist()[:120]:
    print(name)
for target in ["RECORDS","subject_description.txt","file_description.txt","README"]:
    matches=[n for n in z.namelist() if n.endswith("/"+target) or n==target]
    print("\n###",target)
    if matches: print(z.read(matches[0]).decode(errors="replace")[:12000])
# inspect first signal files
for n in [x for x in z.namelist() if re.search(r'\.(let|rit)$',x)][:5]:
    raw=z.read(n)
    txt=raw.decode(errors="replace").splitlines()
    print("\nSIG",n,"bytes",len(raw),"lines",len(txt),"first",txt[:10])
