#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, re, tempfile
from pathlib import Path
import pandas as pd
from scipy import stats
from research.tma_mri_exact_utils import S3, BUCKET, math_audit
from research.tma_clinical_exact_utils import bh_adjust
from research.tma_healthy_fmri_ds005256 import preprocess, SURR

DERIV="ds007070"; RAW="ds005256"
OUT=Path("research/tma_healthy_fmri_ds007070_subset_results")
TASKS=["faces","narratives","shortvideo","fractional"]

def list_keys(task):
    pat=re.compile(rf"^ds007070/(sub-[^/]+)/ses-[^/]+/func/(.*_task-{task}_.*_run-01)_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii\.gz$")
    out=[];token=None
    while True:
        kw={"Bucket":BUCKET,"Prefix":DERIV+"/"}
        if token:kw["ContinuationToken"]=token
        r=S3.list_objects_v2(**kw)
        for x in r.get("Contents",[]):
            if pat.match(x["Key"]):out.append(x["Key"])
        if not r.get("IsTruncated"):break
        token=r["NextContinuationToken"]
    return sorted(out)

def temp(key):
    fd,p=tempfile.mkstemp(suffix=".nii.gz");os.close(fd);S3.download_file(BUCKET,key,p);return Path(p)

def raw_events_key(k):
    # derivative basename prefix before _space-... is the raw BIDS stem
    m=re.match(r"ds007070/(sub-[^/]+)/(ses-[^/]+)/func/(.+)_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii\.gz$",k)
    if not m:raise RuntimeError(k)
    sub,ses,stem=m.groups()
    return f"{RAW}/{sub}/{ses}/func/{stem}_events.tsv"

def read_events(k):
    b=S3.get_object(Bucket=BUCKET,Key=raw_events_key(k))["Body"].read()
    return pd.read_csv(io.BytesIO(b),sep="\t")

def run(task):
    OUT.mkdir(parents=True,exist_ok=True);rows=[];fails={}
    ks=list_keys(task)
    for i,k in enumerate(ks,1):
        print(f"[{i}/{len(ks)}] {k}",flush=True);p=None
        try:
            p=temp(k);f=preprocess(p,read_events(k),task,k)
            sub=re.search(r"/(sub-[^/]+)/",k).group(1);rows.append({"subject":sub,"task":task,"key":k,**f})
        except Exception as e:fails[k]=repr(e);print("FAIL",repr(e),flush=True)
        finally:
            if p and p.exists():p.unlink()
    pd.DataFrame(rows).to_csv(OUT/f"{task}_features.csv",index=False)
    (OUT/f"{task}_failures.json").write_text(json.dumps(fails,indent=2))
    print({"task":task,"success":len(rows),"failures":len(fails)})

def aggregate():
    fs=sorted(OUT.glob("*_features.csv"));d=pd.concat([pd.read_csv(p) for p in fs],ignore_index=True)
    d.to_csv(OUT/"all.csv",index=False); rows=[]
    for task,g in d.groupby("task"):
        ps=[];tmp=[]
        for c in SURR:
            x=g[f"surz_{c}"].dropna()
            t=stats.ttest_1samp(x,0)
            tmp.append({"task":task,"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(t.statistic),"p":float(t.pvalue)})
            ps.append(float(t.pvalue))
        for r,q in zip(tmp,bh_adjust(ps)):r["q_BH"]=float(q);rows.append(r)
    tab=pd.DataFrame(rows);tab.to_csv(OUT/"tasklocking.csv",index=False)
    counts=d.groupby("subject").task.nunique();ids=counts[counts==len(TASKS)].index
    z=d[d.subject.isin(ids)].groupby("subject",as_index=False).mean(numeric_only=True)
    br=[];ps=[]
    for c in SURR:
        x=z[f"surz_{c}"].dropna();t=stats.ttest_1samp(x,0)
        br.append({"feature":c,"n":len(x),"mean_z":float(x.mean()),"t":float(t.statistic),"p":float(t.pvalue)});ps.append(float(t.pvalue))
    for r,q in zip(br,bh_adjust(ps)):r["q_BH"]=float(q)
    broad=pd.DataFrame(br);broad.to_csv(OUT/"across_tasks.csv",index=False)
    summary={"dataset":DERIV,"source":RAW,"math_audit":math_audit(),"n_scans":int(len(d)),"n_subjects":int(d.subject.nunique()),"n_complete":int(len(ids)),"task_results":tab.sort_values(["task","p"]).to_dict(orient="records"),"across_tasks":broad.sort_values("p").to_dict(orient="records")}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=TASKS);p.add_argument("--aggregate",action="store_true");a=p.parse_args()
    aggregate() if a.aggregate else run(a.task)
