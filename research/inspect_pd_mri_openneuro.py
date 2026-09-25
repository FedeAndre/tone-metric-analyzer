#!/usr/bin/env python3
import boto3, json, io, pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
s3=boto3.client("s3",config=Config(signature_version=UNSIGNED),region_name="us-east-1")
for ds in ["ds005892","ds004392","ds005906"]:
    out=[]; token=None
    while True:
        kw={"Bucket":"openneuro.org","Prefix":ds+"/"}
        if token: kw["ContinuationToken"]=token
        r=s3.list_objects_v2(**kw)
        for x in r.get("Contents",[]):
            k=x["Key"]
            if k.endswith("_bold.nii.gz") and "/func/" in k:
                out.append((k,x["Size"]))
        if not r.get("IsTruncated"): break
        token=r["NextContinuationToken"]
    print("\n",ds,"BOLD files",len(out),"GB",sum(x[1] for x in out)/1e9)
    for k,z in out[:10]: print(z,k)
    for small in ["participants.tsv","phenotype/dx_deidentified.tsv","phenotype/hy_deidentified.tsv","phenotype/cognitive_domains.tsv"]:
        key=ds+"/"+small
        try:
            body=s3.get_object(Bucket="openneuro.org",Key=key)["Body"].read().decode()
            print("\n###",key,"\n",body[:2500])
        except Exception as e: pass
