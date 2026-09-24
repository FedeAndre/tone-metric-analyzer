#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import tma_cross_domain_frozen as t

DOMAIN=os.environ["TMA_DOMAIN"]
OUT=Path("research/tma_cross_domain_parts")
OUT.mkdir(parents=True,exist_ok=True)
ADAPTERS={
 "gait":t.gait_rows,
 "cardiorespiratory":t.cardio_rows,
 "neural_theta_spike":t.neural_rows,
 "ultrasound":t.oasbud_rows,
 "cmapss":t.cmapss_rows,
 "binary_counter":t.computing_rows,
 "fmri":t.hcp_rows,
}
if DOMAIN not in ADAPTERS:
    raise SystemExit(f"unknown domain {DOMAIN}")
rows=ADAPTERS[DOMAIN]()
if len(rows)<10:
    raise RuntimeError(f"{DOMAIN}: only {len(rows)//2} paired entities")
df=pd.DataFrame(rows)
df.to_csv(OUT/f"{DOMAIN}.csv",index=False)
print(f"{DOMAIN}: wrote {len(df)} sample rows ({len(df)//2} paired entities)",flush=True)
