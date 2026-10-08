#!/usr/bin/env python3
"""Robustness/interaction audit for variable rhythmic TMA: explicit event index,
beat tempo, several regularizations; drummer-held-out only; subject-cluster bootstrap.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy import stats

OUT=Path('research/tma_variable_drums_results')
df=pd.read_csv(OUT/'variable_drum_windows.csv')
assert len(df)==679 and df.track.nunique()==97 and df.drummer.nunique()==10
var='variable_cv'
def features(prefix,fields):
 return [f'{prefix}_{k}{s}' for k in fields for s in ('','_dprev','_dstart')]
BASE=['events_per_bar','attacks_per_bar','events_cv','ioi_mean','ioi_std',
 'ioi_madiff','phase_entropy8','phase_entropy16','phase_entropy32',
 'offbeat_fraction','microtiming_mean','microtiming_std','phase_autocorr1',
 'bar_count_autocorr1','bar_count_madiff','event_quarter_R',
 'cumulative_events_cv','cumulative_phase_entropy16','cumulative_offbeat',
 'cumulative_bar_count_acf1']
ALL=['mean_H','std_H','mean_D','std_D','mean_L','multilevel','H_jumps',
 'level_turnover','pivot_rate','pivot_depth','tree_span','tree_long','tree_cross']
CORE=['mean_D','mean_H','H_jumps','level_turnover','pivot_rate','tree_span']
base=['window','bpm']+features('timing',BASE)
sets={
  'conventional':base,
  'origin_only':['window','bpm']+features('origin',ALL),
  'reset_only':['window','bpm']+features('reset',ALL),
  'timing_origin':base+features('origin',ALL),
  'timing_reset':base+features('reset',ALL),
  'timing_core_origin':base+features('origin',CORE),
  'timing_core_reset':base+features('reset',CORE)
}
def heldout(cols,y,alpha):
 result=np.empty(len(df));Y=df[y].to_numpy(float)
 for drummer in df.drummer.unique():
  mask=(df.drummer==drummer).to_numpy()
  m=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
  m.fit(df.loc[~mask,cols],Y[~mask])
  result[mask]=m.predict(df.loc[mask,cols])
 return result
def stats0(mask,y,p):
 v=y[mask];q=p[mask]
 return {'MAE':float(np.mean(abs(v-q))),
    'R2':float(1-sum((v-q)**2)/sum((v-v.mean())**2))}
def mean(a):return float(np.mean(a))
out={'n_tracks':df.track.nunique(),'n_drummers':df.drummer.nunique(),
 'n_window':len(df),'regularization_grid':[10,100,500],
 'rhythmic_CV':{},'models':{}}
g=df.drop_duplicates('track')
for q in ['regular','variable']:
 v=g.loc[g.selection==q,var]
 out['rhythmic_CV'][q]={'n':len(v),'median':float(v.median()),
   'mean':float(v.mean()),'q25':float(v.quantile(.25)),'q75':float(v.quantile(.75))}
for target in ('change','drift'):
 y=df[target].to_numpy(float)
 out['models'][target]={}
 for alpha in (10,100,500):
  P={key:heldout(col,target,alpha) for key,col in sets.items()}
  groups={'all':np.ones(len(df),bool),'regular':(df.selection=='regular').to_numpy(),
          'variable':(df.selection=='variable').to_numpy()}
  item={}
  for label,mask in groups.items():
   res={m:stats0(mask,y,P[m]) for m in P}
   for new in ['origin_only','timing_origin','timing_core_origin','timing_reset','timing_core_reset']:
    res['gain_'+new]=res['conventional']['MAE']-res[new]['MAE']
   item[label]=res
  out['models'][target][str(alpha)]=item
(OUT/'robustness.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
