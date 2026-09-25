import json
import research.tma_gait_ndd_exact as x
x.prefetch_dataset()
for rec in ["control1","park1","hunt1","als1"]:
    cycles,val,ts=x.extract_cycles(rec)
    feat,ev,piv,tree=x.analyze_cycles(cycles)
    print("\n",rec,json.dumps({"validation":val,"features":feat,"first_cycle":cycles[0]},indent=2,default=str))
