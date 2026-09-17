#!/usr/bin/env python3
from pathlib import Path
import h5py, json, numpy as np

p=Path("/tmp/session.nwb")
out=Path("research/tma_neural_inventory.txt")
lines=[]
with h5py.File(p,"r") as f:
    def visitor(name,obj):
        if isinstance(obj,h5py.Dataset):
            lines.append(f"DSET {name} shape={obj.shape} dtype={obj.dtype}")
            for k,v in obj.attrs.items():
                try: lines.append(f"  ATTR {k}={v}")
                except: pass
        else:
            lines.append(f"GROUP {name}")
    f.visititems(visitor)
out.write_text("\n".join(lines))
print("\n".join(lines[:1000]))
