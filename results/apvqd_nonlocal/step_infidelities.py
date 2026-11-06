import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from moment_statevector import evolution,sequence
root=Path(__file__).resolve().parent
folders={'local':(Path(__file__).resolve().parent.parent/'magnus5_t2/baseline'),'nonlocal':root/'nonlocal'}
runs={}
for name,folder in folders.items():
 data=np.load(folder/'trajectory.npz');states=data['states'];rows=json.loads((folder/'results.json').read_text())['rows']
 records=[]
 for n,row in enumerate(rows):
  ps,ts,_=evolution((n+.5)*.05)
  target=sequence(states[n],ps,ts)
  records.append(dict(time=row['time'],step_infidelity=float(max(0.,1-abs(np.vdot(target,states[n+1]))**2)),global_infidelity=row['global_infidelity'],local_cost=row['local_cost']))
 runs[name]=records
merged=[]
for i in range(40):
 row={'time':runs['local'][i]['time']}
 for name in runs:
  row.update({f'{name}_{k}':v for k,v in runs[name][i].items() if k!='time'})
 merged.append(row)
with (root/'step_infidelities.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(merged[0]),lineterminator='\n');w.writeheader();w.writerows(merged)
(root/'step_infidelities.json').write_text(json.dumps(runs,indent=2))
lines=['# Local / nonlocal per-step infidelity','','Step infidelity = 1 - |<psi_(n+1)| U_midpoint,Suzuki2,reps2 psi_n>|^2. Global infidelity compares each saved state against continuous exact dynamics. Local cost is the original local echo objective and is separately recorded in CSV. No penalties in either run. All numbers below are percentages.','','|t|Local step (%)|Nonlocal step (%)|Local global (%)|Nonlocal global (%)|','|---:|---:|---:|---:|---:|']
for row in merged:
 lines.append(f"|{row['time']:.2f}|{100*row['local_step_infidelity']:.6f}|{100*row['nonlocal_step_infidelity']:.6f}|{100*row['local_global_infidelity']:.4f}|{100*row['nonlocal_global_infidelity']:.4f}|")
(root/'STEP_INFIDELITIES.md').write_text('\n'.join(lines)+'\n')
for name,rows in runs.items():
 x=np.array([r['step_infidelity'] for r in rows]);print(name,'mean',x.mean(),'max',x.max(),'sum',x.sum())
print('\n'.join(lines[4:]))
