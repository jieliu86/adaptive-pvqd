import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analyze_moment_apvqd import analyze
root=Path(__file__).resolve().parent
local_summary=json.loads((Path(__file__).resolve().parent.parent/'magnus5_t2/summary.json').read_text())['baseline']
nonlocal_summary,nonlocal_raw=analyze(root/'nonlocal')
summaries={'local':local_summary,'nonlocal':nonlocal_summary}
(root/'residual_summary.json').write_text(json.dumps(summaries,indent=2))
with (root/'residual_lags.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['lag','time_lag','pairs','local_C','nonlocal_C','local_rho','nonlocal_rho'],lineterminator='\n');writer.writeheader()
 for a,b in zip(local_summary['correlation'],nonlocal_summary['correlation']):
  writer.writerow(dict(lag=a['lag'],time_lag=a['time_lag'],pairs=a['pairs'],local_C=a['C'],nonlocal_C=b['C'],local_rho=a['rho'],nonlocal_rho=b['rho']))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(11,8))
for name,s in summaries.items():
 lags=s['correlation'];t=[r['time_lag'] for r in lags]
 axs[0,0].plot(t,[r['rho'] for r in lags],label=name)
 axs[0,1].plot(t,[r['C'] for r in lags],label=name)
 axs[1,0].bar(np.arange(3)+(-.18 if name=='local' else .18),[s['sum_residual_norm_squared'],s['cross_terms'],s['vector_error_squared']],width=.36,label=name)
for ax in axs.flat[:2]:ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('Time lag');ax.legend();ax.grid(alpha=.2)
axs[0,0].set_title('Mean normalized pairwise residual correlation')
axs[0,1].set_title('Unnormalized lag correlation C(lag)')
axs[1,0].set_xticks(np.arange(3),['Diagonal sum','Cross terms','Total vector error²']);axs[1,0].legend()
R=np.load(root/'nonlocal/residuals.npz');im=axs[1,1].imshow(R['normalized'],origin='lower',extent=[.025,2.025,.025,2.025],vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=axs[1,1]);axs[1,1].set_title('Nonlocal normalized Gram matrix');axs[1,1].set_xlabel('Step end time');axs[1,1].set_ylabel('Step end time')
fig.tight_layout();fig.savefig(root/'residual_comparison.png',dpi=160)
lines=['# Local / nonlocal transported residual correlation, t=0..2','','Both trajectories use midpoint Suzuki2 reps2, dt=.05, Neel input, identical L-BFGS-B and local-cost acceptance, with no penalties. Local pool has 45 generators; nonlocal pool adds arbitrary-pair XX/YY/ZZ for 108 generators.','','## Definition','','Recursively align each output to the one-step target using the previous aligned output. Define r_n = psi_n - U_n psi_(n-1), and transport r_n back to t=0 with (U_n ... U_1)^dagger. Compute G_ij = Re <rbar_i|rbar_j>. C(lag) is the mean corresponding diagonal of G; rho(lag) is the mean of pairwise normalized inner products, not C(lag)/C(0). No mean subtraction is applied.','','|Metric|Local|Nonlocal|','|---|---:|---:|']
for key in ['global_infidelity','sum_residual_norm_squared','cross_terms','vector_error_squared','amplification','infidelity_to_discrete','reconstruction_error']:
 lines.append(f"|{key}|{local_summary[key]:.9g}|{nonlocal_summary[key]:.9g}|")
lines += ['','The total vector error squared equals the sum of all Gram entries. Its diagonal and off-diagonal decomposition is exact under this convention; it is distinct from phase-invariant infidelity. Reconstruction checks the transported residual sum against the final state error, tolerance 1e-9.','','|Lag time|Pairs|Local rho|Nonlocal rho|Local C|Nonlocal C|','|---:|---:|---:|---:|---:|---:|']
for i in [1,2,5,10,15,20,25,30,35,39]:
 a=local_summary['correlation'][i];b=nonlocal_summary['correlation'][i]
 lines.append(f"|{a['time_lag']:.2f}|{a['pairs']}|{a['rho']:.6f}|{b['rho']:.6f}|{a['C']:.6g}|{b['C']:.6g}|")
reduction=1-nonlocal_summary['cross_terms']/local_summary['cross_terms']
lines += ['',f"Off-diagonal cross-term reduction: {100*reduction:.2f}%. The diagonal residual norm sum changes by {100*(nonlocal_summary['sum_residual_norm_squared']/local_summary['sum_residual_norm_squared']-1):.2f}%.",'','![Residual comparison](residual_comparison.png)','','These are descriptive correlations from one deterministic trajectory per method, not a significance test. Large lags have few pairs. Residuals include optimizer and stopping errors, not certified optimal projection errors. Positive and negative correlations can cancel; a reduced final cross-term sum does not establish statistical independence. Changing the pool also changes selected circuit topology and optimization path.']
(root/'RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({name:{k:v for k,v in s.items() if k!='correlation'} for name,s in summaries.items()},indent=2))
for i in [1,5,10,20,30,39]:print(i,local_summary['correlation'][i]['rho'],nonlocal_summary['correlation'][i]['rho'])
