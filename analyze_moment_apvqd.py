"""Postprocess the two t<=2 branches, including transported residuals."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from moment_statevector import sequence,evolution,initial


def analyze(folder):
    result=json.loads((folder/'results.json').read_text())
    data=np.load(folder/'trajectory.npz');states=data['states'];times=data['times']
    propagated=initial.copy();prior=initial.copy();transported=[];gates=[]
    for j in range(1,len(states)):
        p,angles,_=evolution((times[j]+times[j-1])/2);gates.append((p,angles))
        target=sequence(prior,p,angles)
        overlap=np.vdot(target,states[j]);current=states[j]*overlap.conjugate()/abs(overlap)
        r=current-target
        for ps,ts in gates[::-1]:r=sequence(r,ps[::-1],-ts[::-1])
        transported.append(r);prior=current
        propagated=sequence(propagated,p,angles)
    R=np.array(transported);gram=np.real(R.conj()@R.T);norms=np.sqrt(np.diag(gram))
    normalized=np.divide(gram,norms[:,None]*norms[None,:],out=np.zeros_like(gram),where=norms[:,None]*norms[None,:]>1e-25)
    reconstruction=sequence(R.sum(axis=0),[],[])
    for p,ts in gates:reconstruction=sequence(reconstruction,p,ts)
    discrepancy=float(np.linalg.norm(reconstruction-(prior-propagated)))
    assert discrepancy<1e-9,discrepancy
    lagrows=[dict(lag=l,time_lag=.05*l,pairs=len(R)-l,C=float(np.diag(gram,l).mean()),rho=float(np.diag(normalized,l).mean())) for l in range(len(R))]
    with (folder/'correlation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(lagrows[0]));writer.writeheader();writer.writerows(lagrows)
    np.savez(folder/'residuals.npz',transported=R,gram=gram,normalized=normalized)
    last=result['rows'][-1];diag=float(np.trace(gram));total=float(gram.sum())
    summary=dict(completed=len(R)==40,final_time=float(times[-1]),global_infidelity=last['global_infidelity'],parameters=last['parameters'],raw_cx=last['raw_cx'],sum_residual_norm_squared=diag,cross_terms=total-diag,vector_error_squared=total,amplification=total/diag,reconstruction_error=discrepancy,infidelity_to_discrete=float(1-abs(np.vdot(prior,propagated))**2),max_abs_energy=float(max(abs(x['residual_energy']) for x in result['rows'])),max_abs_variance=float(max(abs(x['residual_variance']) for x in result['rows'])),optimizer_iterations=sum(h['iterations'] for r in result['rows'] for h in r['history']),elapsed_seconds=result['elapsed'],correlation=lagrows)
    return summary,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    summaries={};raw={}
    for name in ['baseline','penalty','penalty_loose']:
        if (args.root/name/'status.json').exists() and json.loads((args.root/name/'status.json').read_text())['completed']:
            summaries[name],raw[name]=analyze(args.root/name)
    (args.root/'summary.json').write_text(json.dumps(summaries,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for name in summaries:
        rows=raw[name]['rows'];t=[x['time'] for x in rows];s=summaries[name]
        axs[0,0].plot(t,[x['global_infidelity'] for x in rows],label=name)
        axs[0,1].plot(t,[x['residual_variance'] for x in rows],label=name)
        axs[1,0].plot([x['time_lag'] for x in s['correlation']],[x['rho'] for x in s['correlation']],label=name)
        axs[1,1].plot(t,[x['parameters'] for x in rows],label=name)
    for ax,title in zip(axs.flat,['Global infidelity','Signed cumulative variance defect','Transported residual correlation','Variational parameters']):
        ax.set_title(title);ax.set_xlabel('Time / time lag');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.root/'comparison.png',dpi=160)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='correlation'} for k,v in summaries.items()},indent=2))

if __name__=='__main__':main()
