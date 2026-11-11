"""Plot fixed-population PAC results: stopping, exact label error, and query cost."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); args=p.parse_args()
    data=np.load(args.input_dir/'trajectories.npz'); trials=json.loads((args.input_dir/'trials.json').read_text()); c=json.loads(str(data['metadata_json']))
    fig,axes=plt.subplots(len(c['epsilons']),3,figsize=(15,8),layout='constrained')
    times=np.arange(1,c['n_draws']+1)
    for row,e in enumerate(c['epsilons']):
        ax=axes[row,0]
        for key,label,color in [('perfect_stop','Perfect labels','#67727e'),('pac_stop','PAC labels','#7950ad')]:
            stops=np.array([r[row][key] if r[row][key] is not None else np.inf for r in trials]); finite=np.sort(stops[np.isfinite(stops)])
            ax.step(np.r_[0,finite,times[-1]],np.r_[0,np.arange(1,len(finite)+1)/len(stops),len(finite)/len(stops)],where='post',label=label,color=color)
        ax.set(title=f'ε = {e:g}: stopping',ylabel='Fraction stopped',ylim=(0,1.02));ax.legend()
        ax=axes[row,1]; risk=data[f'{e}_risk']; lo,hi=np.quantile(risk,[.05,.95],axis=0)
        ax.fill_between(times,lo,hi,color='#7950ad',alpha=.2,label='5–95% of trials');ax.plot(times,risk.mean(0),color='#7950ad',label='Mean population error');ax.axhline(e/2,color='black',ls='--',label='Label-error budget ε/2')
        ax.set(title='Exact randomized-label error',ylabel='Error probability',ylim=(0,max(e/2*1.2,hi.max()*1.1)));ax.legend(fontsize=8)
        ax=axes[row,2]; cumulative=np.cumsum(data[f'{e}_query'],axis=1)/times
        ax.plot(times,cumulative.mean(0),color='#7950ad');ax.set(title='Ground-truth queries / draws',ylabel='Cumulative query fraction',ylim=(0,1.05))
        for ax in axes[row]:ax.set_xlabel('Draws');ax.grid(alpha=.2)
    fig.suptitle('Uniform sampling with replacement · 4,961 sessions · 1,000 trials · α = 0.05\nError and query trajectories continue to the diagnostic horizon after stopping',fontsize=12)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(args.output_dir/f'pac_fixed_population.{suffix}',dpi=180)

if __name__=='__main__':main()
