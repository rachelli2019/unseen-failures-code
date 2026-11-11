"""Focused paper figure: bounds/coverage over sessions and audit stopping cost."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = [('perfect', 'Perfect labels', '#a52aa5'),
        ('bpac', 'IPW B-PAC', '#255c99'), ('ppi', 'PPI', '#ce791d')]


def audit_panel(ax, summary, epsilons, trials, rows):
    # Compare whole trial sets, never just the trials that happened to stop.
    # Omit a threshold for ALL arms if any arm has unfinished trials there.
    complete = [e for e in epsilons if all(
        next(r['censored'] for r in summary if r['method']==m and r['epsilon']==e)==0
        for m,_,_ in ARMS)]
    if not complete:
        raise ValueError('no common fully observed stopping thresholds')
    for method, label, color in ARMS:
        values = [next(r['mean_audits'] for r in summary if r['method']==method and r['epsilon']==e)
                  for e in complete]
        cost_label = r'IPW B-PAC ($\zeta=\varepsilon/2$)' if method=='bpac' else label
        # Empirical trial percentiles describe variability, not uncertainty in the mean.
        samples = [[r['audits'] for r in rows if r['method']==method and r['epsilon']==e]
                   for e in complete]
        if any(len(v)!=trials for v in samples):
            raise ValueError('incomplete stopping-label trial set')
        low, high = np.quantile(samples, [.025,.975], axis=1)
        ax.errorbar(complete, values, yerr=[np.array(values)-low, high-np.array(values)],
                    fmt='o-', color=color, lw=2, markersize=5, capsize=3,
                    elinewidth=1, label=cost_label)
    ax.set(title='Ground-truth labels used',xlabel=r'Stopping threshold $\varepsilon$',
           ylabel='Ground-truth labels collected through stopping',ylim=(0,None))
    ax.set_xticks(complete)
    ax.legend(frameon=False,fontsize=10)
    ax.grid(alpha=.15)


def main():
    c=json.loads(Path('configs/run_imperfect_labels_final.json').read_text())
    root=Path(c['output_dir']);a=np.load(root/'trajectories.npz');times=a['times']
    summary=json.loads((root/'completed_stopping_summary.json').read_text())
    rows=[json.loads(line) for line in (root/'completed_stopping_trials.jsonl').read_text().splitlines()]
    out=Path('data/plots/imperfect_labels')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    # Perfect/PPI do not depend on epsilon: verify before plotting them once.
    for method in ['perfect','ppi']:
        for field in ['upper','mass','coverage']:
            np.testing.assert_array_equal(a[f'{method}_0.1__{field}'],a[f'{method}_0.15__{field}'])
    curves = [
        ('perfect', .1, 'Perfect labels', '#a52aa5'),
        ('ppi', .1, 'PPI', '#ce791d'),
        ('bpac', .1, r'IPW B-PAC: $\zeta=0.05,\ \delta=0.025$', '#255c99'),
    ]
    fig,axes=plt.subplots(1,3,figsize=(18,5.2))
    for method,eps,label,color in curves:
        upper=a[f'{method}_{eps}__upper']
        low,high=np.quantile(upper,[.025,.975],axis=0)
        axes[0].fill_between(times,low,high,color=color,alpha=.15,linewidth=0)
        axes[0].plot(times,upper.mean(0),color=color,label=label,lw=2.2)
    for method,label,color in ARMS:
        label = r'IPW B-PAC ($\zeta=0.05$)' if method=='bpac' else label
        axes[1].plot(times,a[f'{method}_0.1__coverage'].mean(0),color=color,lw=2.2,label=label)
    axes[0].set(title='Mean upper bound',xlabel='Sampled sessions',
                ylabel='Unseen failure probability',ylim=(0,.35),xlim=(0,c['n_draws']))
    axes[0].grid(alpha=.18);axes[0].legend(frameon=False,fontsize=9)
    axes[1].axhline(1-c['alpha'],color='black',ls=':',lw=1)
    axes[1].set(title='Simultaneous coverage',xlabel='Sampled sessions',ylabel='Coverage probability',
                ylim=(.94,1.005),xlim=(0,c['n_draws']))
    axes[1].grid(alpha=.18)
    axes[1].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=9,loc='lower right',bbox_to_anchor=(1, .18))
    audit_panel(axes[2],summary,c['epsilons'],c['n_trials'],rows)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'paper_comparison.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.5))
    audit_panel(ax,summary,c['epsilons'],c['n_trials'],rows);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'audit_cost.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    caption=(
        'Imperfect failure labeling with Q=P. Left: mean missing-mass upper bounds '
        'for perfect labels, PPI, and B-PAC with fixed zeta=0.05 and delta=0.025. Perfect/PPI trajectories do not depend on the '
        'stopping threshold and are shown once. Shading shows pointwise 2.5th–97.5th '
        'percentiles across trials. '
        'Center: empirical simultaneous coverage through each sample count; '
        'the nominal level is 0.95. B-PAC uses the same fixed zeta=0.05 and '
        'delta=0.025 as in the upper-bound panel. Right: mean cumulative ground-truth label '
        'calls through stopping, including all forced B-PAC queries. Error bars show 2.5th–97.5th percentiles across trials. These ranges describe trial variability, not confidence intervals for the mean or simultaneous bands. Only thresholds '
        'where all 1,000 trials stopped in every method are shown. The 273 PPI '
        'trials unfinished at 10,000 draws were continued to stopping (last stop '
        'at 14,858 draws); costs include their full trajectories, not just the '
        'original horizon or the initially completed subset. In the cost sweep '
        'B-PAC uses zeta=epsilon/2 and delta=0.025. Categorizer calls are reported '
        'separately in the results table. Results use '
        '1,000 paired trials with a 10,000-session coverage horizon.\n')
    (out/'paper_caption.md').write_text(caption)
    print(out/'paper_comparison.png')


if __name__=='__main__':main()
