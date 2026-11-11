#!/usr/bin/env python3
"""Three-row P/Q figure from saved bounds, exact stopping times, and coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plot_fixed_q_primary import plt, style, paper_style, bounds_panel, finish_axes, ARM_COLORS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--stopping-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--oracle-interior',action='store_true',help='Require 1 < oracle time <= horizon for every trial in both arms')
    args=parser.parse_args()
    summary=json.loads((args.run_dir/'summary.json').read_text())
    stopping_summary=json.loads((args.stopping_dir/'summary.json').read_text())
    if Path(stopping_summary['source_run']).resolve()!=args.run_dir.resolve():
        raise ValueError('Stopping results come from another source run')
    horizon=summary['config']['horizon']; stop_horizon=stopping_summary['horizon']
    sources=[args.run_dir/'summary.json',args.stopping_dir/'summary.json']
    arrays={}; stops={}; oracle={}
    for arm in ('P','Q'):
        source=args.run_dir/f'{arm}_trajectories.npz'; sources.append(source)
        with np.load(source) as data:
            arrays[arm]={k:data[k] for k in ('true_missing_mass','upper','simultaneous_covered')}
        source=args.stopping_dir/f'{arm}_stopping.npz'; sources.append(source)
        with np.load(source) as data:
            stops[arm]=data['stopping_times']
            oracle[arm]=data['oracle_stopping_times']
            eps=data['epsilons']
            np.testing.assert_array_equal(eps,stopping_summary['epsilons'])
    if any(np.any(s > stop_horizon) for s in list(stops.values()) + list(oracle.values())):
        raise ValueError('Certification-delay means require uncensored stopping times')
    eligible = np.ones(len(eps),dtype=bool)
    if args.oracle_interior:
        for arm in ('P','Q'):
            eligible &= np.all((oracle[arm] > 1) & (oracle[arm] <= stop_horizon),axis=0)
    order = np.flatnonzero(eligible)[np.argsort(eps[eligible])]
    if len(order) < 2:
        raise ValueError('Fewer than two thresholds satisfy the oracle support restriction')
    eps = eps[order]
    times = {}
    for arm in ('P','Q'):
        times[arm] = {}
        for name, source in [('ours',stops),('oracle',oracle)]:
            samples = source[arm][:,order]
            times[arm][name] = {'mean': samples.mean(axis=0).tolist(),
                               'p025': np.quantile(samples,.025,axis=0).tolist(),
                               'p975': np.quantile(samples,.975,axis=0).tolist()}
    if stops['P'].shape != stops['Q'].shape:
        raise ValueError('P and Q trials must align for paired additional stopping cost')
    additional = (stops['Q'] - stops['P'])[:,order]
    cost = {'mean':additional.mean(0).tolist(),
            'p025':np.quantile(additional,.025,axis=0).tolist(),
            'p975':np.quantile(additional,.975,axis=0).tolist()}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with paper_style():
        fig,axes=plt.subplots(2,2,figsize=(18,12.6))
        for col,arm in enumerate(('P','Q')):
            a=arrays[arm]
            bounds_panel(axes[0,col],arm,a['true_missing_mass'],a['upper'],.95)
        ax=axes[1,0]
        for arm in ('P','Q'):
            rates=arrays[arm]['simultaneous_covered'].mean(axis=0)
            ax.plot(np.arange(1,horizon+1),rates,color=ARM_COLORS[arm],linewidth=3.2,
                    label=f'Sampling from ${arm}$')
        nominal=1-summary['config']['alpha']
        ax.axhline(nominal,color=style.NOMINAL_COVERAGE_COLOR,linewidth=1.6,linestyle=(0,(2,2)))
        ax.annotate(r'nominal $1-\alpha$',xy=(horizon,nominal),xytext=(-8,6),
                    textcoords='offset points',ha='right',va='bottom',color=style.NOMINAL_COVERAGE_COLOR)
        finish_axes(ax,horizon); ax.set_ylim(.94,1.002)
        ax.set_yticks(np.arange(.94,1.001,.02))
        ax.set_ylabel('Simultaneous coverage')
        ax.set_title('Coverage',pad=16)
        ax.legend(loc='upper right',frameon=False)
        ax=axes[1,1]
        ax.plot(eps,cost['mean'],color=ARM_COLORS['Q'],linewidth=3.2,label=r'Mean $\tau_Q-\tau_P$')
        ax.axhline(0,color='0.45',linewidth=1.6,linestyle=(0,(2,2)))
        ax.set_xlim(float(eps.min()),float(eps.max()))
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel(r'Stopping threshold, $\varepsilon$')
        ax.set_ylabel(r'Additional samples, $\tau_Q-\tau_P$')
        ax.set_title('Additional stopping cost of $Q$',pad=16)
        ax.grid(True,alpha=.2,linewidth=.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right',frameon=False)
        fig.tight_layout(h_pad=2.4,w_pad=2.0)
        style.save_figure(fig,args.output_dir,'primary_fixed_P_Q_grid')
        plt.close(fig)
    manifest={'script':str(Path(__file__).resolve()),'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'colors':ARM_COLORS,'panels':['P sampling: bound and P-reference missing mass','Q sampling: bound and P-reference missing mass','overlaid simultaneous coverage','paired additional stopping cost'],
              'additional_stopping_cost':cost,'additional_cost_display':'mean only; no ribbon','pairing':'Same trial index / child seed; percentile ribbon describes paired differences, not a confidence interval for the mean',
              'columns':['P','Q'],'epsilons':eps.tolist(),'coverage_horizon':horizon,'stopping_horizon':stop_horizon,
              'stopping_time_summary':times,'stopping_time_definition':'Separate first strict crossings of bound and true missing mass at t >= 1',
              'oracle_interior_required':args.oracle_interior,'oracle_filter':'all trials in both arms; intersect original threshold domain; no trial filtering',
              'stopping_time_band_quantiles':[.025,.975],'central_band':.95,
              'note':'Bounds/coverage use original Q lambda grid (max epsilon .10); stopping uses updated Q grid (max epsilon .50). Dense stopping experiment rerun; upper-bound and coverage arrays unchanged.'}
    (args.output_dir/'primary_fixed_P_Q_grid_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
