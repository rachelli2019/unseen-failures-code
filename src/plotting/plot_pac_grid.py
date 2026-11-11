#!/usr/bin/env python3
"""Build the PAC 2x2 comparison from saved labels and exact replay histories.
Reconstructs UCBs; does not rerun sampling, calibration, or label queries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from av_core import normalized_log_squared_weights
from av_weighted_restart import evaluate
from plot_fixed_q_primary import plt,style,paper_style,bounds_panel,finish_axes,ARM_COLORS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--horizon',type=int,default=3000)
    p.add_argument('--reuse-cache',action='store_true',
                   help='Re-render from an existing --cache-dir instead of reconstructing the bounds. '
                        'Skips the kernel build, the per-trial evaluation and the crossing check, so it '
                        'only applies to a cache this script already wrote and verified.')
    args=p.parse_args()
    source=args.input_dir/'trajectories.npz'
    with np.load(source) as saved:
        c=json.loads(str(saved['metadata_json']))
        if args.epsilon not in c['epsilons'] or not 0<args.horizon<=c['n_draws']:
            raise ValueError('Requested epsilon/horizon not available')
        e=args.epsilon; H=args.horizon
        if not args.reuse_cache:
            z={'P':saved['perfect_z'][:,:H],'Q':saved[f'{e}_z'][:,:H]}
            truth={'P':saved['oracle'][:,:H],'Q':saved[f'{e}_q'][:,:H]}
            logs={'P':saved[f'{e}_perfect_log_at_epsilon'][:,:H],
                  'Q':saved[f'{e}_pac_log_at_epsilon'][:,:H]}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cpp=Path(__file__).resolve().parent.parent/'weighted_restart_kernel.cpp'
    results={}
    if args.reuse_cache:
        truth={}
        for arm in ('P','Q'):
            with np.load(args.cache_dir/f'{arm}_bounds.npz') as d:
                upper,covered,q=d['upper'],d['simultaneous_covered'],d['true_missing_mass']
            if upper.shape!=(c['n_trials'],H):
                raise ValueError(f'{arm} cache is {upper.shape}, not the requested ({c["n_trials"]}, {H})')
            results[arm]=(upper,covered); truth[arm]=q
    else:
        if args.cache_dir.exists(): raise FileExistsError(f'Refusing to overwrite {args.cache_dir}')
        args.cache_dir.mkdir(parents=True)
        library=args.cache_dir/'kernel.so'
        subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(library)],check=True)
        w=normalized_log_squared_weights(H)
        base=(np.arange(c['lambda_grid_points'])+.5)/c['lambda_grid_points']
        for arm in ('P','Q'):
            zeta=0 if arm=='P' else e/2
            alpha=c['alpha'] if arm=='P' else c['alpha']/2
            lam=base/(1+zeta)
            def one(i):
                q=truth[arm][i]
                grid=np.unique(np.r_[np.linspace(0,1,501),q,e])
                upper,coverage,stops=evaluate(z[arm][i]+zeta,grid,lam,q,[e],library,alpha,w,1.)
                hits=np.flatnonzero(logs[arm][i]>np.log(1/alpha))
                expected=hits[0]+1 if len(hits) else H+1
                if stops[0]!=expected: raise AssertionError('Reconstructed crossing differs from saved PAC run')
                return upper,np.logical_and.accumulate(coverage)
            rows=[]
            with ThreadPoolExecutor(max_workers=4) as ex:
                for row in ex.map(one,range(c['n_trials'])):
                    rows.append(row)
                    if len(rows)%100==0: print(f'{arm}: {len(rows)}/{c["n_trials"]}',flush=True)
            upper,covered=[np.stack(x) for x in zip(*rows)]
            results[arm]=(upper,covered)
            np.savez_compressed(args.cache_dir/f'{arm}_bounds.npz',upper=upper,simultaneous_covered=covered,true_missing_mass=truth[arm])
    records=json.loads((args.input_dir/'trials.json').read_text())
    costs=[]
    for epsilon in c['epsilons']:
        diffs=[]
        for trial in records:
            r=next(r for r in trial if r['epsilon']==epsilon)
            if r['pac_stop'] is None or r['perfect_stop'] is None:
                raise ValueError('Cannot average censored stopping times')
            diffs.append(r['pac_stop']-r['perfect_stop'])
        costs.append(float(np.mean(diffs)))
    labels={'P':'Ground-Truth','Q':'Imperfect Failure Labels'}
    # Two panels, PAC arm only: the reconstructed upper bound against the true unseen
    # failure probability, and that same arm's simultaneous coverage. Panels carry no
    # titles -- the paper caption names them, matching plot_bernoulli_coverage.py.
    with paper_style():
        fig,axes=plt.subplots(1,2,figsize=(2*style.FIGSIZE[0],style.FIGSIZE[1]))
        bounds_panel(axes[0],'Q',truth['Q'],results['Q'][0],.95)
        axes[0].set_title('')
        ax=axes[1]
        rate=results['Q'][1].mean(0)
        ax.plot(np.arange(1,H+1),rate,color=ARM_COLORS['Q'],linewidth=3.2)
        ax.axhline(1-c['alpha'],color=style.NOMINAL_COVERAGE_COLOR,linestyle=(0,(2,2)),linewidth=1.6)
        ax.annotate(r'nominal $1-\alpha$',xy=(H,1-c['alpha']),xytext=(-8,6),textcoords='offset points',ha='right',va='bottom',color=style.NOMINAL_COVERAGE_COLOR)
        finish_axes(ax,H)
        # Coverage sits at 1.000 here, so the curve needs headroom or it renders flush
        # against the top spine and reads as part of the frame rather than as data.
        floor=min(.94,float(rate.min())-.01)
        ax.set_ylim(floor,1+(1-floor)*.02);ax.set_yticks(np.round(np.arange(1.,floor-1e-9,-.02)[::-1],2))
        ax.set_ylabel('Simultaneous coverage')
        fig.tight_layout(w_pad=2.)
        style.save_figure(fig,args.output_dir,'pac_labeling_grid');plt.close(fig)
    summary={'source_run':str(args.input_dir),'config':c,'bound_coverage_epsilon':e,'bound_coverage_horizon':H,
             'coverage_at_horizon':{labels[a]:float(results[a][1][:,-1].mean()) for a in results},
             'stopping_epsilons':c['epsilons'],'mean_additional_stopping_cost':costs,
             'uncertainty':'Upper bounds: central 95% trial bands. Additional stopping cost: mean only, paired identical session draws.',
             'caveat':'PAC uses imperfect Luna predictions but queries every pre-stop label at both tested epsilons. Extra stopping cost includes epsilon/2 and alpha/2 budgets.',
             'method':'Normalized log-squared weights, total capital 1, K=100 lambda mixture. PAC zeta=epsilon/2; lambda midpoint grid/(1+epsilon/2); alpha/2.',
             'numerics':'UCB rounded upward by at most .002. Coverage evaluated directly at history-specific true mass. All reconstructed threshold crossings checked against saved logs.',
             'files_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source,args.input_dir/'trials.json',args.input_dir/'summary.json',Path(__file__),cpp]}}
    for directory in [args.cache_dir,args.output_dir]:
        (directory/'pac_labeling_grid_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['coverage_at_horizon','mean_additional_stopping_cost']},indent=2))

if __name__=='__main__':main()
