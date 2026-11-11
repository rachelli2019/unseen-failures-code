#!/usr/bin/env python3
"""Extend stopping-time follow-up for a completed fixed-Q run, preserving prefixes.

Only threshold e-values are needed, so this is much cheaper than full trajectories.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from av_discovery_fixed_primary import SUCCESS_MODE_ID
from av_weighted_restart import evaluate
from run_fixed_q_primary import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=30000)
    p.add_argument('--epsilons',type=float,nargs='+')
    p.add_argument('--eps-range',type=float,nargs=2,metavar=('MIN','MAX'))
    p.add_argument('--n-eps',type=int,default=60)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--lambda-max-epsilon',type=float,help='Preserve a grid valid for this larger epsilon')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--plots-dir',type=Path)
    args=p.parse_args()
    summary=json.loads((args.run_dir/'summary.json').read_text()); c=summary['config']
    if args.horizon<=c['horizon']: raise ValueError('Extension horizon must exceed original horizon')
    if args.eps_range is not None:
        if args.epsilons is not None or args.n_eps < 2 or not 0 < args.eps_range[0] < args.eps_range[1] < 1:
            raise ValueError('Provide either epsilons or an increasing epsilon range')
        args.epsilons = np.linspace(*args.eps_range,args.n_eps).tolist()
    eps = c['stopping_epsilons'] if args.epsilons is None else args.epsilons
    if not eps or len(set(eps)) != len(eps) or any(not 0 < e < 1 for e in eps):
        raise ValueError('Epsilons must be distinct and lie in (0,1)')
    lambda_epsilon = max(eps) if args.lambda_max_epsilon is None else args.lambda_max_epsilon
    if not max(eps) <= lambda_epsilon < 1:
        raise ValueError('Lambda tuning epsilon must cover all thresholds and be below one')
    out=args.output_dir or args.run_dir/f'stopping_h{args.horizon}'
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    for key,digest in summary['input_sha256'].items():
        if hashlib.sha256(Path(c[key]).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Input changed: {key}')
    ids,modes,freq,_,_,q,ratio,_=prepare(c)
    names,codes=np.unique(modes,return_inverse=True)
    success=int(np.flatnonzero(names==SUCCESS_MODE_ID)[0])
    counts=np.bincount(codes); failure_total=len(ids)-counts[success]
    candidates=np.unique(np.r_[0.,eps,1.])
    seeds=np.random.SeedSequence(c['seed']).spawn(c['n_trials'])
    out.mkdir(parents=True)
    library=out/'weighted_restart.so'
    subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(Path(__file__).with_name('weighted_restart_kernel.cpp')),'-o',str(library)],check=True)
    result={'horizon':args.horizon,'n_trials':c['n_trials'],'source_run':str(args.run_dir),'epsilons':eps,'lambda_max_epsilon':lambda_epsilon,'seed':c['seed'],'alpha':c['alpha'],'input_sha256':summary['input_sha256'],'arms':{}}
    all_stops={}
    for arm in ['P','Q']:
        original=np.load(args.run_dir/f'{arm}_trajectories.npz')
        lam=original['lambda_values']
        if arm == 'Q' and args.epsilons is not None:
            cap = 1 / (summary['population']['B'] + lambda_epsilon/2)
            lam = (np.arange(c['lambda_grid_points']) + .5) / c['lambda_grid_points'] * cap
        def one(seed):
            rng=np.random.default_rng(seed)
            draws=rng.integers(0,len(ids),args.horizon) if arm=='P' else rng.choice(len(ids),args.horizon,p=q)
            distinct,first=np.unique(codes[draws],return_index=True)
            mask=distinct!=success; distinct,first=distinct[mask],first[mask]
            y=np.zeros(args.horizon)
            y[first]=1 if arm=='P' else ratio[draws[first]]
            _,_,stops=evaluate(y,candidates,lam,np.zeros(args.horizon),eps,library,c['alpha'])
            mass=np.array([(failure_total-counts[distinct[first < min(t,args.horizon)]].sum())/len(ids) for t in stops])
            discovered = np.zeros(args.horizon,dtype=np.int64)
            discovered[first] = counts[distinct]
            truth = (failure_total - np.cumsum(discovered)) / len(ids)
            oracle = np.array([int(np.flatnonzero(truth < e)[0])+1 if np.any(truth < e) else args.horizon+1 for e in eps])
            return stops,mass,oracle
        rows=[]
        with ThreadPoolExecutor(max_workers=c['workers']) as ex:
            for row in ex.map(one,seeds):
                rows.append(row)
                if len(rows)%100==0: print(f'{arm}: {len(rows)}/{len(seeds)}',flush=True)
        stops,mass,oracle=[np.stack(a) for a in zip(*rows)]
        # A longer horizon must reproduce every original observed/censored prefix.
        same_grid = np.array_equal(lam, original['lambda_values'])
        if same_grid:
            for j,e in enumerate(eps):
                if e in c['stopping_epsilons']:
                    old_j = c['stopping_epsilons'].index(e)
                    np.testing.assert_array_equal(np.minimum(stops[:,j],c['horizon']+1),original['stopping_times'][:,old_j])
        all_stops[arm]=stops
        stats={}
        for j,e in enumerate(eps):
            s=stops[:,j]; observed=s<=args.horizon
            stats[str(e)]={'stopped_fraction':float(observed.mean()),
                'median_stopping_time':float(np.median(s)) if np.median(s)<=args.horizon else None,
                'restricted_mean_min_T_H':float(np.minimum(s,args.horizon).mean()),
                'false_certification_fraction':float(np.mean(observed & (mass[:,j]>e)))}
        result['arms'][arm]=stats
        if np.any(stops > args.horizon) or np.any(oracle > args.horizon):
            raise ValueError('Extend horizon before reporting certification delay: censored trials')
        delay = stops - oracle
        result.setdefault('delay',{})[arm] = {'mean':delay.mean(0).tolist(), 'p025':np.quantile(delay,.025,axis=0).tolist(), 'p975':np.quantile(delay,.975,axis=0).tolist()}
        result['delay_definition'] = 'tau_bound - tau_true, first strict crossing at t >= 1; negative values retained' 
        result.setdefault('lambda_grids', {})[arm] = {'values':lam.tolist(), 'matches_source_run':bool(same_grid)}
        np.savez_compressed(out/f'{arm}_stopping.npz',stopping_times=stops,mass_at_stop_or_horizon=mass,epsilons=eps,lambda_values=lam,oracle_stopping_times=oracle,certification_delays=delay)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    from plotting.plot_fixed_q_primary import render_stopping
    if not args.no_plots:
        render_stopping(all_stops, eps, args.horizon, out)
    if args.plots_dir is not None:
        import shutil
        if args.plots_dir.exists():
            raise FileExistsError(f'Refusing to overwrite {args.plots_dir}')
        args.plots_dir.mkdir(parents=True)
        for figure in out.iterdir():
            if figure.suffix in {'.png','.pdf'}:
                shutil.copy2(figure,args.plots_dir/figure.name)
        (args.plots_dir/'provenance.json').write_text(json.dumps({'source_run':str(out), 'epsilons':eps, 'command_note':'run_fixed_q_stopping.py --epsilons; exact threshold evaluations'},indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
