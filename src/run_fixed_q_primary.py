#!/usr/bin/env python3
"""Run the fixed-Q primary-taxonomy experiment.

Run from repository root with --config configs/run_fixed_q_primary.json.
Requires a C++ compiler; results are written to a fresh output directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pyarrow.parquet as pq
from av_discovery_fixed_primary import load_session_to_failure_mode, build_session_population, run_bootstrap_trial
from evaluation_population import load_population_session_ids
from av_weighted_restart import length_distribution,evaluate


def wilson(k,n):
    z=1.959963984540054; p=np.asarray(k)/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half


def prepare(c):
    ids=load_population_session_ids(Path(c['population_json']))
    if len(ids)!=4961: raise ValueError('Expected 4961 reference sessions')
    mapping=load_session_to_failure_mode(Path(c['taxonomy_json']))
    rows=[json.loads(s) for s in Path(c['labels_jsonl']).read_text().splitlines()]
    if len(rows)!=len({r['session_id'] for r in rows}): raise ValueError('Duplicate labels')
    if {r['session_id'] for r in rows if r['session_contains_failure']}!=set(mapping):
        raise ValueError('Failure labels and taxonomy disagree')
    if any(len(r['failure_modes'])!=int(r['session_contains_failure']) for r in rows):
        raise ValueError('Expected single primary labels')
    modes,freq,_=build_session_population(mapping,ids)
    lengths={}
    table=pq.read_table(c['sessions_parquet'],columns=['session_id','conversation_transcript_json'])
    for r in table.to_pylist():
        if r['session_id'] in lengths: raise ValueError('Duplicate transcript')
        transcript=json.loads(r['conversation_transcript_json'])
        if not isinstance(transcript,list) or any(not isinstance(m['content'],str) for m in transcript):
            raise ValueError('Malformed transcript')
        lengths[r['session_id']]=sum(len(m['content']) for m in transcript)
    lengths=np.array([lengths[s] for s in ids])
    median,long,q,ratio=length_distribution(lengths)
    # Small fixed margin makes the ratio bounds strictly interior.
    B=float(max(ratio.max(),1/ratio.min())+1e-6)
    cap=1/(B+max(c['stopping_epsilons'])/2)
    failure=np.array([s in mapping for s in ids])
    info={'n_population':len(ids),'n_failure_sessions':len(mapping),'n_modes':len(freq),
          'median_characters':median,'n_short':int((~long).sum()),'n_long':int(long.sum()),
          'Q_long_probability':float(q[long].sum()),'P_over_Q_short':float(ratio[~long][0]),
          'P_over_Q_long':float(ratio[long][0]),'B':B,'Q_lambda_cap':cap,
          'failure_fraction_short':float(failure[~long].mean()),'failure_fraction_long':float(failure[long].mean())}
    return ids,modes,freq,lengths,long,q,ratio,info


def run(c):
    for key in ['n_trials','horizon','lambda_grid_points','plot_grid_intervals','workers']:
        if not isinstance(c[key],int) or c[key]<=0: raise ValueError(f'Invalid {key}')
    if not 0<c['alpha']<1 or not c['stopping_epsilons'] or any(not 0<e<1 for e in c['stopping_epsilons']):
        raise ValueError('Invalid alpha/epsilon')
    out=Path(c['output_dir'])
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    ids,modes,freq,lengths,long,q,ratio,info=prepare(c)
    out.mkdir(parents=True)
    library=out/'weighted_restart.so'
    source=Path(__file__).with_name('weighted_restart_kernel.cpp')
    subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    np.savez_compressed(out/'population.npz',session_ids=ids,lengths=lengths,long=long,q=q,p_over_q=ratio)
    seeds=np.random.SeedSequence(c['seed']).spawn(c['n_trials'])
    base=(np.arange(c['lambda_grid_points'])+.5)/c['lambda_grid_points']
    summary={'config':c,'population':info,'arms':{},'input_sha256':{
        key:hashlib.sha256(Path(c[key]).read_bytes()).hexdigest() for key in ['population_json','taxonomy_json','labels_jsonl','sessions_parquet']},
        'restart_convention':'Existing baseline: omega_s=1/[2.11(s+1)log(s+1)^2], total capital 0.999, residual/unstarted capital retained.',
        'numerics':'Native scaled recurrence; exact evaluations at oracle mass and epsilon for coverage/stopping. Upper bounds rounded outward on a grid with maximum gap 0.002 (default).',
        'coverage_intervals':'95% pointwise Wilson intervals for Monte Carlo coverage fractions, not a simultaneous band over time.'}
    results={}; start=time.monotonic()
    for arm in ['P','Q']:
        lambdas=base if arm=='P' else base*info['Q_lambda_cap']
        def one(seed):
            rng=np.random.default_rng(seed)
            draws=rng.integers(0,len(ids),c['horizon']) if arm=='P' else rng.choice(len(ids),c['horizon'],p=q)
            z,_,truth=run_bootstrap_trial(draws,modes,freq)
            # Exact rational finite-population masses, avoiding accumulated subtraction error.
            truth=np.rint(truth*len(ids))/len(ids)
            y=z.astype(float) if arm=='P' else z*ratio[draws]
            grid=np.unique(np.r_[np.linspace(0,1,c['plot_grid_intervals']+1),truth,c['stopping_epsilons']])
            upper,covered,stops=evaluate(y,grid,lambdas,truth,c['stopping_epsilons'],library,c['alpha'])
            return truth,upper,np.logical_and.accumulate(covered),stops
        rows=[]
        with ThreadPoolExecutor(max_workers=c['workers']) as executor:
            for row in executor.map(one,seeds):
                rows.append(row)
                if len(rows)%25==0: print(f'{arm}: {len(rows)}/{len(seeds)} ({time.monotonic()-start:.0f}s)',flush=True)
        truth,upper,covered,stops=[np.stack(a) for a in zip(*rows)]
        results[arm]=(truth,upper,covered,stops)
        stats={'lambda_min':float(lambdas.min()),'lambda_max':float(lambdas.max()),
               'simultaneous_coverage':float(covered[:,-1].mean()),
               'coverage_95pct_MC_interval':[float(x) for x in wilson(covered[:,-1].sum(),len(seeds))],'stopping':{}}
        for j,e in enumerate(c['stopping_epsilons']):
            s=stops[:,j]; observed=s<=c['horizon']; med=float(np.median(s))
            bad=observed & (truth[np.arange(len(seeds)),np.minimum(s,c['horizon'])-1]>e)
            stats['stopping'][str(e)]={'stopped_fraction':float(observed.mean()),
                'median_stopping_time':med if med<=c['horizon'] else None,
                'restricted_mean_min_T_H':float(np.minimum(s,c['horizon']).mean()),
                'false_certification_fraction':float(bad.mean())}
        summary['arms'][arm]=stats
        np.savez_compressed(out/f'{arm}_trajectories.npz',true_missing_mass=truth,upper=upper,simultaneous_covered=covered,stopping_times=stops,lambda_values=lambdas)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(results,c,out)
    print(json.dumps(summary,indent=2),flush=True)


def plot(results,c,out):
    from plotting.plot_fixed_q_primary import render_main
    render_main(results,c,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    run(json.loads(parser.parse_args().config.read_text()))
