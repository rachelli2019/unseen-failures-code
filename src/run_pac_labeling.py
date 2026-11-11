"""Replay the fixed-population PAC experiment; run with --config PATH."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from av_core import binary_restart_log_e_at_threshold, normalized_log_squared_weights
from av_discovery_fixed_primary import load_session_to_failure_mode
from evaluation_population import load_population_session_ids
from pac_labeling_core import PacCalibrator


def rows(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r['session_id'] in result:
            raise ValueError(f'duplicate session in {path}')
        result[r['session_id']] = r
    return result


def load_population(c):
    ids = load_population_session_ids(Path(c['population_json']))
    pred, truth, judge = (rows(c[k]) for k in ['predictions_jsonl', 'ground_truth_jsonl', 'failure_modes_jsonl'])
    taxonomy = load_session_to_failure_mode(Path(c['taxonomy_json']))
    if len(ids) != 4961 or len(judge) != 993 or not set(judge) <= set(ids):
        raise ValueError('unexpected test population')
    cats = {v:i for i,v in enumerate(sorted(set(taxonomy.values())))}
    y, h, u, modes = [], [], [], []
    for sid in ids:
        score = truth[sid]['session_success_numeric']
        if score is None:
            raise ValueError('missing score')
        unsafe = int(score < 70 and bool(judge[sid]['failure_modes'])) if score < 70 else 0
        if unsafe != int(sid in taxonomy) or unsafe != truth[sid]['ground_truth_unsafe']:
            raise ValueError(f'inconsistent truth/taxonomy for {sid}')
        if sid in judge and bool(judge[sid]['failure_modes']) != judge[sid]['session_contains_failure']:
            raise ValueError('judge fields disagree')
        y.append(unsafe); h.append(pred[sid]['predicted_unsafe']); u.append(pred[sid]['uncertainty_score'])
        modes.append(cats[taxonomy[sid]] if sid in taxonomy else -1)
    y,h,u,modes = map(np.asarray,(y,h,u,modes))
    if not np.all((h==0)|(h==1)) or not np.all(np.isfinite(u)) or y.sum()!=738:
        raise ValueError('invalid population labels')
    return ids,y,h,u,modes


def trial(args):
    c, arrays, seed = args
    y,h,u,modes = arrays
    N,T = len(y),c['n_draws']
    children = seed.spawn(len(c['epsilons']))
    # Use the original trial seed for draws, matching primary_fixed.
    draws = np.random.default_rng(seed).integers(0,N,size=T)
    grid = np.unique(u)
    mass = np.bincount(modes[modes>=0])/N
    seen = np.zeros(len(mass),dtype=bool)
    novelty = np.zeros(T,dtype=np.int8); oracle=np.empty(T)
    for i,j in enumerate(draws):
        k=modes[j]
        if k>=0:
            novelty[i]=not seen[k]; seen[k]=True
        oracle[i]=mass[~seen].sum()
    K=c['lambda_grid_points']; weights=normalized_log_squared_weights(T)
    out={'draws':draws,'oracle':oracle,'perfect_z':novelty}
    records=[]
    for ei,eps in enumerate(c['epsilons']):
        lam=(np.arange(K)+.5)/K
        perfect_log=binary_restart_log_e_at_threshold(novelty,eps,lam,np.ones(K)/K,weights)
        pc=PacCalibrator(grid,eps,c['alpha']); rng=np.random.default_rng(children[ei])
        tau=np.empty(T); pi=np.empty(T); query=np.zeros(T,dtype=np.int8); tilde=np.empty(T,dtype=np.int8)
        risk=np.empty(T); z=np.zeros(T,dtype=np.int8); q=np.empty(T)
        seen=np.zeros(len(mass),dtype=bool)
        # Population errors are diagnostics only, never supplied to calibration.
        error_mass=np.array([np.mean((h!=y)&(u==v)) for v in grid])
        prefix=np.r_[0,np.cumsum(error_mass)]
        for i,j in enumerate(draws):
            t=i+1; tau[i]=pc.threshold(); floor=t**-.25
            pi[i]=floor if u[j]<tau[i] else 1.
            risk[i]=(1-floor)*prefix[np.searchsorted(grid,tau[i],side='left')]
            query[i]=rng.random()<pi[i]
            tilde[i]=y[j] if query[i] else h[j]
            pc.update(u[j],h[j],y[j] if query[i] else None,pi[i])
            k=modes[j]
            if k>=0:
                z[i]=int(tilde[i] and not seen[k])
                if c['history']=='all_traces' or tilde[i]:
                    seen[k]=True
            q[i]=mass[~seen].sum()
        pac_log=binary_restart_log_e_at_threshold(z,eps/2,lam/(1+eps/2),np.ones(K)/K,weights)
        def stopping(log,a):
            hits=np.flatnonzero(log>np.log(1/a))
            return int(hits[0]+1) if len(hits) else None
        perfect_stop=stopping(perfect_log,c['alpha']); stop=stopping(pac_log,c['alpha']/2)
        idx=stop-1 if stop else T-1
        record={'epsilon':eps,'perfect_stop':perfect_stop,'pac_stop':stop,
                'queries_at_stop_or_horizon':int(query[:idx+1].sum()),
                'population_error_at_stop_or_horizon':float(risk[idx]),
                'max_population_error_through_stop_or_horizon':float(risk[:idx+1].max()),
                'ever_label_error_exceeds_epsilon_half':bool(np.any(risk>eps/2)),
                'realized_error_through_stop_or_horizon':float(np.mean(tilde[:idx+1]!=y[draws[:idx+1]])),
                'missing_mass_at_stop_or_horizon':float(q[idx]),
                'perfect_missing_mass_at_stop_or_horizon':float(oracle[perfect_stop-1 if perfect_stop else T-1])}
        records.append(record)
        for key,value in dict(tau=tau,pi=pi,query=query,tilde=tilde,risk=risk,z=z,q=q,pac_log_at_epsilon=pac_log,perfect_log_at_epsilon=perfect_log).items():
            out[f'{eps}_{key}']=value
    return records,out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    c=json.loads(Path(parser.parse_args().config).read_text())
    if c['history'] not in ('all_traces','flagged_only') or c['lcheck_prime'] != 'lcheck':
        raise ValueError('Resolve history and L-check-prime definitions in the config before running')
    if c['sampling'] != 'uniform_with_replacement':
        raise ValueError('This experiment requires uniform sampling with replacement')
    ids,*arrays=load_population(c)
    seeds=np.random.SeedSequence(c['seed']).spawn(c['n_trials'])
    jobs=((c,arrays,s) for s in seeds)
    records=[]; trajectories=[]
    with ProcessPoolExecutor(max_workers=c['n_workers']) as pool:
        for i,(rec,out) in enumerate(pool.map(trial,jobs,chunksize=5)):
            records.append(rec); trajectories.append(out)
            if (i+1)%100==0: print(f'{i+1}/{c["n_trials"]} trials',flush=True)
    summary={'config':c,'n_population':len(ids),'n_unsafe':int(arrays[0].sum()),
             'raw_classifier_error':float(np.mean(arrays[0]!=arrays[1])),
             'input_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in c.items() if k.endswith('_json') or k.endswith('_jsonl')},'results':[]}
    for ei,eps in enumerate(c['epsilons']):
        rs=[r[ei] for r in records]; result={'epsilon':eps}
        for key in rs[0]:
            if key=='epsilon': continue
            vals=[r[key] for r in rs if r[key] is not None]
            result[key]={'n':len(vals),'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None,'min':float(np.min(vals)) if vals else None,'max':float(np.max(vals)) if vals else None}
        result['pac_stops_above_missing_mass_tolerance']=sum(r['pac_stop'] is not None and r['missing_mass_at_stop_or_horizon']>eps for r in rs)
        summary['results'].append(result)
    output=Path(c['output_dir']); output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'trials.json').write_text(json.dumps(records)+'\n')
    np.savez_compressed(output/'trajectories.npz',session_ids=np.array(ids),metadata_json=json.dumps(c),**{k:np.stack([x[k] for x in trajectories]) for k in trajectories[0]})
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
