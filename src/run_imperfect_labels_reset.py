"""Paired perfect/IPW B-PAC/PPI experiment. No API calls or future-data training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from av_core import restart_mixture_weights_for_scheme
from av_weighted_restart import evaluate
from imperfect_labels_reset_core import FalseNegativeCalibrator, OnlineNegativeResidual, label_round


def load_inputs(c):
    root = Path(c['inputs'])
    def rows(name):
        data = [json.loads(line) for line in (root/name).read_text().splitlines()]
        result = {r['session_id']: r for r in data}
        if len(result) != len(data): raise ValueError('duplicate session')
        return result
    ids = json.loads((root/'test_population.json').read_text())['session_ids']
    pred = rows('pac_luna_predictions_fullcorpus5014.jsonl')
    truth = rows('pac_safety_labeling_fullcorpus_ground_truth.jsonl')
    judge = rows('primary_test993.jsonl')
    modes = json.loads((root/'primary_test_taxonomy.json').read_text())['modes']
    cats = {}
    for k, mode in enumerate(modes):
        for sid in mode['member_session_ids']:
            if sid in cats: raise ValueError('multiple categories for one session')
            cats[sid] = k
    y, h, u, categories = [], [], [], []
    for sid in ids:
        score = truth[sid]['session_success_numeric']
        if score is None: raise ValueError('missing success score')
        unsafe = int(score < 70 and bool(judge[sid]['failure_modes']))
        if unsafe != truth[sid]['ground_truth_unsafe'] or unsafe != int(sid in cats):
            raise ValueError('truth/category mismatch')
        y.append(unsafe); h.append(pred[sid]['predicted_unsafe'])
        u.append(pred[sid]['uncertainty_score']/100)
        categories.append(cats.get(sid, -1))
    y,h,u,categories = map(np.asarray, (y,h,u,categories))
    if len(ids)!=4961 or y.sum()!=738 or len(modes)!=538:
        raise ValueError('unexpected historical population')
    if not np.all((h==0)|(h==1)) or not np.all((u>=0)&(u<=1)) or not np.allclose(u*100,np.round(u*100)):
        raise ValueError('invalid Luna predictions/scores')
    return y,h,u,categories


_STATE = None

def initialize(c, arrays):
    global _STATE
    _STATE = c, arrays


def trial(index):
    c, (truth, hat, scores, categories) = _STATE
    T = c['n_draws']; eps = c['epsilons']; rho = c['rho']
    seed = np.random.SeedSequence(c['seed'], spawn_key=(index,))
    coin_seed = seed.spawn(1)[0]
    draws = np.random.default_rng(seed).integers(len(truth), size=T)
    coins = np.random.default_rng(coin_seed).random(T)
    counts = np.bincount(categories[categories>=0])  # diagnostics only
    weights, total = restart_mixture_weights_for_scheme('log_squared', T)
    base = (np.arange(c['bet_points'])+.5)/c['bet_points']
    times = np.unique(np.r_[1,np.arange(c['plot_stride'],T+1,c['plot_stride']),T])
    records, paths = [], {}
    # Perfect and constant-audit PPI paths do not depend on epsilon; reuse them.
    methods = c.get('methods', ['perfect', 'bpac', 'ppi'])
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {'perfect','bpac','ppi','ppi_learned'}:
        raise ValueError('invalid methods')
    arms = [(method, target) for method in methods for target in (eps if method=='bpac' else [None])]
    for method, target in arms:
        calibrator = FalseNegativeCalibrator(target/2,c['alpha']/2,rho,c['bet_points']) if method=='bpac' else None
        residual_model = OnlineNegativeResidual(c['residual_bins'], c['residual_prior_count']) if method=='ppi_learned' else None
        seen = set(); observations=np.empty(T); q=np.empty(T)
        seen_count = 0; remaining = int(counts.sum())
        audits=np.zeros(T,dtype=int); catcalls=np.zeros(T,dtype=int)
        for t, sid in enumerate(draws):
            if method == 'perfect':
                audits[t] = 1
                category = int(categories[sid]) if truth[sid] else None
                catcalls[t] = int(truth[sid])
                observations[t] = int(category is not None and category not in seen)
                if category is not None: seen.add(category)
            else:
                pi = calibrator.propensity(scores[sid]) if calibrator else rho
                # Hidden fields are accessed only through callbacks at permitted points.
                def audit(): return int(truth[sid])
                def categorize():
                    category = int(categories[sid])
                    return category if category >= 0 else None
                residual = residual_model.predict(int(hat[sid]), scores[sid]) if residual_model else 0.
                y, z, audits[t], catcalls[t], fn = label_round(int(hat[sid]), pi, coins[t], seen, audit, categorize, residual)
                observations[t] = z+target/2 if calibrator else y
                if calibrator: calibrator.update(scores[sid], fn, pi)
                # On an audited draw the emitted discovery z equals true novelty.
                # Update AFTER current prediction/correction; never train on an
                # unaudited label or use hidden novelty as a prediction feature.
                if residual_model and audits[t] and hat[sid]==0:
                    residual_model.update(scores[sid], z)
            if len(seen) != seen_count:
                remaining = int(counts.sum())-sum(int(counts[k]) for k in seen)
                seen_count = len(seen)
            q[t] = remaining/len(truth)  # diagnostics, never fed to learners
        q = np.maximum(q,0)
        alpha = c['alpha']/2 if calibrator else c['alpha']
        lambdas = base/(1+target/2) if calibrator else base*(rho if method.startswith('ppi') else 1)
        targets = [target] if calibrator else eps
        grid = np.unique(np.r_[np.arange(301)/1000,np.arange(31,101)/100,q,targets])
        upper, covered, stops = evaluate(observations,grid,lambdas,q,targets,c['library'],
                                         alpha=alpha,restart_weights=weights,total_mass=total,allow_signed=True)
        audit_count=np.cumsum(audits); category_count=np.cumsum(catcalls)
        for e, stop in zip(targets,stops):
            stop = int(stop) if stop<=T else None; end = stop or T
            records.append(dict(trial=index,method=method,epsilon=e,stop=stop,
                                audits=int(audit_count[end-1]),categorizer_calls=int(category_count[end-1]),
                                missing_mass=float(q[end-1]),stop_violation=bool(stop and q[end-1]>e),
                                coverage_violation=bool(np.any(~covered))))
            prefix = f'{method}_{e}'
            for field, values in [('upper',upper),('mass',q),('audits',audit_count),
                                  ('categorizer_calls',category_count),('coverage',np.logical_and.accumulate(covered))]:
                paths[f'{prefix}__{field}'] = values[times-1]
    return records,paths


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    c=json.loads(Path(parser.parse_args().config).read_text())
    if c['sampling']!='uniform_with_replacement' or not 0<c['rho']<1 or not 0<c['alpha']<1:
        raise ValueError('invalid experiment configuration')
    arrays=load_inputs(c);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False)
    indices=c.get('trial_indices', list(range(c['n_trials'])))
    if len(indices)!=c['n_trials'] or len(set(indices))!=len(indices) or any(not isinstance(i,int) or i<0 for i in indices):
        raise ValueError('invalid trial indices')
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(c['inputs']).iterdir() if p.is_file()}
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    records=[];paths=[]
    with (out/'trials.jsonl').open('w') as checkpoint:
        with ProcessPoolExecutor(max_workers=c['workers'],initializer=initialize,initargs=(c,arrays)) as pool:
            for i,(rows,path) in enumerate(pool.map(trial,indices)):
                records.extend(rows);paths.append(path)
                checkpoint.write(''.join(json.dumps(r)+'\n' for r in rows));checkpoint.flush()
                if (i+1)%10==0:print(f'{i+1}/{c["n_trials"]} trials',flush=True)
    times=np.unique(np.r_[1,np.arange(c['plot_stride'],c['n_draws']+1,c['plot_stride']),c['n_draws']])
    np.savez_compressed(out/'trajectories.npz',times=times,**{k:np.stack([p[k] for p in paths]) for k in paths[0]})
    summary=[]
    for e in c['epsilons']:
        for method in c.get('methods', ['perfect','bpac','ppi']):
            rows=[r for r in records if r['epsilon']==e and r['method']==method]
            summary.append(dict(epsilon=e,method=method,censored=sum(r['stop'] is None for r in rows),
                mean_draws_restricted=float(np.mean([r['stop'] or c['n_draws'] for r in rows])),
                mean_audits=float(np.mean([r['audits'] for r in rows])),
                mean_categorizer_calls=float(np.mean([r['categorizer_calls'] for r in rows])),
                coverage_violations=sum(r['coverage_violation'] for r in rows),
                stop_violations=sum(r['stop_violation'] for r in rows)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
