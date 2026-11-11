"""Paired exploratory alpha/epsilon sweep using the fixed-population PAC runner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from run_pac_labeling import load_population, trial


def one_job(args):
    c,arrays,index=args
    seed=np.random.SeedSequence(c['seed'],spawn_key=(index,))
    records,trajectories=trial((c,arrays,seed))
    for r in records:
        e=r['epsilon']; tau=trajectories[f'{e}_tau']; query=trajectories[f'{e}_query']; risk=trajectories[f'{e}_risk']
        finite=np.flatnonzero(tau>-np.inf)
        t=r['pac_stop'] or c['n_draws']
        r.update(alpha=c['alpha'],trial=index,
                 first_finite_tau=int(finite[0]+1) if len(finite) else None,
                 skipped_before_stop_or_horizon=int(t-query[:t].sum()),
                 fraction_skipped_before_stop_or_horizon=float(1-query[:t].mean()),
                 horizon_query_fraction=float(query.mean()),
                 horizon_population_error=float(risk[-1]),
                 max_population_error=float(risk.max()))
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());base=json.loads(Path(config['base_config']).read_text())
    base.update(n_trials=config['n_trials'],n_draws=config['n_draws'],epsilons=config['epsilons'])
    ids,*arrays=load_population(base)
    jobs=[(dict(base,alpha=alpha),arrays,i) for alpha in config['alphas'] for i in range(config['n_trials'])]
    records=[];output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=base['n_workers']) as pool:
        for i,result in enumerate(pool.map(one_job,jobs,chunksize=2)):
            records.extend(result)
            if (i+1)%20==0:
                print(f'{i+1}/{len(jobs)} trial groups',flush=True)
                (output/'trials.partial.json').write_text(json.dumps(records))
    summary=[]
    for alpha in config['alphas']:
        for e in config['epsilons']:
            rs=[r for r in records if r['alpha']==alpha and r['epsilon']==e]
            result=dict(alpha=alpha,epsilon=e,n_trials=len(rs),n_stopped=sum(r['pac_stop'] is not None for r in rs),
                        trials_saving_queries=sum(r['skipped_before_stop_or_horizon']>0 for r in rs),
                        label_error_violations=sum(r['ever_label_error_exceeds_epsilon_half'] for r in rs),
                        missing_mass_violations=sum(r['pac_stop'] is not None and r['missing_mass_at_stop_or_horizon']>e for r in rs))
            for key in ['pac_stop','perfect_stop','queries_at_stop_or_horizon','fraction_skipped_before_stop_or_horizon','horizon_query_fraction','horizon_population_error','first_finite_tau','population_error_at_stop_or_horizon','missing_mass_at_stop_or_horizon']:
                v=[r[key] for r in rs if r[key] is not None];result['mean_'+key]=float(np.mean(v)) if v else None
            result['maximum_population_error']=max(r['max_population_error'] for r in rs)
            summary.append(result)
    (output/'trials.json').write_text(json.dumps(records)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(config=config,base_config=base,results=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
