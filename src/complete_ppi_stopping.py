"""Merge continued PPI stopping costs, retaining the common 10,000-draw coverage."""
import json
from pathlib import Path
import numpy as np


def main():
    c=json.loads(Path('configs/run_imperfect_labels_final.json').read_text())
    extended=json.loads(Path('configs/run_imperfect_labels_ppi_continuation.json').read_text())
    original=Path(c['output_dir']);continuation=Path(extended['output_dir'])
    for key in ['seed','rho','alpha','bet_points','inputs','plot_stride','library']:
        if c[key]!=extended[key]:raise ValueError(f'continuation mismatch: {key}')
    if json.loads((original/'input_hashes.json').read_text())!=json.loads((continuation/'input_hashes.json').read_text()):
        raise ValueError('input hashes differ')
    def read_rows(root):return [json.loads(line) for line in (root/'trials.jsonl').read_text().splitlines()]
    rows=read_rows(original);extra=read_rows(continuation)
    unfinished={(r['trial'],r['method'],r['epsilon']) for r in rows if r['stop'] is None}
    by_key={(r['trial'],r['method'],r['epsilon']):r for r in extra}
    if set(by_key)!=unfinished or len(extra)!=len(unfinished):raise ValueError('wrong continuation trial set')
    if any(r['stop'] is None for r in extra):raise ValueError('continuation still censored: extend before merging')
    a=np.load(original/'trajectories.npz');b=np.load(continuation/'trajectories.npz')
    n=len(a['times']);np.testing.assert_array_equal(a['times'],b['times'][:n])
    # Replay, not resampling: every stored prefix quantity must agree exactly.
    for key in b.files:
        if key=='times':continue
        np.testing.assert_array_equal(a[key][extended['trial_indices']],b[key][:,:n])
    fields=['stop','audits','categorizer_calls','missing_mass','stop_violation']
    for row in rows:
        key=row['trial'],row['method'],row['epsilon']
        row['continued_to_stop']=key in by_key
        if key in by_key:
            replacement=by_key[key]
            if replacement['stop']<=c['n_draws']:raise ValueError('previously censored trial stopped in old horizon')
            for field in fields:row[field]=replacement[field]
        # Keep ORIGINAL coverage_violation: all arms share the 10,000-draw window.
    summary=[]
    for e in c['epsilons']:
        for method in c['methods']:
            selected=[r for r in rows if r['method']==method and r['epsilon']==e]
            if len(selected)!=c['n_trials'] or any(r['stop'] is None for r in selected):
                raise ValueError('incomplete paired trial set')
            summary.append(dict(epsilon=e,method=method,censored=0,
                coverage_horizon=c['n_draws'],continued_trials=sum(r['continued_to_stop'] for r in selected),
                mean_draws_restricted=float(np.mean([r['stop'] for r in selected])),
                mean_audits=float(np.mean([r['audits'] for r in selected])),
                mean_categorizer_calls=float(np.mean([r['categorizer_calls'] for r in selected])),
                max_stop=max(r['stop'] for r in selected),
                coverage_violations=sum(r['coverage_violation'] for r in selected),
                stop_violations=sum(r['stop_violation'] for r in selected)))
    (original/'completed_stopping_trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (original/'completed_stopping_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps([r for r in summary if r['method']=='ppi' and r['epsilon']==.1],indent=2))


if __name__=='__main__':main()
