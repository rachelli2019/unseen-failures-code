"""Add a Howard et al. empirical-Bernstein CS to saved, paired headline trials.

Run from any directory:
  .venv/bin/python src/run_empirical_bernstein_headline.py

rho=100 is a fixed variance-time scale, chosen before inspecting this comparison,
not an optimum estimated from the evaluation trajectories. No alpha splitting is
required to compare methods individually; selecting their pointwise minimum would
require a separate joint-validity argument and is NOT done here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from empirical_bernstein import empirical_bernstein_upper

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    'restart': ('Restart mixture', '#9b28b0', '-'),
    'eb': ('Empirical-Bernstein CS', '#d36c16', '-'),
    'no_restart': ('No-restart mixture', '#228B22', ':'),
    'gt_simultaneous': ('GT UCB (simultaneous)', '#008c95', '-'),
    'gt_pointwise': ('GT UCB (pointwise)', '#a7a7a7', '--'),
}


def first_crossing(bound, epsilon):
    """Strict crossing; infinity denotes administrative censoring at horizon."""
    crossed = bound < epsilon
    observed = crossed.any(axis=1)
    return np.where(observed, crossed.argmax(axis=1) + 1, np.inf)


def stopping_statistics(bound, q, epsilon, reference):
    horizon = bound.shape[1]
    tau = first_crossing(bound, epsilon)
    observed = np.isfinite(tau)
    restricted = np.minimum(tau, horizon)
    difference = restricted - np.minimum(first_crossing(reference, epsilon), horizon)
    half_width = 1.96 * difference.std(ddof=1) / np.sqrt(len(tau))
    stopped_rows = np.flatnonzero(observed)
    premature = q[stopped_rows, tau[observed].astype(int) - 1] >= epsilon
    # Do not report complete-case means; they systematically omit late trials.
    return {
        'epsilon': float(epsilon), 'stopped_fraction': float(observed.mean()),
        'censored_trials': int((~observed).sum()),
        'mean_stopping_time': float(tau.mean()) if observed.all() else None,
        'restricted_mean_draws': float(restricted.mean()),
        'restricted_mean_minus_restart': float(difference.mean()),
        'paired_difference_ci95_low': float(difference.mean() - half_width),
        'paired_difference_ci95_high': float(difference.mean() + half_width),
        'premature_stop_fraction_all_trials': float(premature.sum() / len(tau)),
    }


def save_figure(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')


def render(output, q, methods, summary, alpha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    x = np.arange(1, q.shape[1]+1)

    main_keys = ('restart', 'eb', 'gt_simultaneous', 'gt_pointwise')
    ablation_keys = ('restart', 'no_restart')

    def bounds(ax, keys):
        for key in keys:
            matrix = methods[key]
            label, color, style = STYLES[key]
            low, high = np.quantile(matrix, [.025, .975], axis=0)
            ax.plot(x, matrix.mean(axis=0), color=color, linestyle=style, label=label, lw=1.8)
            ax.fill_between(x, low, high, color=color, alpha=.10, lw=0)
        ax.plot(x, q.mean(axis=0), color='black', lw=1.6, label='True unseen-failure probability')
        ax.set(xlabel='Number of sampled sessions', ylabel='Unseen-failure probability',
               xlim=(1, len(x)), ylim=(0, 1))
        ax.grid(alpha=.2)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bounds(ax, main_keys)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    save_figure(fig, output, 'headline_bounds')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bounds(ax, ablation_keys)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    save_figure(fig, output, 'restart_ablation_bounds')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key in ('restart', 'eb', 'no_restart'):
        matrix = methods[key]
        label, color, style = STYLES[key]
        low, high = np.quantile(matrix, [.025, .975], axis=0)
        axes[0].plot(x, matrix.mean(axis=0), color=color, linestyle=style, label=label, lw=1.8)
        axes[0].fill_between(x, low, high, color=color, alpha=.12, lw=0)
        rows = summary['methods'][key]['stopping']
        eps = [row['epsilon'] for row in rows]
        axes[1].plot(eps, [row['stopped_fraction'] for row in rows], color=color, marker='o')
        axes[2].plot(eps, [row['restricted_mean_draws'] for row in rows], color=color, marker='o')
    axes[0].plot(x, q.mean(axis=0), color='black', label='True unseen-failure probability')
    axes[0].set(xlim=(1, len(x)), ylim=(0, .25), xlabel='Number of sampled sessions',
                ylabel='Unseen-failure probability')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel=r'Stopping threshold $\varepsilon$', ylabel='Fraction certified by 3,000 draws',
                ylim=(-.03, 1.03))
    axes[2].set(xlabel=r'Stopping threshold $\varepsilon$', ylabel='Mean draws, truncated at 3,000')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, output, 'sequential_comparison')
    plt.close(fig)

    for name, keys in [('coverage', main_keys), ('restart_ablation_coverage', ablation_keys)]:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for key in keys:
            matrix = methods[key]
            label, color, style = STYLES[key]
            coverage = (~np.maximum.accumulate(matrix < q, axis=1)).mean(axis=0)
            ax.plot(x, coverage, label=label, color=color, linestyle=style, lw=1.8)
        ax.axhline(1-alpha, color='black', linestyle=':', label='Nominal coverage')
        ax.set(xlabel='Number of sampled sessions', ylabel='Simultaneous coverage',
               ylim=(.94, 1.005), xlim=(1, len(x)))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, framealpha=1)
        fig.tight_layout()
        save_figure(fig, output, name)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/run_empirical_bernstein_headline.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['first_prediction'] != .5:
        raise ValueError('this implementation initializes the predictor at 1/2')
    with np.load(ROOT/config['headline_npz']) as data:
        metadata = json.loads(str(data['metadata_json']))
        z, q = data['z_matrix'], data['q_matrix']
        restart = data['bound__uniform_K100']
    if metadata['alpha'] != config['alpha']:
        raise ValueError('confidence levels must match')
    methods = {'restart': restart}
    for path_key in ('no_restart_npz', 'gt_npz'):
        with np.load(ROOT/config[path_key]) as data:
            other_meta = json.loads(str(data['metadata_json']))
            if other_meta['alpha'] != config['alpha']:
                raise ValueError(f'{path_key}: confidence levels differ')
            for key in ('seed', 'n_trials', 'n_bootstrap_samples'):
                if other_meta[key] != metadata[key]:
                    raise ValueError(f'{path_key}: mismatched {key}')
            np.testing.assert_allclose(data['q_matrix'], q, atol=1e-7, rtol=0)
            if path_key == 'no_restart_npz':
                np.testing.assert_array_equal(data['z_matrix'], z)
                methods['no_restart'] = data['bound__uniform_K100_norestart']
            else:
                methods['gt_simultaneous'] = data['gt_bonferroni_bound_matrix']
                methods['gt_pointwise'] = data['gt_upper_bound_matrix']

    eb = np.empty(z.shape, dtype=float)
    variance = np.empty(z.shape, dtype=float)
    batch_size = config['batch_size']
    for start in range(0, len(z), batch_size):
        end = min(start+batch_size, len(z))
        eb[start:end], variance[start:end] = empirical_bernstein_upper(
            z[start:end], alpha=config['alpha'], rho=config['rho'])
        print(f'Computed EB trials {start+1}-{end}/{len(z)}', flush=True)
    methods = {'restart': restart, 'eb': eb, **{k: v for k, v in methods.items() if k != 'restart'}}
    for key, matrix in methods.items():
        if matrix.shape != q.shape or not np.isfinite(matrix).all():
            raise ValueError(f'invalid bound matrix: {key}')

    output, artifact = ROOT/config['output_dir'], ROOT/config['artifact_dir']
    output.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    record = {
        'config': config, 'source_metadata': metadata,
        'source_sha256': {key: hashlib.sha256((ROOT/config[key]).read_bytes()).hexdigest()
                          for key in ('headline_npz', 'no_restart_npz', 'gt_npz')},
        'n_trials': len(z), 'horizon': z.shape[1],
        'oracle_final_mean': float(q[:, -1].mean()),
        'eb_predictor': 'strictly past sample mean; first prediction 0.5',
        'eb_boundary': 'Howard et al. 2021 Theorem 4 + Proposition 9; one-sided; c=1',
        'tuning': 'rho=100 fixed before comparison, no test-data optimization',
        'restricted_mean_definition': 'mean(min(tau, horizon)); not an estimate of uncensored mean',
        'paired_ci': 'normal approximation across independent paired trials, descriptive 95% intervals',
        'methods': {},
    }
    csv_rows = []
    for key, matrix in methods.items():
        coverage = float((matrix >= q).all(axis=1).mean())
        stopped = [] if key == 'gt_pointwise' else [
            stopping_statistics(matrix, q, eps, restart) for eps in config['thresholds']]
        record['methods'][key] = {
            'label': STYLES[key][0], 'final_mean_bound': float(matrix[:, -1].mean()),
            'simultaneous_coverage': coverage, 'stopping': stopped,
        }
        csv_rows.extend({'method': key, **row} for row in stopped)
    (output/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    with (output/'stopping.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(artifact/'trajectories.npz', metadata_json=json.dumps(record),
                        z_matrix=z, q_matrix=q, bound__empirical_bernstein=eb,
                        eb_variance=variance, **{f'bound__{k}': v for k, v in methods.items()})
    render(output, q, methods, record, config['alpha'])
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'stopping'}
                      for key, value in record['methods'].items()}, indent=2))
    print(f'Wrote figures and summary to {output}')


if __name__ == '__main__':
    main()
