#!/usr/bin/env python3
"""Restyle saved fixed-Q results using the main paper-figure plotting helpers.

No simulation is run. Exact coverage/stopping arrays are used, not rounded UCBs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting import plot_av_pipeline as style

ARM_COLORS = {'P': style.RESTART_COLOR, 'Q': '#17365D'}
ARM_LABELS = {'P': r'Sampling from $P$', 'Q': r'Sampling from $Q$'}


def paper_style():
    return plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': style.FONT_BASE,
        'axes.labelsize': style.FONT_AXIS_LABEL,
        'xtick.labelsize': style.FONT_TICK,
        'ytick.labelsize': style.FONT_TICK,
        'legend.fontsize': style.FONT_LEGEND,
        'axes.labelpad': 14,
    })


def finish_axes(ax, horizon):
    ax.set_xlim(0, horizon)
    ax.set_xlabel('Number of sampled sessions, $t$')
    ax.set_xticks(np.linspace(0, horizon, 7))
    ax.grid(True, alpha=.2, linewidth=.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save(fig, out, name):
    fig.tight_layout()
    style.save_figure(fig, out, name)
    plt.close(fig)


def bounds_panel(ax, arm, truth, upper, ci_level):
    x = np.arange(1, upper.shape[1] + 1)
    qm, qlo, qhi = style.percentile_band(truth, ci_level)
    um, ulo, uhi = style.percentile_band(upper, ci_level)
    style.plot_bounds(ax, x, qm, qlo, qhi, ci_level, None, um, ulo, uhi)
    # The method is identical in both panels; color identifies sampling distribution.
    ax.lines[0].set_color(ARM_COLORS[arm])
    ax.collections[0].set_color(ARM_COLORS[arm])
    ax.set_title(ARM_LABELS[arm], pad=16)
    ax.legend(loc='upper right', bbox_to_anchor=(.98, .98), frameon=False)


def render_main(results, config, out, ci_level=.95):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(2 * style.FIGSIZE[0], style.FIGSIZE[1]))
        for ax, arm in zip(axes, ('P', 'Q')):
            truth, upper, _, _ = results[arm]
            bounds_panel(ax, arm, truth, upper, ci_level)
        save(fig, out, 'upper_bounds')
        for arm in ('P', 'Q'):
            fig, ax = plt.subplots(figsize=style.FIGSIZE)
            truth, upper, _, _ = results[arm]
            bounds_panel(ax, arm, truth, upper, ci_level)
            save(fig, out, f'upper_bounds_{arm}')

        fig, ax = plt.subplots(figsize=style.FIGSIZE)
        minimum = 1.
        for arm in ('P', 'Q'):
            covered = results[arm][2]
            rate = covered.mean(axis=0)
            minimum = min(minimum, float(rate.min()))
            ax.plot(np.arange(1, len(rate) + 1), rate, color=ARM_COLORS[arm],
                    linewidth=3.2, label=ARM_LABELS[arm])
        nominal = 1 - config['alpha']
        ax.axhline(nominal, color=style.NOMINAL_COVERAGE_COLOR, linewidth=1.6,
                   linestyle=(0, (2, 2)), zorder=1)
        ax.annotate(r'nominal $1-\alpha$', xy=(config['horizon'], nominal),
                    xytext=(-8, 6), textcoords='offset points', ha='right', va='bottom',
                    color=style.NOMINAL_COVERAGE_COLOR, fontsize=style.FONT_LEGEND)
        floor = min(.9, minimum - .01)
        ax.set_ylim(floor, 1 + (1 - floor) * .02)
        ax.set_yticks(np.round(np.arange(1., floor - 1e-9, -.02)[::-1], 2))
        ax.set_ylabel('Simultaneous coverage')
        finish_axes(ax, config['horizon'])
        ax.legend(loc='lower left', frameon=False)
        save(fig, out, 'simultaneous_coverage')
    render_stopping({arm: results[arm][3] for arm in ('P', 'Q')},
                    config['stopping_epsilons'], config['horizon'], out)


def stopping_panel(ax, all_stops, j, epsilon, horizon):
    for arm in ('P', 'Q'):
        stops = all_stops[arm][:, j]
        observed = np.sort(stops[stops <= horizon])
        x = np.r_[0, observed, horizon]
        cdf = np.r_[0, np.arange(1, len(observed) + 1) / len(stops), len(observed) / len(stops)]
        ax.step(x, cdf, where='post', linewidth=3.2, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.set_title(rf'$\varepsilon={epsilon:g}$', pad=16)
    ax.set_ylim(-.02, 1.02)
    ax.set_yticks(np.arange(0, 1.01, .2))
    finish_axes(ax, horizon)
    ax.set_xticks(np.linspace(0, horizon, 4))
    median_stop = np.median(np.concatenate([all_stops[arm][:, j] for arm in ('P', 'Q')]))
    ax.legend(loc='upper left' if median_stop > horizon / 2 else 'lower right', frameon=False)


def render_stopping(all_stops, epsilons, horizon, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with paper_style():
        fig, axes = plt.subplots(1, len(epsilons), figsize=(style.FIGSIZE[0] * len(epsilons),
                                                         style.FIGSIZE[1]), squeeze=False)
        for j, epsilon in enumerate(epsilons):
            stopping_panel(axes[0, j], all_stops, j, epsilon, horizon)
        axes[0, 0].set_ylabel('Fraction of runs stopped by $t$')
        save(fig, out, 'stopping_times')
        for j, epsilon in enumerate(epsilons):
            fig, ax = plt.subplots(figsize=style.FIGSIZE)
            stopping_panel(ax, all_stops, j, epsilon, horizon)
            ax.set_ylabel('Fraction of runs stopped by $t$')
            save(fig, out, f'stopping_times_epsilon_{epsilon:g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stopping-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / 'summary.json').read_text())
    stopping_summary = json.loads((args.stopping_dir / 'summary.json').read_text())
    if Path(stopping_summary['source_run']).resolve() != args.run_dir.resolve():
        raise ValueError('Stopping extension belongs to another run')
    files = [args.run_dir / 'summary.json', args.stopping_dir / 'summary.json']
    results, all_stops = {}, {}
    for arm in ('P', 'Q'):
        path = args.run_dir / f'{arm}_trajectories.npz'
        files.append(path)
        with np.load(path) as data:
            results[arm] = tuple(data[key] for key in ['true_missing_mass', 'upper',
                                                       'simultaneous_covered', 'stopping_times'])
        path = args.stopping_dir / f'{arm}_stopping.npz'
        files.append(path)
        with np.load(path) as data:
            all_stops[arm] = data['stopping_times']
            np.testing.assert_array_equal(data['epsilons'], summary['config']['stopping_epsilons'])
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    render_main(results, summary['config'], args.run_dir)
    render_stopping(all_stops, summary['config']['stopping_epsilons'],
                    stopping_summary['horizon'], args.stopping_dir)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if before != after:
        raise RuntimeError('Source results changed during plotting')
    manifest = {
        'script': str(Path(__file__).resolve()),
        'style_source': str(Path(style.__file__).resolve()),
        'source_sha256': before,
        'central_trajectory_band': .95,
        'coverage': 'Exact saved simultaneous coverage; no Monte Carlo shading, matching main plots.',
        'colors': ARM_COLORS,
        'outputs': [str(p) for directory in (args.run_dir, args.stopping_dir)
                    for p in sorted(directory.glob('*.png'))],
    }
    (args.run_dir / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
