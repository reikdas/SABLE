#!/usr/bin/env python3
import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

_HATCH_POOL = ['//', '||', '++', '--', 'xx', 'oo', '..', '**',
               '/', '+', '-', 'x']


def _assign_hatches(strats):
    all_strats = sorted(set(s for s in strats if s))
    return {s: _HATCH_POOL[i % len(_HATCH_POOL)] for i, s in enumerate(all_strats)}


def load_solver_data(results_json, algorithm):
    """Return {matrix: {speedup, strat, nnz, iterations}} for one solver."""
    with open(results_json) as f:
        entries = [e for e in json.load(f) if 'error' not in e
                   and e['algorithm'] == algorithm]
    composed, baseline = {}, {}
    for e in entries:
        m = e['matrix_name']
        if e['composition'] == 'vbr_csr':
            if m not in composed or e['total_time_ns'] < composed[m]['total_time_ns']:
                composed[m] = e
        elif e['composition'] == 'csr':
            if m not in baseline or e['total_time_ns'] < baseline[m]['total_time_ns']:
                baseline[m] = e

    data = {}
    for m, c in composed.items():
        b = baseline.get(m)
        if b is None:
            continue
        assert c['iterations'] == b['iterations'], (m, c['iterations'], b['iterations'])
        data[m] = {
            'speedup': b['total_time_ns'] / c['total_time_ns'],
            'strat': f"{c['kernels']['csr']} vs {b['kernels']['csr']}",
            'nnz': c['matrix']['nnz'],
            'iterations': c['iterations'],
        }
    return data


def plot_solvers(gd_data, jac_data, output_path, y_limit=1.7):
    matrices = sorted(gd_data, key=lambda m: gd_data[m]['nnz'])
    gd_vals = [gd_data[m]['speedup'] for m in matrices]
    jac_vals = [jac_data[m]['speedup'] for m in matrices]
    gd_strats = [gd_data[m]['strat'] for m in matrices]
    jac_strats = [jac_data[m]['strat'] for m in matrices]

    all_vals = gd_vals + jac_vals
    y_bottom = math.floor(min(all_vals) * 10) / 10.0
    print(f"Plotting {len(matrices)} matrices, speedups "
          f"[{min(all_vals):.3f}, {max(all_vals):.3f}]")

    plt.rcParams['hatch.linewidth'] = 0.5
    fig, ax = plt.subplots(figsize=(3.4, 1.9))

    x = np.arange(len(matrices))
    width = 0.35
    gd_bars = ax.bar(x - width / 2, gd_vals, width,
                     color='steelblue', edgecolor='navy', alpha=0.85,
                     linewidth=0.4)
    jac_bars = ax.bar(x + width / 2, jac_vals, width,
                      color='coral', edgecolor='darkred', alpha=0.85,
                      linewidth=0.4)

    hatch_map = _assign_hatches(gd_strats + jac_strats)
    for bars, strats in [(gd_bars, gd_strats), (jac_bars, jac_strats)]:
        for bar, strat in zip(bars, strats):
            bar.set_hatch(hatch_map[strat])

    label_pad = (y_limit - y_bottom) * 0.03
    for bars, vals in [(gd_bars, gd_vals), (jac_bars, jac_vals)]:
        for bar, v in zip(bars, vals):
            if v > y_limit:
                ax.text(bar.get_x() + bar.get_width() / 2., y_limit + label_pad,
                        f'{v:.1f}\u00d7', ha='center', va='bottom',
                        fontsize=4.5, fontweight='bold', rotation=90)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)

    margin = (y_limit - y_bottom) * 0.05
    ax.set_ylim(bottom=y_bottom, top=min(y_limit, max(all_vals) + margin) + margin)

    ax.set_xlabel('Matrix (sorted by nnz)', fontsize=7)
    ax.set_ylabel('Speedup (Best Baseline / Best SABLE)', fontsize=6)
    ax.tick_params(axis='y', labelsize=6, width=0.5, length=2)
    ax.tick_params(axis='x', width=0.4, length=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45, ha='right', fontsize=5)

    bar_handles = [
        Line2D([0], [0], color='steelblue', linewidth=5,
               solid_capstyle='butt', label='Gradient descent'),
        Line2D([0], [0], color='coral', linewidth=5,
               solid_capstyle='butt', label='Jacobi iteration'),
    ]
    hatch_handles = [Patch(facecolor='white', edgecolor='black', hatch=h, label=s)
                     for s, h in sorted(hatch_map.items())]
    all_handles = bar_handles + hatch_handles
    ax.legend(all_handles, [h.get_label() for h in all_handles],
              loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.02),
              fontsize=6, framealpha=0.9,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.3,
              borderpad=0.25, labelspacing=0.3)

    fig.savefig(output_path, bbox_inches='tight', dpi=300,
                metadata={'Creator': 'SABLE'})
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_stats(data, label):
    speedups = [d['speedup'] for d in data.values()]
    wins = sum(1 for s in speedups if s > 1.0)
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    iters = [d['iterations'] for d in data.values()]
    print(f"{label}: geomean = {geo:.4f}x over {len(speedups)} matrices "
          f"({wins} speedups, {len(speedups) - wins} slowdowns), "
          f"iterations {min(iters)}-{max(iters)}")


def main():
    parser = argparse.ArgumentParser(description="Chained-kernel speedup plots.")
    parser.add_argument('--results-json',
                        default=os.path.join(os.path.dirname(__file__),
                                             os.pardir, 'results',
                                             'iterative_eval.json'))
    parser.add_argument('--output-dir',
                        default=os.path.join(os.path.dirname(__file__), 'images'))
    args = parser.parse_args()

    gd_data = load_solver_data(args.results_json, 'gd')
    jac_data = load_solver_data(args.results_json, 'jacobi')
    plot_solvers(gd_data, jac_data,
                 os.path.join(args.output_dir, 'iterative_best_times.pdf'))
    print_stats(gd_data, 'gd')
    print_stats(jac_data, 'jacobi')


if __name__ == '__main__':
    main()
