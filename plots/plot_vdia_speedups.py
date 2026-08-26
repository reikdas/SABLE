#!/usr/bin/env python3
"""VDIA best-times-comparison plots.

Generates bar-chart PDFs analogous to best_times_comparison.pdf (VBR+CSR)
for two VDIA configurations:
  1. VDIA+CSR vs CSR
  2. VDIA+VBR+CSR vs VBR+CSR
"""

import json
import os
import math
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

from fukaya_results import EXCLUDED_MATRICES, load_fukaya


# ---------------------------------------------------------------------------
# Hatching helpers
# ---------------------------------------------------------------------------

# Single-density patterns: at single-column size, dense hatches ('///')
# smear into solid fill.
_HATCH_POOL = ['//', '||', '++', '--', 'xx', 'oo', '..', '**',
               '/', '+', '-', 'x']


def _assign_hatches(*strat_lists):
    """Build a mapping from strategy string -> hatch pattern."""
    all_strats = sorted(set(s for sl in strat_lists for s in sl if s))
    return {s: _HATCH_POOL[i % len(_HATCH_POOL)] for i, s in enumerate(all_strats)}


def _apply_hatches(bars, strats, hatch_map):
    """Set hatch pattern on each bar according to its strategy."""
    for bar, strat in zip(bars, strats):
        if strat and strat in hatch_map:
            bar.set_hatch(hatch_map[strat])


def _hatch_legend_handles(hatch_map):
    """Create legend handles for strategy hatching."""
    return [Patch(facecolor='white', edgecolor='black', hatch=h, label=s)
            for s, h in sorted(hatch_map.items(), key=lambda x: x[0])]


SPMV_BACKENDS = ['naive', 'mkl', 'spv8']
SPMM_BACKENDS = ['naive', 'mkl']


def _best_from_per_backend(per_backend, backends):
    best_vdia_us = None
    best_vdia_backend = None
    best_baseline_us = None
    best_baseline_backend = None
    for b in backends:
        if b not in per_backend:
            continue
        info = per_backend[b]
        vdia = info['vdia_us']
        baseline = info['baseline_us']
        if vdia > 0 and (best_vdia_us is None or vdia < best_vdia_us):
            best_vdia_us = vdia
            best_vdia_backend = b
        if baseline > 0 and (best_baseline_us is None or baseline < best_baseline_us):
            best_baseline_us = baseline
            best_baseline_backend = b
    if best_vdia_us is None or best_baseline_us is None:
        return None
    return {
        'speedup': best_baseline_us / best_vdia_us,
        'best_vdia_backend': best_vdia_backend,
        'best_baseline_backend': best_baseline_backend,
    }


def load_vdia_data(results_dir, spmv_file, spmm_file):
    spmv_path = os.path.join(results_dir, spmv_file)
    spmm_path = os.path.join(results_dir, spmm_file)

    with open(spmv_path) as f:
        spmv_raw = {m['matrix_name']: m for m in json.load(f)
                    if m['matrix_name'] not in EXCLUDED_MATRICES}
    with open(spmm_path) as f:
        spmm_raw = {m['matrix_name']: m for m in json.load(f)
                    if m['matrix_name'] not in EXCLUDED_MATRICES}

    for d, backends in [(spmv_raw, SPMV_BACKENDS), (spmm_raw, SPMM_BACKENDS)]:
        for name, m in d.items():
            computed = _best_from_per_backend(m['per_backend'], backends)
            if computed:
                m['best_vs_best'] = computed

    return spmv_raw, spmm_raw


def plot_vdia(spmv_data, spmm_data, output_path, y_limit=1.5):
    all_matrices = set(spmv_data.keys()) | set(spmm_data.keys())
    sorted_matrices = sorted(all_matrices,
                             key=lambda m: spmv_data.get(m, spmm_data.get(m, {})).get('nnz', float('inf')))

    matrix_labels = []
    spmv_vals = []
    spmm_vals = []
    spmv_strats = []
    spmm_strats = []

    for m in sorted_matrices:
        sv = spmv_data.get(m, {}).get('best_vs_best')
        sm = spmm_data.get(m, {}).get('best_vs_best')
        if sv is not None or sm is not None:
            matrix_labels.append(m)
            spmv_vals.append(sv['speedup'] if sv else 0)
            spmm_vals.append(sm['speedup'] if sm else 0)
            spmv_strats.append(
                f"{sv['best_vdia_backend']} vs {sv['best_baseline_backend']}"
                if sv else ''
            )
            spmm_strats.append(
                f"{sm['best_vdia_backend']} vs {sm['best_baseline_backend']}"
                if sm else ''
            )

    if not matrix_labels:
        print("No valid data")
        return

    all_vals = [v for v in spmv_vals + spmm_vals if v > 0]
    min_speed = min(all_vals)
    max_speed = max(all_vals)
    y_bottom = math.floor(min_speed * 10) / 10.0

    print(f"Plotting {len(matrix_labels)} matrices, speedups [{min_speed:.3f}, {max_speed:.3f}]")

    # Single-column figure: render at the physical size it will occupy
    # (\columnwidth is ~3.33in in acmart), so fonts below are true-size.
    plt.rcParams['hatch.linewidth'] = 0.5
    fig, ax = plt.subplots(figsize=(3.4, 1.9))

    x = np.arange(len(matrix_labels))
    width = 0.35

    spmv_bars = ax.bar(x - width / 2, spmv_vals, width,
                        color='steelblue', edgecolor='navy', alpha=0.85,
                        linewidth=0.4)
    spmm_bars = ax.bar(x + width / 2, spmm_vals, width,
                        color='coral', edgecolor='darkred', alpha=0.85,
                        linewidth=0.4)

    # Assign hatching per strategy pair
    hatch_map = _assign_hatches(spmv_strats, spmm_strats)
    _apply_hatches(spmv_bars, spmv_strats, hatch_map)
    _apply_hatches(spmm_bars, spmm_strats, hatch_map)

    # For clipped bars, label with actual value only
    # Offset SpMM label when both SpMV and SpMM are clipped at same matrix
    label_pad = (y_limit - y_bottom) * 0.03  # gap above the top spine
    for i in range(len(matrix_labels)):
        sv_clipped = spmv_vals[i] > y_limit
        sm_clipped = spmm_vals[i] > y_limit
        if sv_clipped:
            bar = spmv_bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2., y_limit + label_pad,
                    f'{spmv_vals[i]:.1f}×', ha='center', va='bottom',
                    fontsize=4.5, fontweight='bold', rotation=90)
        if sm_clipped:
            bar = spmm_bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2., y_limit + label_pad,
                    f'{spmm_vals[i]:.1f}×', ha='center', va='bottom',
                    fontsize=4.5, fontweight='bold', rotation=90)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)

    margin = (y_limit - y_bottom) * 0.05
    ax.set_ylim(bottom=y_bottom, top=y_limit + margin)

    yticks = list(ax.get_yticks())
    if y_bottom not in yticks:
        yticks.append(y_bottom)
        yticks.sort()
    ax.set_yticks(yticks)

    ax.set_xlabel('Matrix (sorted by nnz)', fontsize=7)
    ax.set_ylabel('Speedup (Best Baseline / Best SABLE)', fontsize=6)
    ax.tick_params(axis='y', labelsize=6, width=0.5, length=2)
    ax.tick_params(axis='x', width=0.4, length=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(matrix_labels, rotation=45, ha='right', fontsize=5)

    # Combined legend: operation type + strategy hatching.
    # Single row: ncol equals the number of handles, with tightened
    # horizontal spacing so all entries fit on one line.
    bar_handles = [
        Line2D([0], [0], color='steelblue', linewidth=5, solid_capstyle='butt', label='SpMV'),
        Line2D([0], [0], color='coral', linewidth=5, solid_capstyle='butt', label='SpMM'),
    ]
    hatch_handles = _hatch_legend_handles(hatch_map)
    all_handles = bar_handles + hatch_handles
    all_labels = [h.get_label() for h in all_handles]
    ax.legend(all_handles, all_labels, loc='lower center',
              ncol=3,
              bbox_to_anchor=(0.5, 1.17),
              fontsize=6, framealpha=0.9,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.3,
              borderpad=0.25, labelspacing=0.3)

    if max_speed > y_limit:
        ax.set_ylim(top=y_limit)

    fig.savefig(output_path, bbox_inches='tight', dpi=300,
                metadata={'Creator': 'SABLE'})
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_stats(data, label):
    speedups = [m['best_vs_best']['speedup'] for m in data.values()]
    wins = sum(1 for s in speedups if s > 1.0)
    losses = len(speedups) - wins
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print(f"{label}: geomean = {geo:.4f}x over {len(speedups)} matrices "
          f"({wins} speedups, {losses} slowdowns)")


def main():
    parser = argparse.ArgumentParser(description="VDIA speedup plots.")
    parser.add_argument('--results-dir',
                        default=os.path.join(os.path.dirname(__file__),
                                             'results'),
                        help='Directory with VDIA JSON files')
    parser.add_argument('--output-dir',
                        default=os.path.join(os.path.dirname(__file__), 'images'),
                        help='Output directory for PDFs')
    args = parser.parse_args()

    print("=== VDIA+CSR vs CSR ===")
    spmv_vc, spmm_vc = load_vdia_data(
        args.results_dir,
        'spmv_vdia_csr_d075.json',
        'spmm_vdia_csr_d075.json',
    )
    spmv_vc.update(load_fukaya(args.results_dir, 'spmv'))
    spmm_vc.update(load_fukaya(args.results_dir, 'spmm'))
    plot_vdia(spmv_vc, spmm_vc,
              os.path.join(args.output_dir, 'vdia_csr_best_times.pdf'))
    print_stats(spmv_vc, 'SpMV: VDIA+CSR vs CSR')
    print_stats(spmm_vc, 'SpMM: VDIA+CSR vs CSR')

    print("\n=== VDIA+VBR+CSR vs VBR+CSR ===")
    spmv_vvc, spmm_vvc = load_vdia_data(
        args.results_dir,
        'spmv_vdia_vbr_csr_d075.json',
        'spmm_vdia_vbr_csr_d075.json',
    )
    plot_vdia(spmv_vvc, spmm_vvc,
              os.path.join(args.output_dir, 'vdia_vbr_csr_best_times.pdf'))
    print_stats(spmv_vvc, 'SpMV: VDIA+VBR+CSR vs VBR+CSR')
    print_stats(spmm_vvc, 'SpMM: VDIA+VBR+CSR vs VBR+CSR')

    print("\nDone!")


if __name__ == '__main__':
    main()
