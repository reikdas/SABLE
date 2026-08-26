#!/usr/bin/env python3
"""Merged SpMV+SpMM best-times-comparison plot.

Generates a single PDF with two bars per matrix (SpMV and SpMM speedup).
Bars exceeding the y-axis limit are clipped and labeled with their actual value.
"""

import json
import glob
import os
import argparse
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Hatching helpers
# ---------------------------------------------------------------------------

_HATCH_POOL = ['///', '|||', '+++', '---', 'xxx', 'ooo', '...', '**',
               '//', '++', '--', 'xx']


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


# ---------------------------------------------------------------------------
# SpMV data loading
# ---------------------------------------------------------------------------

def _parse_spmv_filename(filename):
    """Extract (dense, sparse, reordering) from sable_spmv_*.json filename."""
    basename = os.path.basename(filename)
    if not basename.startswith('sable_spmv_') or not basename.endswith('.json'):
        return None, None, None
    middle = basename[len('sable_spmv_'):-len('.json')]
    parts = middle.split('_')
    if len(parts) < 2:
        return None, None, None
    dense, sparse = parts[0], parts[1]
    allowed = {'mkl', 'naive', 'spv8', 'uzp'}
    if sparse not in allowed:
        return None, None, None
    reordering = '_'.join(parts[2:]) if len(parts) > 2 else 'none'
    return dense, sparse, reordering


def _get_timing_spmv(entry):
    """Return timing dict; SpMV may nest under '1 thread'."""
    timing = entry.get('timing', {})
    if '1 thread' in timing:
        return timing['1 thread']
    if timing.get('total_time_ns') is not None:
        return timing
    for v in timing.values():
        if isinstance(v, dict) and v.get('total_time_ns') is not None:
            return v
    return timing


def _entry_sort_key(entry, get_timing_fn):
    timing = get_timing_fn(entry)
    total_time = timing.get('total_time_ns')
    speedup = timing.get('speedup')
    if isinstance(total_time, (int, float)) and total_time > 0:
        return (0, total_time, 0)
    if isinstance(speedup, (int, float)):
        return (1, float('inf'), -speedup)
    return (2, float('inf'), 0)


def load_spmv_data(results_dir):
    pattern = os.path.join(results_dir, 'sable_spmv_*.json')
    all_data = defaultdict(dict)
    matrix_nnz = {}
    for json_file in glob.glob(pattern):
        dense, sparse, _ = _parse_spmv_filename(json_file)
        if dense is None or dense == 'naive':
            continue
        with open(json_file) as f:
            data = json.load(f)
        for entry in data:
            matrix = entry.get('matrix_name')
            if not matrix:
                continue
            key = (dense, sparse)
            existing = all_data[key].get(matrix)
            if existing is None or _entry_sort_key(entry, _get_timing_spmv) < _entry_sort_key(existing, _get_timing_spmv):
                all_data[key][matrix] = entry
            if matrix not in matrix_nnz:
                nnz = entry.get('matrix_dimensions', {}).get('nnz')
                if nnz is not None:
                    matrix_nnz[matrix] = nnz
    return all_data, matrix_nnz


# ---------------------------------------------------------------------------
# SpMM data loading
# ---------------------------------------------------------------------------

def _parse_spmm_filename(filename):
    basename = os.path.basename(filename)
    if not basename.startswith('sable_spmm_') or not basename.endswith('.json'):
        return None, None
    middle = basename[len('sable_spmm_'):-len('.json')]
    parts = middle.split('_')
    if len(parts) >= 3:
        with open(filename) as f:
            data = json.load(f)
        if len(data) == 1 and data[0].get('matrix_name') == parts[2]:
            return None, None
        return parts[0], '_'.join(parts[1:])
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _get_timing_spmm(entry):
    """Return timing dict; SpMM may nest under '1 thread'."""
    timing = entry.get('timing', {})
    if '1 thread' in timing:
        return timing['1 thread']
    if timing.get('total_time_ns') is not None:
        return timing
    for v in timing.values():
        if isinstance(v, dict) and v.get('total_time_ns') is not None:
            return v
    return timing


def load_spmm_data(results_dir):
    pattern = os.path.join(results_dir, 'sable_spmm_*.json')
    all_data = defaultdict(dict)
    matrix_nnz = {}
    for json_file in glob.glob(pattern):
        dense, sparse = _parse_spmm_filename(json_file)
        if dense is None or dense == 'naive':
            continue
        with open(json_file) as f:
            data = json.load(f)
        for entry in data:
            matrix = entry.get('matrix_name')
            if not matrix:
                continue
            all_data[(dense, sparse)][matrix] = entry
            if matrix not in matrix_nnz:
                nnz = entry.get('matrix_dimensions', {}).get('nnz')
                if nnz is not None:
                    matrix_nnz[matrix] = nnz
    return all_data, matrix_nnz


# ---------------------------------------------------------------------------
# Best speedup computation (per-operation)
# ---------------------------------------------------------------------------

def compute_best_speedups(all_data, get_timing_fn, exclude_matrices):
    """For each matrix, find best SABLE time and best fully-sparse time.

    Returns:
        speedups: {matrix: speedup}
        strategies: {matrix: "dense/sparse vs baseline"}
    """
    best_total = {}   # {matrix: (time, dense, sparse)}
    best_sparse = {}  # {matrix: (time, sparse)}

    for (dense, sparse), matrices in all_data.items():
        for matrix, entry in matrices.items():
            if matrix in exclude_matrices:
                continue
            timing = get_timing_fn(entry)
            total_time = timing.get('total_time_ns')
            fully_sparse = timing.get('fully_sparse_time')

            if total_time is not None and total_time > 0:
                if matrix not in best_total or total_time < best_total[matrix][0]:
                    best_total[matrix] = (total_time, dense, sparse)

            if fully_sparse is not None and fully_sparse > 0:
                if matrix not in best_sparse or fully_sparse < best_sparse[matrix][0]:
                    best_sparse[matrix] = (fully_sparse, sparse)

    speedups = {}
    strategies = {}
    for matrix in best_total:
        if matrix in best_sparse and best_total[matrix][0] > 0 and best_sparse[matrix][0] > 0:
            speedups[matrix] = best_sparse[matrix][0] / best_total[matrix][0]
            _, dense, sparse = best_total[matrix]
            _, baseline = best_sparse[matrix]
            strategies[matrix] = f'{sparse} vs {baseline}'
    return speedups, strategies


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------

def plot_combined(spmv_speedups, spmv_strategies, spmm_speedups, spmm_strategies,
                  matrix_nnz, output_path):
    all_matrices = set(spmv_speedups.keys()) | set(spmm_speedups.keys())
    sorted_matrices = sorted(all_matrices, key=lambda m: matrix_nnz.get(m, float('inf')))

    matrix_labels = []
    spmv_vals = []
    spmm_vals = []
    spmv_strats = []
    spmm_strats = []
    for m in sorted_matrices:
        sv = spmv_speedups.get(m)
        sm = spmm_speedups.get(m)
        if sv is not None or sm is not None:
            matrix_labels.append(m)
            spmv_vals.append(sv if sv is not None else 0)
            spmm_vals.append(sm if sm is not None else 0)
            spmv_strats.append(spmv_strategies.get(m, ''))
            spmm_strats.append(spmm_strategies.get(m, ''))

    if not matrix_labels:
        print("No valid data for combined plot")
        return

    all_vals = [v for v in spmv_vals + spmm_vals if v > 0]
    min_speed = min(all_vals)
    max_speed = max(all_vals)
    y_bottom = math.floor(min_speed * 10) / 10.0
    y_limit = 1.5

    print(f"Plotting {len(matrix_labels)} matrices, speedups [{min_speed:.3f}, {max_speed:.3f}]")

    fig, ax = plt.subplots(figsize=(max(14, len(matrix_labels) * 0.45), 5))

    x = np.arange(len(matrix_labels))
    width = 0.35

    spmv_bars = ax.bar(x - width / 2, spmv_vals, width, color='steelblue', edgecolor='navy', alpha=0.85)
    spmm_bars = ax.bar(x + width / 2, spmm_vals, width, color='coral', edgecolor='darkred', alpha=0.85)

    # Assign hatching per strategy pair
    hatch_map = _assign_hatches(spmv_strats, spmm_strats)
    _apply_hatches(spmv_bars, spmv_strats, hatch_map)
    _apply_hatches(spmm_bars, spmm_strats, hatch_map)

    # For clipped bars, label with actual value only
    # Offset SpMM label when both SpMV and SpMM are clipped at same matrix
    for i in range(len(matrix_labels)):
        sv_clipped = spmv_vals[i] > y_limit
        sm_clipped = spmm_vals[i] > y_limit
        offset = 0.04 if sv_clipped and sm_clipped else 0
        if sv_clipped:
            bar = spmv_bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2., y_limit,
                    f'{spmv_vals[i]:.1f}×', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        if sm_clipped:
            bar = spmm_bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2., y_limit + offset,
                    f'{spmm_vals[i]:.1f}×', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Breakeven line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2)

    margin = (y_limit - y_bottom) * 0.05
    ax.set_ylim(bottom=y_bottom, top=y_limit + margin)

    yticks = list(ax.get_yticks())
    if y_bottom not in yticks:
        yticks.append(y_bottom)
        yticks.sort()
    ax.set_yticks(yticks)

    ax.set_xlabel('Matrix (sorted by nnz)', fontsize=12)
    ax.set_ylabel('Speedup (Best Fully Sparse / Best SABLE)', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(matrix_labels, rotation=45, ha='right', fontsize=9)

    # Combined legend: plain color patches for SpMV/SpMM, hatched for strategies
    bar_handles = [
        Line2D([0], [0], color='steelblue', linewidth=8, solid_capstyle='butt', label='SpMV'),
        Line2D([0], [0], color='coral', linewidth=8, solid_capstyle='butt', label='SpMM'),
    ]
    hatch_handles = _hatch_legend_handles(hatch_map)
    all_handles = bar_handles + hatch_handles
    all_labels = [h.get_label() for h in all_handles]
    ncol = min(2 + len(hatch_handles), 4)
    ax.legend(all_handles, all_labels, loc='upper center', ncol=ncol,
              fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.30))
    fig.subplots_adjust(top=0.80)

    # Clip bars exceeding y_limit
    if max_speed > y_limit:
        ax.set_ylim(top=y_limit)

    fig.savefig(output_path, bbox_inches='tight', dpi=300,
                metadata={'Creator': 'SABLE'})
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combined SpMV+SpMM speedup plot.")
    parser.add_argument('--results-dir', default='results',
                        help='Directory with sable_sp{mv,mm}_*.json files')
    parser.add_argument('--output', default=None,
                        help='Output PDF path (default: images/best_times_comparison.pdf)')
    args = parser.parse_args()

    results_dir = args.results_dir
    output = args.output or os.path.join('images', 'best_times_comparison.pdf')

    print("Loading SpMV data...")
    spmv_all, spmv_nnz = load_spmv_data(results_dir)
    spmv_speedups, spmv_strategies = compute_best_speedups(spmv_all, _get_timing_spmv, set())
    print(f"  {len(spmv_speedups)} matrices")

    print("Loading SpMM data...")
    spmm_all, spmm_nnz = load_spmm_data(results_dir)
    spmm_speedups_all, spmm_strategies_all = compute_best_speedups(spmm_all, _get_timing_spmm, set())
    # Filter SpMM to only include matrices in the SpMV evaluation set
    spmm_speedups = {m: v for m, v in spmm_speedups_all.items() if m in spmv_speedups}
    spmm_strategies = {m: v for m, v in spmm_strategies_all.items() if m in spmv_speedups}
    print(f"  {len(spmm_speedups)} matrices (filtered to SpMV set)")

    matrix_nnz = {**spmv_nnz, **spmm_nnz}

    plot_combined(spmv_speedups, spmv_strategies, spmm_speedups, spmm_strategies,
                  matrix_nnz, output)

    # Print geometric means
    if spmv_speedups:
        spmv_geo = math.exp(sum(math.log(v) for v in spmv_speedups.values()) / len(spmv_speedups))
        spmv_gt1 = sum(1 for v in spmv_speedups.values() if v > 1.0)
        spmv_le1 = len(spmv_speedups) - spmv_gt1
        print(f"\nSpMV: geomean = {spmv_geo:.4f}x over {len(spmv_speedups)} matrices "
              f"({spmv_gt1} speedups, {spmv_le1} slowdowns)")
    if spmm_speedups:
        spmm_geo = math.exp(sum(math.log(v) for v in spmm_speedups.values()) / len(spmm_speedups))
        spmm_gt1 = sum(1 for v in spmm_speedups.values() if v > 1.0)
        spmm_le1 = len(spmm_speedups) - spmm_gt1
        print(f"SpMM: geomean = {spmm_geo:.4f}x over {len(spmm_speedups)} matrices "
              f"({spmm_gt1} speedups, {spmm_le1} slowdowns)")

    print("\nDone!")


if __name__ == '__main__':
    main()
