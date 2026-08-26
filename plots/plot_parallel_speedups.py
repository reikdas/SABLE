#!/usr/bin/env python3
"""Parallel SpMV+SpMM analysis and plotting.

1. Generates a bar chart comparing SABLE vs best fully-sparse baseline
   at 8 threads for both SpMV and SpMM.
2. Generates a 4-subplot figure showing SpMV thread scaling for selected matrices.
"""

import json
import os
import math
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np


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


THREAD_COUNTS = ['1 thread', '2 thread', '4 thread', '8 thread', '12 thread']
PARALLEL_TC = '8 thread'
SPMV_FILES = {
    'mkl':   'sable_spmv_blas_mkl.json',
    'naive': 'sable_spmv_blas_naive.json',
    'spv8':  'sable_spmv_blas_spv8.json',
    'uzp':   'sable_spmv_blas_uzp.json',
}
SPMM_FILES = {
    'mkl':   'sable_spmm_mkl_mkl.json',
    'naive': 'sable_spmm_mkl_naive.json',
}
BASELINE_DISPLAY = {
    'mkl': 'MKL', 'naive': 'Naive', 'spv8': 'SpV8', 'uzp': 'UZP',
}
EXCLUDE = set()
MIN_BLOCK_COVERAGE = 15.0
MAX_BLOCK_ASPECT_RATIO = 100.0
# Colorblind-safe palette (Wong 2011)
COLORS = {
    'mkl': '#0072B2',
    'naive': '#D55E00',
    'spv8': '#009E73',
    'uzp': '#CC79A7',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_spmv(results_dir):
    """Load all SpMV JSON files.

    Returns: {baseline: [entry, ...]} where each entry has timing with thread keys.
    """
    all_data = {}
    for baseline, fname in SPMV_FILES.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        with open(path) as f:
            all_data[baseline] = json.load(f)
    return all_data


def load_all_spmm(results_dir):
    """Load all SpMM JSON files.

    Returns: {baseline: [entry, ...]} where each entry has flat timing.
    """
    all_data = {}
    for baseline, fname in SPMM_FILES.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        with open(path) as f:
            all_data[baseline] = json.load(f)
    return all_data


def get_matrix_nnz(*data_dicts):
    """Extract matrix nnz mapping from any number of {baseline: [entry, ...]} dicts."""
    nnz_map = {}
    for all_data in data_dicts:
        for baseline, entries in all_data.items():
            for entry in entries:
                name = entry['matrix_name']
                if name not in nnz_map:
                    nnz = entry.get('matrix_dimensions', {}).get('nnz')
                    if nnz is not None:
                        nnz_map[name] = nnz
    return nnz_map


def get_parallel_eligible_matrices(*data_dicts):
    """Return the set of matrices that pass the parallelization filters.

    A matrix is eligible if:
    1. Its dense blocks cover at least MIN_BLOCK_COVERAGE % of total nnz, AND
    2. No individual block has an aspect ratio exceeding MAX_BLOCK_ASPECT_RATIO.
    """
    eligible = set()
    for all_data in data_dicts:
        for baseline, entries in all_data.items():
            for entry in entries:
                name = entry['matrix_name']
                if name in eligible:
                    continue
                timing_1t = entry.get('timing', {}).get('1 thread', {})
                blocks = timing_1t.get('individual_dense_block_timings', {})
                if not blocks:
                    continue
                total_pct = sum(b.get('percentage_of_total_nnz', 0) for b in blocks.values())
                if total_pct < MIN_BLOCK_COVERAGE:
                    continue
                worst_aspect = 0
                for b in blocks.values():
                    r, c = b.get('rows', 1), b.get('cols', 1)
                    aspect = max(r, c) / max(min(r, c), 1)
                    worst_aspect = max(worst_aspect, aspect)
                if worst_aspect <= MAX_BLOCK_ASPECT_RATIO:
                    eligible.add(name)
    return eligible


# ---------------------------------------------------------------------------
# Speedup computation
# ---------------------------------------------------------------------------

def compute_spmv_speedups(all_data, tc_key):
    """Compute per-matrix speedup of best SABLE over best fully-sparse at tc_key.

    Returns: {matrix: {'speedup': float, 'sable_baseline': str, 'sparse_baseline': str, ...}}
    """
    matrices = set()
    for entries in all_data.values():
        for e in entries:
            matrices.add(e['matrix_name'])
    matrices -= EXCLUDE

    speedups = {}
    for matrix in matrices:
        best_sable_time = None
        best_sparse_time = None
        best_sable_baseline = None
        best_sparse_baseline = None

        for baseline, entries in all_data.items():
            for entry in entries:
                if entry['matrix_name'] != matrix:
                    continue
                timing = entry['timing'].get(tc_key, {})
                total = timing.get('total_time_ns')
                fully_sparse = timing.get('fully_sparse_time')

                if total is not None and total > 0 and (best_sable_time is None or total < best_sable_time):
                    best_sable_time = total
                    best_sable_baseline = baseline

                if fully_sparse is not None and fully_sparse > 0 and (best_sparse_time is None or fully_sparse < best_sparse_time):
                    best_sparse_time = fully_sparse
                    best_sparse_baseline = baseline

        if best_sable_time and best_sparse_time and best_sable_time > 0:
            speedups[matrix] = {
                'speedup': best_sparse_time / best_sable_time,
                'sable_baseline': best_sable_baseline,
                'sparse_baseline': best_sparse_baseline,
                'sable_time': best_sable_time,
                'sparse_time': best_sparse_time,
            }

    return speedups


def compute_spmm_speedups(all_data, tc_key):
    """Compute per-matrix speedup of best SABLE over best fully-sparse for SpMM at tc_key.

    Returns: {matrix: {'speedup': float, 'sable_baseline': str, 'sparse_baseline': str, ...}}
    """
    matrices = set()
    for entries in all_data.values():
        for e in entries:
            matrices.add(e['matrix_name'])
    matrices -= EXCLUDE

    speedups = {}
    for matrix in matrices:
        best_sable_time = None
        best_sparse_time = None
        best_sable_baseline = None
        best_sparse_baseline = None

        for baseline, entries in all_data.items():
            for entry in entries:
                if entry['matrix_name'] != matrix:
                    continue
                timing = entry['timing'].get(tc_key, {})
                total = timing.get('total_time_ns')
                fully_sparse = timing.get('fully_sparse_time')

                if total is not None and total > 0 and (best_sable_time is None or total < best_sable_time):
                    best_sable_time = total
                    best_sable_baseline = baseline

                if fully_sparse is not None and fully_sparse > 0 and (best_sparse_time is None or fully_sparse < best_sparse_time):
                    best_sparse_time = fully_sparse
                    best_sparse_baseline = baseline

        if best_sable_time and best_sparse_time and best_sable_time > 0:
            speedups[matrix] = {
                'speedup': best_sparse_time / best_sable_time,
                'sable_baseline': best_sable_baseline,
                'sparse_baseline': best_sparse_baseline,
                'sable_time': best_sable_time,
                'sparse_time': best_sparse_time,
            }

    return speedups


# ---------------------------------------------------------------------------
# Step 1: Parallel best-times-comparison bar chart (Figure 8)
# ---------------------------------------------------------------------------

def plot_parallel_bar(spmv_data, spmm_data, matrix_nnz, output_path):
    """Plot a paired bar chart showing SpMV and SpMM speedups at 8 threads."""
    all_matrices = set(spmv_data.keys()) | set(spmm_data.keys())
    sorted_matrices = sorted(all_matrices, key=lambda m: matrix_nnz.get(m, float('inf')))

    labels = []
    spmv_vals = []
    spmm_vals = []
    spmv_strats = []
    spmm_strats = []
    for m in sorted_matrices:
        sv = spmv_data.get(m)
        sm = spmm_data.get(m)
        if sv is not None or sm is not None:
            labels.append(m)
            spmv_vals.append(sv['speedup'] if sv is not None else 0)
            spmm_vals.append(sm['speedup'] if sm is not None else 0)
            spmv_strats.append(
                f"{BASELINE_DISPLAY.get(sv['sable_baseline'], sv['sable_baseline'])} vs "
                f"{BASELINE_DISPLAY.get(sv['sparse_baseline'], sv['sparse_baseline'])}"
                if sv else ''
            )
            spmm_strats.append(
                f"{BASELINE_DISPLAY.get(sm['sable_baseline'], sm['sable_baseline'])} vs "
                f"{BASELINE_DISPLAY.get(sm['sparse_baseline'], sm['sparse_baseline'])}"
                if sm else ''
            )

    if not labels:
        print("No data to plot")
        return

    all_vals = [v for v in spmv_vals + spmm_vals if v > 0]
    min_speed = min(all_vals)
    max_speed = max(all_vals)
    y_bottom = math.floor(min_speed * 10) / 10.0
    y_limit = 1.6

    print(f"Plotting {len(labels)} matrices, speedups [{min_speed:.3f}, {max_speed:.3f}]")

    # Single-column figure: render at the physical size it will occupy
    # (\columnwidth is ~3.33in in acmart), so fonts below are true-size.
    plt.rcParams['hatch.linewidth'] = 0.5
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    x = np.arange(len(labels))
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
    for i in range(len(labels)):
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
    ax.set_ylabel('Speedup (Best Fully Sparse / Best SABLE)', fontsize=6)
    ax.tick_params(axis='y', labelsize=6, width=0.5, length=2)
    ax.tick_params(axis='x', width=0.4, length=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)

    # Combined legend: plain color patches for SpMV/SpMM, hatched for strategies
    bar_handles = [
        Line2D([0], [0], color='steelblue', linewidth=5, solid_capstyle='butt', label='SpMV (8 threads)'),
        Line2D([0], [0], color='coral', linewidth=5, solid_capstyle='butt', label='SpMM (8 threads)'),
    ]
    hatch_handles = _hatch_legend_handles(hatch_map)
    all_handles = bar_handles + hatch_handles
    all_labels = [h.get_label() for h in all_handles]
    ax.legend(all_handles, all_labels, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, 1.17),
              fontsize=6, framealpha=0.9,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.3,
              borderpad=0.25, labelspacing=0.3)

    # Clip bars exceeding y_limit
    if max_speed > y_limit:
        ax.set_ylim(top=y_limit)

    fig.savefig(output_path, bbox_inches='tight', dpi=300,
                metadata={'Creator': 'SABLE'})
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Step 2: 4-subplot scaling figure (Figure 9, unchanged)
# ---------------------------------------------------------------------------

def select_matrices_for_scaling(all_data, matrix_nnz):
    """Select 4 representative matrices for the scaling figure.

    Hand-picked to span nnz ranges and show diverse parallel behavior:
    - orani678: small (90K nnz), moderate coverage, overhead-dominated SpMV
    - heart2: medium (683K nnz), many blocks, SpMM wins
    - exdata_1: large (2.3M nnz), near-total coverage, dramatic scaling
    - TSOPF_RS_b2383: very large (16M nnz), moderate coverage, near-linear scaling
    """
    selected = ['orani678', 'heart2', 'exdata_1', 'TSOPF_RS_b2383']
    print(f"Selected matrices for scaling plot: {selected}")
    return selected


def _plot_scaling_subplot(ax, all_data, matrix, baselines, thread_keys, thread_labels):
    """Plot thread scaling for a single matrix on one axes.

    Normalizes to the best single-threaded fully-sparse time across all
    baselines in all_data.  Solid lines = SABLE, dashed = fully sparse.
    """
    best_1t_sparse = None
    for entries in all_data.values():
        for entry in entries:
            if entry['matrix_name'] != matrix:
                continue
            timing = entry['timing'].get('1 thread', {})
            fs = timing.get('fully_sparse_time')
            if fs is not None and (best_1t_sparse is None or fs < best_1t_sparse):
                best_1t_sparse = fs

    if best_1t_sparse is None or best_1t_sparse == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    for baseline in baselines:
        entries = all_data.get(baseline, [])
        for entry in entries:
            if entry['matrix_name'] != matrix:
                continue
            speedups = []
            for tc in thread_keys:
                timing = entry['timing'].get(tc, {})
                t = timing.get('total_time_ns')
                speedups.append(best_1t_sparse / t if t and t > 0 else None)
            ax.plot(range(len(thread_labels)), speedups,
                    marker='o', linewidth=2.5, markersize=6,
                    color=COLORS[baseline],
                    label=f'SABLE ({BASELINE_DISPLAY[baseline]})')

    for baseline in baselines:
        entries = all_data.get(baseline, [])
        for entry in entries:
            if entry['matrix_name'] != matrix:
                continue
            speedups = []
            for tc in thread_keys:
                timing = entry['timing'].get(tc, {})
                t = timing.get('fully_sparse_time')
                speedups.append(best_1t_sparse / t if t and t > 0 else None)
            ax.plot(range(len(thread_labels)), speedups,
                    marker='s', linewidth=1.5, markersize=5, linestyle='--',
                    markerfacecolor='none', markeredgewidth=1.5,
                    color=COLORS[baseline],
                    label=f'Fully Sparse ({BASELINE_DISPLAY[baseline]})')

    ax.set_xticks(range(len(thread_labels)))
    ax.set_xticklabels(thread_labels)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)


def plot_combined_scaling_subplots(spmv_data, spmm_data, selected_matrices,
                                   matrix_nnz, output_path):
    """Create a 2x4 subplot figure: SpMV scaling (top) and SpMM scaling (bottom).

    Each column corresponds to the same matrix.  Solid lines show SABLE
    dispatch times, dashed lines show fully-sparse baselines.
    """
    thread_labels = ['1', '2', '4', '8', '12']
    thread_keys = THREAD_COUNTS

    spmv_baselines = ['mkl', 'naive', 'spv8', 'uzp']
    spmm_baselines = ['mkl', 'naive']

    fig, axes = plt.subplots(2, 4, figsize=(20, 7))

    for col, matrix in enumerate(selected_matrices):
        nnz = matrix_nnz.get(matrix, '?')

        # Top row: SpMV
        ax = axes[0][col]
        _plot_scaling_subplot(ax, spmv_data, matrix, spmv_baselines,
                              thread_keys, thread_labels)
        ax.set_title(f'{matrix} (nnz={nnz:,})', fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel('SpMV\nSpeedup', fontsize=10)
        ax.tick_params(labelbottom=False)

        # Bottom row: SpMM
        ax = axes[1][col]
        _plot_scaling_subplot(ax, spmm_data, matrix, spmm_baselines,
                              thread_keys, thread_labels)
        ax.set_xlabel('Threads', fontsize=9)
        if col == 0:
            ax.set_ylabel('SpMM\nSpeedup', fontsize=10)

    # Unified legend: collect from all subplots, preserve desired order
    handle_dict = {}
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handle_dict:
                    handle_dict[l] = h

    desired_order = (
        [f'SABLE ({BASELINE_DISPLAY[b]})' for b in ['mkl', 'naive', 'spv8', 'uzp']]
        + [f'Fully Sparse ({BASELINE_DISPLAY[b]})' for b in ['mkl', 'naive', 'spv8', 'uzp']]
    )
    ordered_labels = [l for l in desired_order if l in handle_dict]
    ordered_handles = [handle_dict[l] for l in ordered_labels]

    fig.legend(ordered_handles, ordered_labels, loc='lower center',
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    fig.savefig(output_path, bbox_inches='tight', dpi=300,
                metadata={'Creator': 'SABLE'})
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel SpMV+SpMM analysis and plotting.")
    parser.add_argument('--results-dir', default='results',
                        help='Directory with sable_sp{mv,mm}_*.json files')
    parser.add_argument('--output-bar', default=None,
                        help='Output PDF for parallel bar chart')
    parser.add_argument('--output-scaling', default=None,
                        help='Output PDF for scaling subplots')
    parser.add_argument('--matrices', nargs=4, default=None,
                        help='4 matrix names for scaling plot (auto-selected if omitted)')
    args = parser.parse_args()

    results_dir = args.results_dir
    output_bar = args.output_bar or os.path.join('images', 'parallel_best_times_comparison.pdf')
    output_scaling = args.output_scaling or os.path.join('images', 'thread_scaling.pdf')

    print("Loading SpMV data...")
    spmv_data = load_all_spmv(results_dir)
    print(f"  Loaded {len(spmv_data)} baselines")

    print("Loading SpMM data...")
    spmm_data = load_all_spmm(results_dir)
    print(f"  Loaded {len(spmm_data)} baselines")

    matrix_nnz = get_matrix_nnz(spmv_data, spmm_data)
    print(f"  {len(matrix_nnz)} matrices total")

    eligible = get_parallel_eligible_matrices(spmv_data, spmm_data)
    print(f"  {len(eligible)} matrices pass parallel filters "
          f"(>={MIN_BLOCK_COVERAGE}% coverage, aspect<={MAX_BLOCK_ASPECT_RATIO})")

    print(f"\nComputing SpMV speedups at {PARALLEL_TC}...")
    spmv_speedups_all = compute_spmv_speedups(spmv_data, PARALLEL_TC)
    spmv_speedups = {m: v for m, v in spmv_speedups_all.items() if m in eligible}
    spmv_geomean = math.exp(sum(math.log(v['speedup']) for v in spmv_speedups.values()) / len(spmv_speedups)) if spmv_speedups else 0
    n_spmv_better = sum(1 for v in spmv_speedups.values() if v['speedup'] > 1.0)
    print(f"  {len(spmv_speedups)} matrices, {n_spmv_better} with speedup>1, geomean={spmv_geomean:.3f}")

    print(f"\nComputing SpMM speedups...")
    spmm_speedups_all = compute_spmm_speedups(spmm_data, PARALLEL_TC)
    spmm_speedups = {m: v for m, v in spmm_speedups_all.items() if m in eligible}
    spmm_geomean = math.exp(sum(math.log(v['speedup']) for v in spmm_speedups.values()) / len(spmm_speedups)) if spmm_speedups else 0
    n_spmm_better = sum(1 for v in spmm_speedups.values() if v['speedup'] > 1.0)
    print(f"  {len(spmm_speedups)} matrices (filtered), {n_spmm_better} with speedup>1, geomean={spmm_geomean:.3f}")

    print(f"\nGenerating parallel bar chart...")
    plot_parallel_bar(spmv_speedups, spmm_speedups, matrix_nnz, output_bar)

    print(f"\nSelecting matrices for scaling plot...")
    if args.matrices:
        selected = args.matrices
    else:
        selected = select_matrices_for_scaling(spmv_data, matrix_nnz)

    print(f"\nGenerating scaling subplots...")
    plot_combined_scaling_subplots(spmv_data, spmm_data, selected, matrix_nnz, output_scaling)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for SpMV at {PARALLEL_TC}:")
    print(f"{'='*60}")
    sorted_m = sorted(spmv_speedups.keys(), key=lambda m: matrix_nnz.get(m, 0))
    for m in sorted_m:
        info = spmv_speedups[m]
        spmm_info = spmm_speedups.get(m)
        spmm_str = f", SpMM={spmm_info['speedup']:.3f}x" if spmm_info else ""
        print(f"  {m:25s}: SpMV={info['speedup']:.3f}x  "
              f"(SABLE/{info['sable_baseline']} vs {info['sparse_baseline']}){spmm_str}")

    print(f"\n  SpMV geometric mean speedup: {spmv_geomean:.3f}x")
    print(f"  SpMV matrices with speedup > 1: {n_spmv_better}/{len(spmv_speedups)}")
    print(f"  SpMM geometric mean speedup: {spmm_geomean:.3f}x")
    print(f"  SpMM matrices with speedup > 1: {n_spmm_better}/{len(spmm_speedups)}")

    print("\nDone!")


if __name__ == '__main__':
    main()