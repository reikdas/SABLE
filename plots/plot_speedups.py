#!/usr/bin/env python3
"""Speedup bar charts for every SABLE format composition.

Every chart in this file is the same picture: one matrix per x position, a
SpMV and an SpMM bar side by side, hatched by which backend pair won, with a
break-even line at 1.0.  Only the data source, the physical size, and the
y-axis differ, so they all go through `plot_grouped_bars` below.

  --figure vbr       Fig 5   best_times_comparison.pdf
                     VBR+CSR vs the best fully-sparse baseline
  --figure vdia      Fig 6   vdia_csr_best_times.pdf
                     VDIA+CSR vs CSR
  --figure vdia-vbr  Fig 7   vdia_vbr_csr_best_times.pdf
                     VDIA+VBR+CSR vs VBR+CSR
  --figure order     Fig 12  vbr_vdia_csr_best_times.pdf
                     VBR+VDIA+CSR vs VDIA+VBR+CSR (extraction order, appendix)
  --figure all       all four (default)
"""

import json
import glob
import os
import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

from fukaya_results import EXCLUDED_MATRICES, load_fukaya


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Style:
    hatch_pool: tuple
    figsize: Callable[[int], tuple]
    hatch_linewidth: Optional[float]
    bar_linewidth: Optional[float]
    axhline_linewidth: float
    spine_linewidth: Optional[float]
    axis_label_fontsize: float
    ylabel_fontsize: float
    xtick_fontsize: float
    tick_params: tuple
    clip_fontsize: float
    clip_rotation: float
    clip_pad_frac: float
    clip_stagger: float
    legend_line_linewidth: float
    legend_ncol: Optional[int]
    legend_kwargs: dict = field(default_factory=dict)
    subplots_adjust_top: Optional[float] = None


WIDE = Style(
    hatch_pool=('///', '|||', '+++', '---', 'xxx', 'ooo', '...', '**',
                '//', '++', '--', 'xx'),
    figsize=lambda n: (max(14, n * 0.45), 5),
    hatch_linewidth=None,
    bar_linewidth=None,
    axhline_linewidth=2,
    spine_linewidth=None,
    axis_label_fontsize=12,
    ylabel_fontsize=12,
    xtick_fontsize=9,
    tick_params=(),
    clip_fontsize=9,
    clip_rotation=0,
    clip_pad_frac=0.0,
    clip_stagger=0.04,
    legend_line_linewidth=8,
    legend_ncol=None,          # computed: min(2 + n_hatches, 4)
    legend_kwargs=dict(loc='upper center', fontsize=10, framealpha=0.9,
                       bbox_to_anchor=(0.5, 1.30)),
    subplots_adjust_top=0.80,
)

COLUMN = Style(
    hatch_pool=('//', '||', '++', '--', 'xx', 'oo', '..', '**',
                '/', '+', '-', 'x'),
    figsize=lambda n: (3.4, 1.9),
    hatch_linewidth=0.5,
    bar_linewidth=0.4,
    axhline_linewidth=0.8,
    spine_linewidth=0.5,
    axis_label_fontsize=7,
    ylabel_fontsize=6,
    xtick_fontsize=5,
    tick_params=(dict(axis='y', labelsize=6, width=0.5, length=2),
                 dict(axis='x', width=0.4, length=1.5)),
    clip_fontsize=4.5,
    clip_rotation=90,
    clip_pad_frac=0.03,
    clip_stagger=0.0,
    legend_line_linewidth=5,
    legend_ncol=3,
    legend_kwargs=dict(loc='lower center', fontsize=6, framealpha=0.9,
                       bbox_to_anchor=(0.5, 1.17),
                       columnspacing=0.8, handletextpad=0.4, handlelength=1.3,
                       borderpad=0.25, labelspacing=0.3),
)


# ---------------------------------------------------------------------------
# Hatching helpers
# ---------------------------------------------------------------------------

def _assign_hatches(pool, *strat_lists):
    """Build a mapping from strategy string -> hatch pattern."""
    all_strats = sorted(set(s for sl in strat_lists for s in sl if s))
    return {s: pool[i % len(pool)] for i, s in enumerate(all_strats)}


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
# The shared renderer
# ---------------------------------------------------------------------------

def plot_grouped_bars(spmv, spmm, matrix_nnz, output_path, style,
                      ylabel, y_limit=1.5):
    """Render one speedup chart.

    spmv, spmm  {matrix: (speedup, strategy)} -- either may omit a matrix.
    matrix_nnz  {matrix: nnz}, for the x ordering.
    """
    all_matrices = set(spmv) | set(spmm)
    # Break nnz ties by name: all_matrices is a set, so without a total order
    # matrices with equal nnz (e.g. heart2/heart3) swap places between runs.
    sorted_matrices = sorted(all_matrices,
                             key=lambda m: (matrix_nnz.get(m, float('inf')), m))

    matrix_labels = []
    spmv_vals, spmm_vals = [], []
    spmv_strats, spmm_strats = [], []
    for m in sorted_matrices:
        sv, sm = spmv.get(m), spmm.get(m)
        if sv is None and sm is None:
            continue
        matrix_labels.append(m)
        spmv_vals.append(sv[0] if sv else 0)
        spmm_vals.append(sm[0] if sm else 0)
        spmv_strats.append(sv[1] if sv else '')
        spmm_strats.append(sm[1] if sm else '')

    if not matrix_labels:
        print("No valid data")
        return

    all_vals = [v for v in spmv_vals + spmm_vals if v > 0]
    min_speed, max_speed = min(all_vals), max(all_vals)
    y_bottom = math.floor(min_speed * 10) / 10.0

    print(f"Plotting {len(matrix_labels)} matrices, "
          f"speedups [{min_speed:.3f}, {max_speed:.3f}]")

    if style.hatch_linewidth is not None:
        plt.rcParams['hatch.linewidth'] = style.hatch_linewidth
    fig, ax = plt.subplots(figsize=style.figsize(len(matrix_labels)))

    x = np.arange(len(matrix_labels))
    width = 0.35
    bar_kw = {} if style.bar_linewidth is None else {'linewidth': style.bar_linewidth}
    spmv_bars = ax.bar(x - width / 2, spmv_vals, width, color='steelblue',
                       edgecolor='navy', alpha=0.85, **bar_kw)
    spmm_bars = ax.bar(x + width / 2, spmm_vals, width, color='coral',
                       edgecolor='darkred', alpha=0.85, **bar_kw)

    # Assign hatching per strategy pair
    hatch_map = _assign_hatches(style.hatch_pool, spmv_strats, spmm_strats)
    _apply_hatches(spmv_bars, spmv_strats, hatch_map)
    _apply_hatches(spmm_bars, spmm_strats, hatch_map)

    # Clipped bars are labelled with their actual value.  The wide chart
    # staggers a colliding SpMM label upwards; the column chart instead rotates
    # both and lifts them clear of the top spine.
    pad = (y_limit - y_bottom) * style.clip_pad_frac
    for i in range(len(matrix_labels)):
        sv_clipped = spmv_vals[i] > y_limit
        sm_clipped = spmm_vals[i] > y_limit
        stagger = style.clip_stagger if (sv_clipped and sm_clipped) else 0
        for bars, vals, clipped, extra in (
                (spmv_bars, spmv_vals, sv_clipped, 0),
                (spmm_bars, spmm_vals, sm_clipped, stagger)):
            if not clipped:
                continue
            bar = bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2., y_limit + pad + extra,
                    f'{vals[i]:.1f}×', ha='center', va='bottom',
                    fontsize=style.clip_fontsize, fontweight='bold',
                    rotation=style.clip_rotation)

    # Breakeven line
    ax.axhline(y=1.0, color='black', linestyle='--',
               linewidth=style.axhline_linewidth)

    margin = (y_limit - y_bottom) * 0.05
    ax.set_ylim(bottom=y_bottom, top=y_limit + margin)

    yticks = list(ax.get_yticks())
    if y_bottom not in yticks:
        yticks.append(y_bottom)
        yticks.sort()
    ax.set_yticks(yticks)

    ax.set_xlabel('Matrix (sorted by nnz)', fontsize=style.axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=style.ylabel_fontsize)
    for tp in style.tick_params:
        ax.tick_params(**tp)
    if style.spine_linewidth is not None:
        for s in ax.spines.values():
            s.set_linewidth(style.spine_linewidth)

    ax.set_xticks(x)
    ax.set_xticklabels(matrix_labels, rotation=45, ha='right',
                       fontsize=style.xtick_fontsize)

    # Combined legend: operation colour + strategy hatching.
    bar_handles = [
        Line2D([0], [0], color='steelblue', linewidth=style.legend_line_linewidth,
               solid_capstyle='butt', label='SpMV'),
        Line2D([0], [0], color='coral', linewidth=style.legend_line_linewidth,
               solid_capstyle='butt', label='SpMM'),
    ]
    hatch_handles = _hatch_legend_handles(hatch_map)
    all_handles = bar_handles + hatch_handles
    ncol = (style.legend_ncol if style.legend_ncol is not None
            else min(2 + len(hatch_handles), 4))
    ax.legend(all_handles, [h.get_label() for h in all_handles],
              ncol=ncol, **style.legend_kwargs)
    if style.subplots_adjust_top is not None:
        fig.subplots_adjust(top=style.subplots_adjust_top)

    # Clip bars exceeding y_limit
    if max_speed > y_limit:
        ax.set_ylim(top=y_limit)

    fig.savefig(output_path, bbox_inches='tight', dpi=300,
                metadata={'Creator': 'SABLE'})
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Loader A -- sable_sp{mv,mm}_*.json (Fig 5, VBR+CSR)
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


def _get_timing(entry):
    """Return the timing dict; single-threaded runs may nest under '1 thread'."""
    timing = entry.get('timing', {})
    if '1 thread' in timing:
        return timing['1 thread']
    if timing.get('total_time_ns') is not None:
        return timing
    for v in timing.values():
        if isinstance(v, dict) and v.get('total_time_ns') is not None:
            return v
    return timing


def _entry_sort_key(entry):
    timing = _get_timing(entry)
    total_time = timing.get('total_time_ns')
    speedup = timing.get('speedup')
    if isinstance(total_time, (int, float)) and total_time > 0:
        return (0, total_time, 0)
    if isinstance(speedup, (int, float)):
        return (1, float('inf'), -speedup)
    return (2, float('inf'), 0)


EVAL_SET_MARKER = 'blas'


def _load_sable_data(results_dir, op, parse_filename, dedupe):
    """Load the evaluation set into {(dense, sparse): {matrix: entry}}."""
    all_data = defaultdict(dict)
    matrix_nnz = {}
    for json_file in glob.glob(os.path.join(results_dir, f'sable_{op}_*.json')):
        parsed = parse_filename(json_file)
        dense, sparse = parsed[0], parsed[1]
        if dense != EVAL_SET_MARKER:
            continue
        with open(json_file) as f:
            data = json.load(f)
        for entry in data:
            matrix = entry.get('matrix_name')
            if not matrix:
                continue
            key = (dense, sparse)
            existing = all_data[key].get(matrix)
            # SpMV ships several runs per matrix and keeps the fastest; SpMM
            # has one run per file, so it simply takes the last.
            if (not dedupe or existing is None
                    or _entry_sort_key(entry) < _entry_sort_key(existing)):
                all_data[key][matrix] = entry
            if matrix not in matrix_nnz:
                nnz = entry.get('matrix_dimensions', {}).get('nnz')
                if nnz is not None:
                    matrix_nnz[matrix] = nnz
    return all_data, matrix_nnz


def compute_best_speedups(all_data):
    """For each matrix, best SABLE time vs best fully-sparse time.

    Returns {matrix: (speedup, "sparse vs baseline")}.
    """
    best_total = {}   # {matrix: (time, dense, sparse)}
    best_sparse = {}  # {matrix: (time, sparse)}
    for (dense, sparse), matrices in all_data.items():
        for matrix, entry in matrices.items():
            timing = _get_timing(entry)
            total_time = timing.get('total_time_ns')
            fully_sparse = timing.get('fully_sparse_time')
            if total_time is not None and total_time > 0:
                if matrix not in best_total or total_time < best_total[matrix][0]:
                    best_total[matrix] = (total_time, dense, sparse)
            if fully_sparse is not None and fully_sparse > 0:
                if matrix not in best_sparse or fully_sparse < best_sparse[matrix][0]:
                    best_sparse[matrix] = (fully_sparse, sparse)

    out = {}
    for matrix, (total, _dense, sparse) in best_total.items():
        if matrix not in best_sparse:
            continue
        baseline_time, baseline = best_sparse[matrix]
        if total > 0 and baseline_time > 0:
            out[matrix] = (baseline_time / total, f'{sparse} vs {baseline}')
    return out


# ---------------------------------------------------------------------------
# Loader B -- *_d075.json (Figs 6/7/12, VDIA family)
# ---------------------------------------------------------------------------

SPMV_BACKENDS = ['naive', 'mkl', 'spv8']
SPMM_BACKENDS = ['naive', 'mkl']


def _load_vdia_json(path):
    with open(path) as f:
        return {m['matrix_name']: m for m in json.load(f)
                if m['matrix_name'] not in EXCLUDED_MATRICES}


def _best_from_per_backend(per_backend, backends):
    """Best measured SABLE and baseline time over `backends`, with their names.
    """
    best_v = best_v_b = best_b = best_b_b = None
    for b in backends:
        info = per_backend.get(b)
        if not info:
            continue
        if info['vdia_us'] > 0 and (best_v is None or info['vdia_us'] < best_v):
            best_v, best_v_b = info['vdia_us'], b
        if info['baseline_us'] > 0 and (best_b is None or info['baseline_us'] < best_b):
            best_b, best_b_b = info['baseline_us'], b
    if best_v is None or best_b is None:
        return None
    return (best_b / best_v, f'{best_v_b} vs {best_b_b}')


def load_vdia_speedups(results_dir, filename, backends):
    """{matrix: (speedup, strategy)} plus {matrix: nnz}, from one summary file."""
    raw = _load_vdia_json(os.path.join(results_dir, filename))
    speedups = {}
    for name, m in raw.items():
        computed = _best_from_per_backend(m['per_backend'], backends)
        if computed:
            speedups[name] = computed
    return speedups, {n: m.get('nnz') for n, m in raw.items() if m.get('nnz')}


def _best_sable_of(entry, backends):
    """(time, backend) of the fastest measured SABLE backend for one matrix."""
    cands = [(i['vdia_us'], b) for b, i in entry['per_backend'].items()
             if b in backends and i['vdia_us'] > 0]
    return min(cands) if cands else (None, None)


def load_order_comparison(results_dir, vbr_first_file, vdia_first_file, backends):
    """Speedup of VBR+VDIA+CSR over VDIA+VBR+CSR.

    Restricted to the matrices on which *both* orders yield a genuine
    three-format composition: where one order drops every block (or every
    band) the plan degenerates to a two-format one and there is nothing
    like-for-like to compare against.
    """
    vbr_first = _load_vdia_json(os.path.join(results_dir, vbr_first_file))
    vdia_first = _load_vdia_json(os.path.join(results_dir, vdia_first_file))
    speedups, nnz = {}, {}
    for name in set(vbr_first) & set(vdia_first):
        new_us, new_b = _best_sable_of(vbr_first[name], backends)
        ref_us, ref_b = _best_sable_of(vdia_first[name], backends)
        if not new_us or not ref_us:
            continue
        speedups[name] = (ref_us / new_us, f'{new_b} vs {ref_b}')
        if vbr_first[name].get('nnz'):
            nnz[name] = vbr_first[name]['nnz']
    return speedups, nnz


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(speedups, label):
    vals = [s for s, _ in speedups.values()]
    if not vals:
        return
    wins = sum(1 for s in vals if s > 1.0)
    geo = math.exp(sum(math.log(s) for s in vals) / len(vals))
    print(f"{label}: geomean = {geo:.4f}x over {len(vals)} matrices "
          f"({wins} speedups, {len(vals) - wins} slowdowns)")


def print_order_stats(speedups, label, tol=1e-9):
    """Like print_stats, but reports exact ties separately: where neither
    order drops a region the two decompositions are identical, so the ratio
    is 1 by construction and counting it as a loss would mislead."""
    vals = [s for s, _ in speedups.values()]
    if not vals:
        return
    ties = sum(1 for s in vals if abs(s - 1.0) <= tol)
    wins = sum(1 for s in vals if s > 1.0 + tol)
    losses = sum(1 for s in vals if s < 1.0 - tol)
    geo = math.exp(sum(math.log(s) for s in vals) / len(vals))
    print(f"{label}: geomean = {geo:.4f}x over {len(vals)} matrices "
          f"({wins} faster, {losses} slower, {ties} identical)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_vbr(results_dir, output_dir):
    """Fig 5: VBR+CSR vs the best fully-sparse baseline."""
    print("=== VBR+CSR vs best fully-sparse baseline ===")
    spmv_all, spmv_nnz = _load_sable_data(results_dir, 'spmv',
                                          _parse_spmv_filename, dedupe=True)
    spmv = compute_best_speedups(spmv_all)
    print(f"  SpMV: {len(spmv)} matrices")

    spmm_all, spmm_nnz = _load_sable_data(results_dir, 'spmm',
                                          _parse_spmm_filename, dedupe=False)
    # Restrict SpMM to the SpMV evaluation set.
    spmm = {m: v for m, v in compute_best_speedups(spmm_all).items() if m in spmv}
    print(f"  SpMM: {len(spmm)} matrices (filtered to SpMV set)")

    plot_grouped_bars(spmv, spmm, {**spmv_nnz, **spmm_nnz},
                      os.path.join(output_dir, 'best_times_comparison.pdf'),
                      WIDE, ylabel='Speedup (Best Fully Sparse / Best SABLE)')
    print_stats(spmv, 'SpMV: VBR+CSR vs best fully-sparse')
    print_stats(spmm, 'SpMM: VBR+CSR vs best fully-sparse')


def figure_vdia(results_dir, output_dir):
    """Fig 6: VDIA+CSR vs CSR."""
    print("=== VDIA+CSR vs CSR ===")
    spmv, spmv_nnz = load_vdia_speedups(results_dir, 'spmv_vdia_csr_d075.json',
                                        SPMV_BACKENDS)
    spmm, spmm_nnz = load_vdia_speedups(results_dir, 'spmm_vdia_csr_d075.json',
                                        SPMM_BACKENDS)
    # The four Fukaya et al. matrices were measured in a separate session and
    # arrive with their own best_vs_best already computed.
    fukaya_nnz = {}
    for target, op in ((spmv, 'spmv'), (spmm, 'spmm')):
        for name, entry in load_fukaya(results_dir, op).items():
            bvb = entry.get('best_vs_best')
            if bvb:
                target[name] = (bvb['speedup'],
                                f"{bvb['best_vdia_backend']} vs "
                                f"{bvb['best_baseline_backend']}")
            if entry.get('nnz'):
                fukaya_nnz.setdefault(name, entry['nnz'])

    plot_grouped_bars(spmv, spmm, {**fukaya_nnz, **spmm_nnz, **spmv_nnz},
                      os.path.join(output_dir, 'vdia_csr_best_times.pdf'),
                      COLUMN, ylabel='Speedup (Best Baseline / Best SABLE)')
    print_stats(spmv, 'SpMV: VDIA+CSR vs CSR')
    print_stats(spmm, 'SpMM: VDIA+CSR vs CSR')


def figure_vdia_vbr(results_dir, output_dir):
    """Fig 7: VDIA+VBR+CSR vs VBR+CSR."""
    print("=== VDIA+VBR+CSR vs VBR+CSR ===")
    spmv, spmv_nnz = load_vdia_speedups(results_dir,
                                        'spmv_vdia_vbr_csr_d075.json',
                                        SPMV_BACKENDS)
    spmm, spmm_nnz = load_vdia_speedups(results_dir,
                                        'spmm_vdia_vbr_csr_d075.json',
                                        SPMM_BACKENDS)
    plot_grouped_bars(spmv, spmm, {**spmm_nnz, **spmv_nnz},
                      os.path.join(output_dir, 'vdia_vbr_csr_best_times.pdf'),
                      COLUMN, ylabel='Speedup (Best Baseline / Best SABLE)')
    print_stats(spmv, 'SpMV: VDIA+VBR+CSR vs VBR+CSR')
    print_stats(spmm, 'SpMM: VDIA+VBR+CSR vs VBR+CSR')


def figure_order(results_dir, output_dir):
    """Fig 12 (appendix): VBR+VDIA+CSR vs VDIA+VBR+CSR."""
    print("=== VBR+VDIA+CSR vs VDIA+VBR+CSR (extraction order) ===")
    spmv, spmv_nnz = load_order_comparison(
        results_dir, 'spmv_vbr_vdia_csr_d075.json',
        'spmv_vdia_vbr_csr_d075.json', SPMV_BACKENDS)
    spmm, spmm_nnz = load_order_comparison(
        results_dir, 'spmm_vbr_vdia_csr_d075.json',
        'spmm_vdia_vbr_csr_d075.json', SPMM_BACKENDS)
    plot_grouped_bars(spmv, spmm, {**spmm_nnz, **spmv_nnz},
                      os.path.join(output_dir, 'vbr_vdia_csr_best_times.pdf'),
                      COLUMN, y_limit=1.1,
                      ylabel='Speedup (VDIA+VBR+CSR / VBR+VDIA+CSR)')
    print_order_stats(spmv, 'SpMV: VBR+VDIA+CSR vs VDIA+VBR+CSR')
    print_order_stats(spmm, 'SpMM: VBR+VDIA+CSR vs VDIA+VBR+CSR')


FIGURES = {
    'vbr': figure_vbr,
    'vdia': figure_vdia,
    'vdia-vbr': figure_vdia_vbr,
    'order': figure_order,
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Speedup bar charts for every SABLE format composition.")
    parser.add_argument('--figure', default='all',
                        choices=['all'] + list(FIGURES),
                        help='Which chart to draw (default: all)')
    parser.add_argument('--results-dir',
                        default=os.path.join(here, os.pardir, 'results'),
                        help='Directory with the result JSON files')
    parser.add_argument('--output-dir', default=os.path.join(here, 'images'),
                        help='Output directory for PDFs')
    args = parser.parse_args()

    wanted = list(FIGURES) if args.figure == 'all' else [args.figure]
    for i, name in enumerate(wanted):
        if i:
            print()
        FIGURES[name](args.results_dir, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
