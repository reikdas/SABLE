import json
import glob
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def parse_filename(filename):
    """Extract (dense_dispatch, sparse_dispatch, reordering) from result filename.

    Expected nomenclature: sable_spmv_{dense dispatch}_{sparse dispatch}.json
    Some files additionally encode a reordering algorithm suffix, e.g.
      - sable_spmv_blas_mkl_rcm.json
      - sable_spmv_naive_spv8_spectral_tuned.json

    Only sparse dispatches are: mkl, naive, spv8, uzp
    Any remaining suffix after those two fields is treated as a reordering tag.
    """
    basename = os.path.basename(filename)
    if not basename.startswith('sable_spmv_') or not basename.endswith('.json'):
        return None, None, None

    # Remove prefix and suffix
    middle = basename[len('sable_spmv_'):-len('.json')]
    parts = middle.split('_')

    # We require at least dense + sparse dispatch.
    if len(parts) < 2:
        return None, None, None

    dense = parts[0]
    sparse = parts[1]

    # Filter out filenames that don't match dispatch nomenclature.
    allowed_sparse_dispatches = {'mkl', 'naive', 'spv8', 'uzp'}
    if sparse not in allowed_sparse_dispatches:
        return None, None, None

    reordering = '_'.join(parts[2:]) if len(parts) > 2 else 'none'
    return dense, sparse, reordering


def _get_timing(entry):
    """Return the timing dict for an entry. SpMV JSON may nest it under e.g. '1 thread'."""
    timing = entry.get('timing', {})
    if not timing:
        return {}
    if timing.get('total_time_ns') is not None:
        return timing
    for v in timing.values():
        if isinstance(v, dict) and v.get('total_time_ns') is not None:
            return v
    return timing


def _entry_sort_key(entry):
    """Sort key for choosing the best entry per matrix (lower is better)."""
    timing = _get_timing(entry)
    total_time = timing.get('total_time_ns')
    speedup = timing.get('speedup')

    # Prefer defined, positive total_time_ns.
    if isinstance(total_time, (int, float)) and total_time > 0:
        return (0, total_time, 0)

    # Fall back to higher speedup (convert to a 'lower is better' key).
    if isinstance(speedup, (int, float)):
        return (1, float('inf'), -speedup)

    # Worst case: keep but deprioritize.
    return (2, float('inf'), 0)


def load_all_data(results_dir='results'):
    """Load spmv result files and organize by (dense_dispatch, sparse_dispatch).

    If multiple reorderings exist for the same (dense, sparse) and matrix, we keep
    the best entry per matrix (by total_time_ns, then speedup).
    """
    pattern = os.path.join(results_dir, 'sable_spmv_*.json')
    json_files = glob.glob(pattern)

    # {(dense, sparse): {matrix_name: entry_dict}}
    all_data = defaultdict(dict)
    matrix_nnz = {}

    for json_file in json_files:
        dense, sparse, reordering = parse_filename(json_file)
        if dense is None or sparse is None:
            continue
        if dense == 'naive':
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        for entry in data:
            matrix_name = entry.get('matrix_name')
            if not matrix_name:
                continue

            key = (dense, sparse)
            existing = all_data[key].get(matrix_name)
            if existing is None or _entry_sort_key(entry) < _entry_sort_key(existing):
                # Keep the best entry among reorderings.
                all_data[key][matrix_name] = entry

            # Store nnz value
            if matrix_name not in matrix_nnz:
                nnz = entry.get('matrix_dimensions', {}).get('nnz')
                if nnz is not None:
                    matrix_nnz[matrix_name] = nnz

    return all_data, matrix_nnz


def calculate_max_theoretical_speedup(entry):
    """Calculate max_theoretical_speedup_possible = fully_sparse_time_ns / sparse_time_ns"""
    timing = _get_timing(entry)
    # Note: field is 'fully_sparse_time' not 'fully_sparse_time_ns' in the JSON
    fully_sparse_time = timing.get('fully_sparse_time')
    sparse_time_ns = timing.get('sparse_time_ns')

    if fully_sparse_time is not None and sparse_time_ns is not None and sparse_time_ns > 0:
        return fully_sparse_time / sparse_time_ns
    return None


def plot_speedup_vs_theoretical(all_data, matrix_nnz, output_prefix):
    """
    For each (dense, sparse) combination, create a separate PDF plot showing:
    - speedup (what we get) as bars
    - max_theoretical_speedup_possible (what we aim for) as a line on top
    """
    combinations = sorted(all_data.keys())

    # Matrices to exclude (disrupt the graph with wildly different speedups)
    exclude_matrices = {'exdata_1'}

    for dense, sparse in combinations:
        matrices_data = all_data[(dense, sparse)]
        if not matrices_data:
            continue

        # Sort matrices by nnz, excluding problematic ones
        sorted_matrices = sorted(
            [m for m in matrices_data.keys() if m not in exclude_matrices],
            key=lambda m: matrix_nnz.get(m, float('inf'))
        )

        speedups = []
        max_theoretical = []
        matrix_labels = []

        for matrix_name in sorted_matrices:
            entry = matrices_data[matrix_name]
            timing = _get_timing(entry)

            speedup = timing.get('speedup')
            max_theo = calculate_max_theoretical_speedup(entry)

            if speedup is not None and max_theo is not None:
                speedups.append(speedup)
                max_theoretical.append(max_theo)
                matrix_labels.append(matrix_name)

        if not speedups:
            continue

        # Create the plot
        fig, ax = plt.subplots(figsize=(max(14, len(matrix_labels) * 0.4), 8))

        x = np.arange(len(matrix_labels))
        width = 0.6

        # Plot bars for actual speedup
        ax.bar(x, speedups, width,
               label='Actual Speedup',
               color='steelblue', edgecolor='navy')

        # Plot line for max theoretical speedup (what we aim for)
        ax.plot(x, max_theoretical, 'o-', linewidth=2, markersize=6,
               label='Max Theoretical Speedup (Target)', color='darkred')

        # Add a horizontal line at speedup = 1
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                  label='Speedup = 1 (Breakeven)')

        ax.set_xlabel('Matrix (sorted by nnz)', fontsize=10)
        ax.set_ylabel('Speedup', fontsize=10)
        ax.set_title(f'SpMV Speedups: dense={dense}, sparse={sparse}\n'
                    f'(Line: Max Theoretical Target, Bars: Actual Achieved)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_labels, rotation=45, ha='right', fontsize=7)
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        out_path = f"{output_prefix}_{dense}_{sparse}.pdf"
        fig.savefig(out_path, bbox_inches='tight', dpi=300,
                    metadata={'Creator': 'SABLE-spv8'})
        plt.close(fig)

        print(f"Saved: {out_path} ({len(matrix_labels)} matrices)")


def plot_best_times_comparison(all_data, matrix_nnz, output_prefix):
    """
    Create a single speedup plot showing for each matrix:
    - Speedup = best_fully_sparse_time / best_total_time
    - Labels show: (dense/sparse) vs (sparse) dispatch strategies
    """
    # Matrices to exclude (disrupt the graph with wildly different speedups)
    exclude_matrices = {'exdata_1'}

    # Collect all matrices
    all_matrices = set()
    for combination_data in all_data.values():
        all_matrices.update(m for m in combination_data.keys() if m not in exclude_matrices)

    # Sort matrices by nnz
    sorted_matrices = sorted(all_matrices,
                            key=lambda m: matrix_nnz.get(m, float('inf')))

    # For each matrix, find best total_time_ns and best fully_sparse_time_ns
    best_total_times = {}  # {matrix: (time_ns, dense, sparse)}
    best_fully_sparse_times = {}  # {matrix: (time_ns, sparse)}

    for (dense, sparse), matrices_data in all_data.items():
        for matrix_name, entry in matrices_data.items():
            timing = _get_timing(entry)

            total_time = timing.get('total_time_ns')
            fully_sparse_time = timing.get('fully_sparse_time')

            if total_time is not None:
                if matrix_name not in best_total_times or total_time < best_total_times[matrix_name][0]:
                    best_total_times[matrix_name] = (total_time, dense, sparse)

            if fully_sparse_time is not None:
                if matrix_name not in best_fully_sparse_times or fully_sparse_time < best_fully_sparse_times[matrix_name][0]:
                    best_fully_sparse_times[matrix_name] = (fully_sparse_time, sparse)

    # Only include matrices that have both values
    valid_matrices = [m for m in sorted_matrices
                     if m in best_total_times and m in best_fully_sparse_times]

    if not valid_matrices:
        print("No valid matrices found for best times comparison plot")
        return

    # Debug: print matrices that are missing data
    missing_total = [m for m in sorted_matrices if m not in best_total_times]
    missing_sparse = [m for m in sorted_matrices if m not in best_fully_sparse_times]
    if missing_total:
        print(f"Warning: {len(missing_total)} matrices missing total_time_ns: {missing_total[:5]}...")
    if missing_sparse:
        print(f"Warning: {len(missing_sparse)} matrices missing fully_sparse_time: {missing_sparse[:5]}...")

    # Prepare data for plotting
    speedups = []
    strategy_labels = []  # Format: "(dense/sparse) vs (sparse)"
    matrix_labels = []

    for matrix_name in valid_matrices:
        total_time, dense, sparse = best_total_times[matrix_name]
        fully_sparse_time, sparse_method = best_fully_sparse_times[matrix_name]

        # Calculate speedup
        if total_time > 0 and fully_sparse_time > 0:
            speedup = fully_sparse_time / total_time
        else:
            print(f"Warning: Invalid times for {matrix_name}: total={total_time}, sparse={fully_sparse_time}")
            continue  # Skip matrices with invalid data instead of using 0
        
        speedups.append(speedup)
        
        # Create label showing both strategies
        strategy_labels.append(f'{dense}/{sparse} vs {sparse_method}')
        matrix_labels.append(matrix_name)
    
    if not speedups:
        print("No valid speedups to plot")
        return
    
    print(f"Plotting {len(matrix_labels)} matrices with speedups ranging from {min(speedups):.3f} to {max(speedups):.3f}")

    # Create PDF with single speedup plot
    combined_pdf_path = f"{output_prefix}.pdf"
    
    with PdfPages(combined_pdf_path) as pdf:
        # Single speedup plot with strategy labels
        fig, ax = plt.subplots(figsize=(max(14, len(matrix_labels) * 0.3), 8))

        x = np.arange(len(matrix_labels))
        width = 0.6

        colors = ['green' if s >= 1 else 'red' for s in speedups]
        bars = ax.bar(x, speedups, width=width, color=colors, edgecolor='black', alpha=0.7)
        
        # Add actual speedup values as text on bars for very small bars (might be hard to see)
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            if speedup < 0.05:  # Very small bars - show value to make them visible
                height = max(bar.get_height(), 0.01)  # Ensure some height for text placement
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{speedup:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

        # Add horizontal line at speedup = 1 (prominent)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2.5, label='Speedup = 1 (Breakeven)')

        ax.set_xlabel('Matrix (sorted by nnz)', fontsize=14)
        ax.set_ylabel('Speedup (Best Fully Sparse / Best SABLE)', fontsize=14)
        ax.set_title('SpMV Speedup: Best SABLE Configuration vs Best Fully Sparse\n'
                    '(Green = SABLE faster, Red = Fully Sparse faster)', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_labels, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=12)
        ax.set_ylim(bottom=0)

        # Add strategy labels on top of bars
        for i, (speedup, strategy_label) in enumerate(zip(speedups, strategy_labels)):
            y_pos = speedup + max(speedups) * 0.02 if speedup >= 0 else speedup - max(speedups) * 0.05
            ax.text(i, y_pos, strategy_label,
                   ha='center', va='bottom', fontsize=7, rotation=90)

        plt.tight_layout()
        # Save to PDF
        pdf.savefig(fig, bbox_inches='tight', dpi=300,
                   metadata={'Creator': 'SABLE-spv8'})
        plt.close(fig)
    
    print(f"Saved combined PDF: {combined_pdf_path}")


def main():
    # Nomenclature: results/sable_spmv_{dense dispatch}_{sparse dispatch}.json
    results_dir = 'results'

    print("Loading SpMV result data...")
    all_data, matrix_nnz = load_all_data(results_dir)

    combinations = list(all_data.keys())
    print(f"Found {len(combinations)} (dense, sparse) combinations: {combinations}")

    total_matrices = set()
    for combo_data in all_data.values():
        total_matrices.update(combo_data.keys())
    print(f"Total unique matrices: {len(total_matrices)}")

    # Plot 1: Speedup vs Max Theoretical — one PDF per (dense, sparse) combination
    speedup_prefix = os.path.join(results_dir, 'spmv_speedup_vs_theoretical')
    print(f"\nGenerating speedup vs theoretical plots (one PDF per combination)...")
    plot_speedup_vs_theoretical(all_data, matrix_nnz, speedup_prefix)

    # Plot 2: Best times comparison (as high-res PNGs)
    best_times_prefix = os.path.join(results_dir, 'spmv_best_times_comparison')
    print(f"\nGenerating best times comparison plots...")
    plot_best_times_comparison(all_data, matrix_nnz, best_times_prefix)

    print("\nDone!")


if __name__ == '__main__':
    main()
