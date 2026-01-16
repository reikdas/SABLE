import json
import csv
import math
from pathlib import Path


def load_json_data(filepath):
    """Load JSON data from file (read-only)."""
    with open(filepath, 'r') as f:
        return json.load(f)


def process_stats():
    """Process JSON files and generate stats.csv."""
    # Load all JSON files
    mkl_data = load_json_data('results/sable_spmv_mkl.json')
    spv8_data = load_json_data('results/sable_spmv_spv8.json')
    naive_data = load_json_data('results/sable_spmv_naive.json')
    
    # Create dictionaries keyed by matrix_name for easy lookup
    mkl_dict = {entry['matrix_name']: entry for entry in mkl_data}
    spv8_dict = {entry['matrix_name']: entry for entry in spv8_data}
    naive_dict = {entry['matrix_name']: entry for entry in naive_data}
    
    # Get all unique matrix names (should be the same in all files)
    all_matrices = set(mkl_dict.keys()) | set(spv8_dict.keys()) | set(naive_dict.keys())
    
    # Prepare CSV rows and collect values for geometric mean
    csv_rows = []
    mkl_speedups = []
    spv8_speedups = []
    naive_speedups = []
    best_speedups = []
    
    for matrix_name in sorted(all_matrices):
        if matrix_name not in mkl_dict or matrix_name not in spv8_dict or matrix_name not in naive_dict:
            continue
        
        mkl_entry = mkl_dict[matrix_name]
        spv8_entry = spv8_dict[matrix_name]
        naive_entry = naive_dict[matrix_name]
        
        # Extract common fields
        matrix = matrix_name
        nnz = mkl_entry['matrix_dimensions']['nnz']
        dense_nnz_perc = mkl_entry['nnz']['dense_nnz_perc']
        
        # Extract speedup values (round to 2 decimal places)
        mkl_speedup = round(mkl_entry['timing']['speedup'], 2)
        spv8_speedup = round(spv8_entry['timing']['speedup'], 2)
        naive_speedup = round(naive_entry['timing']['speedup'], 2)
        
        # Find best baseline (lowest fully_sparse_time)
        mkl_fully_sparse = mkl_entry['timing']['fully_sparse_time']
        spv8_fully_sparse = spv8_entry['timing']['fully_sparse_time']
        naive_fully_sparse = naive_entry['timing']['fully_sparse_time']
        
        baseline_times = {
            'MKL': mkl_fully_sparse,
            'spv8': spv8_fully_sparse,
            'naive': naive_fully_sparse
        }
        best_baseline = min(baseline_times, key=baseline_times.get)
        x = baseline_times[best_baseline]
        
        # Find best SABLE (lowest total_time_ns)
        mkl_total_time = mkl_entry['timing']['total_time_ns']
        spv8_total_time = spv8_entry['timing']['total_time_ns']
        naive_total_time = naive_entry['timing']['total_time_ns']
        
        sable_times = {
            'MKL': mkl_total_time,
            'spv8': spv8_total_time,
            'naive': naive_total_time
        }
        best_sable = min(sable_times, key=sable_times.get)
        y = sable_times[best_sable]
        
        # Calculate best SABLE speedup over best baseline
        best_speedup = round(x / y, 2)
        
        # Collect values for geometric mean
        mkl_speedups.append(mkl_speedup)
        spv8_speedups.append(spv8_speedup)
        naive_speedups.append(naive_speedup)
        best_speedups.append(best_speedup)
        
        # Add row to CSV
        csv_rows.append({
            'Matrix': matrix,
            'nnz': nnz,
            'dense_nnz_perc': dense_nnz_perc,
            'SABLE(MKL) speedup over MKL': mkl_speedup,
            'SABLE(spv8) speedup over spv8': spv8_speedup,
            'SABLE(naive) speedup over naive': naive_speedup,
            'Best baseline': best_baseline,
            'Best SABLE': best_sable,
            'Best SABLE speedup over best baseline': best_speedup
        })
    
    # Sort rows by nnz
    csv_rows.sort(key=lambda x: x['nnz'])
    
    # Write CSV file
    output_path = Path('results/stats.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'Matrix',
        'nnz',
        'dense_nnz_perc',
        'SABLE(MKL) speedup over MKL',
        'SABLE(spv8) speedup over spv8',
        'SABLE(naive) speedup over naive',
        'Best baseline',
        'Best SABLE',
        'Best SABLE speedup over best baseline'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Calculate geometric means
    def geometric_mean(values):
        """Calculate geometric mean of a list of values."""
        if not values:
            return 0.0
        product = 1.0
        for val in values:
            product *= val
        return product ** (1.0 / len(values))
    
    geomean_mkl = geometric_mean(mkl_speedups)
    geomean_spv8 = geometric_mean(spv8_speedups)
    geomean_best = geometric_mean(best_speedups)
    
    geomean_naive = geometric_mean(naive_speedups)
    
    print(f"Generated {output_path} with {len(csv_rows)} rows")
    print(f"Geometric mean of SABLE(MKL) speedup over MKL: {geomean_mkl:.4f}")
    print(f"Geometric mean of SABLE(spv8) speedup over spv8: {geomean_spv8:.4f}")
    print(f"Geometric mean of SABLE(naive) speedup over naive: {geomean_naive:.4f}")
    print(f"Geometric mean of Best SABLE speedup over best baseline: {geomean_best:.4f}")


if __name__ == '__main__':
    process_stats()
