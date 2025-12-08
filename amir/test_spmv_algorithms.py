"""
SpMV Algorithm Comparison - PROPERLY FIXED VERSION

 SpV8 panel data transposition
The SpV8 paper explicitly states: "Step-3: Transpose all panels into column major format"
Without this, cross-row access has terrible cache locality!

This version:
1. Transposes panel data to column-major during preprocessing
2. Cross-row kernel now accesses contiguous memory
3. Proper AVX-512 vectorization
"""

import csv
import os
import subprocess
import random
import math
from pathlib import Path

# Configuration  
CFLAGS = ["-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx512f", "-ffast-math"]
MATRIX_SIZE = 5000
BENCH_FREQ = 30
SIMD_WIDTH = 8

CSR_DIR = "csr_data"
VECTOR_DIR = "vec_data"  
RESULTS_DIR = "results_proper"


def generate_sparse_matrix(size, pattern, density_pct):
    """Generate sparse matrix with specified pattern."""
    data = []
    indices = []
    indptr = [0]
    
    random.seed(42)
    
    if pattern == "random":
        avg_nnz = int(size * density_pct / 100.0)
        for row in range(size):
            if avg_nnz > 0:
                num_elements = max(1, int(random.gauss(avg_nnz, avg_nnz * 0.1)))
                num_elements = min(num_elements, size)
                cols = sorted(random.sample(range(size), num_elements))
                for col in cols:
                    data.append(random.uniform(0.5, 1.5))
                    indices.append(col)
            indptr.append(len(data))
    
    elif pattern == "powerlaw":
        # Power-law: most rows short, few very long
        alpha = 2.0
        total_target = int(size * size * density_pct / 100.0)
        
        row_lengths = []
        for i in range(size):
            u = random.uniform(0.01, 1.0)
            k = int(u ** (-1.0 / (alpha - 1)))
            row_lengths.append(max(1, min(k, size)))
        
        # Scale to target
        scale = total_target / max(1, sum(row_lengths))
        row_lengths = [max(0, min(int(k * scale), size)) for k in row_lengths]
        
        for row in range(size):
            if row_lengths[row] > 0:
                cols = sorted(random.sample(range(size), row_lengths[row]))
                for col in cols:
                    data.append(random.uniform(0.5, 1.5))
                    indices.append(col)
            indptr.append(len(data))
    
    elif pattern == "banded":
        bandwidth = max(1, int(size * density_pct / 100.0))
        half = bandwidth // 2
        for row in range(size):
            for col in range(max(0, row - half), min(size, row + half + 1)):
                data.append(random.uniform(0.5, 1.5))
                indices.append(col)
            indptr.append(len(data))
    
    return data, indices, indptr


def save_csr_file(filename, data, indices, indptr):
    os.makedirs(CSR_DIR, exist_ok=True)
    filepath = os.path.join(CSR_DIR, filename)
    with open(filepath, 'w') as f:
        f.write("indptr=[" + ",".join(map(str, indptr)) + "]\n")
        f.write("indices=[" + ",".join(map(str, indices)) + "]\n")
        f.write("data=[" + ",".join(f"{v:.6f}" for v in data) + "]\n")
    return filepath


def generate_vector(size):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    filename = os.path.join(VECTOR_DIR, "vector.txt")
    random.seed(123)
    with open(filename, 'w') as f:
        f.write(','.join(f"{random.uniform(0.5, 1.5):.6f}" for _ in range(size)) + ',')
    return filename


def get_file_parsing_code(csr_file, vector_file, rows, cols, nnz):
    return f"""
    FILE *file1 = fopen("{csr_file}", "r");
    if (!file1) {{ fprintf(stderr, "CSR file error\\n"); return 1; }}
    
    char c;
    int val_size = 0;
    
    assert(fscanf(file1, "indptr=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indptr[val_size++]) == 1);
        while (fscanf(file1, "%c", &c) == 1 && c == ',')
            assert(fscanf(file1, "%d", &indptr[val_size++]) == 1);
    }}
    assert(fscanf(file1, "%c", &c) == 1);
    
    val_size = 0;
    assert(fscanf(file1, "indices=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indices[val_size++]) == 1);
        while (fscanf(file1, "%c", &c) == 1 && c == ',')
            assert(fscanf(file1, "%d", &indices[val_size++]) == 1);
    }}
    assert(fscanf(file1, "%c", &c) == 1);
    
    val_size = 0;
    assert(fscanf(file1, "data=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%lf", &csr_val[val_size++]) == 1);
        while (fscanf(file1, "%c", &c) == 1 && c == ',')
            assert(fscanf(file1, "%lf", &csr_val[val_size++]) == 1);
    }}
    fclose(file1);
    
    FILE *file2 = fopen("{vector_file}", "r");
    if (!file2) {{ fprintf(stderr, "Vector file error\\n"); return 1; }}
    for (int i = 0; i < {cols}; i++) assert(fscanf(file2, "%lf,", &x[i]) == 1);
    fclose(file2);
"""


def generate_csr_baseline(csr_file, vector_file, rows, cols, nnz):
    """Standard CSR SpMV - well-optimized baseline."""
    return f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define SIMD_WIDTH 8

// Baseline CSR with manual vectorization for fair comparison
void spmv_csr(double *restrict y, const double *restrict val, 
              const int *restrict col, const int *restrict ptr, 
              const double *restrict x, int rows) {{
    for (int i = 0; i < rows; i++) {{
        int start = ptr[i];
        int end = ptr[i+1];
        int len = end - start;
        
        double sum = 0.0;
        int j = start;
        
        // Vectorized with gather
        int vec_end = start + (len / SIMD_WIDTH) * SIMD_WIDTH;
        if (j < vec_end) {{
            __m512d sum_vec = _mm512_setzero_pd();
            for (; j < vec_end; j += SIMD_WIDTH) {{
                __m512d v = _mm512_loadu_pd(&val[j]);
                __m256i idx = _mm256_loadu_si256((__m256i*)&col[j]);
                __m512d xv = _mm512_i32gather_pd(idx, x, 8);
                sum_vec = _mm512_fmadd_pd(v, xv, sum_vec);
            }}
            sum = _mm512_reduce_add_pd(sum_vec);
        }}
        
        // Scalar tail
        for (; j < end; j++) sum += val[j] * x[col[j]];
        y[i] = sum;
    }}
}}

int main() {{
    double *y = (double*)calloc({rows}, sizeof(double));
    double *x = (double*)malloc({cols} * sizeof(double));
    double *csr_val = (double*)malloc({max(1,nnz)} * sizeof(double));
    int *indices = (int*)malloc({max(1,nnz)} * sizeof(int));
    int *indptr = (int*)malloc(({rows}+1) * sizeof(int));
    struct timespec t1, t2;
    double times[{BENCH_FREQ}];
    
    {get_file_parsing_code(csr_file, vector_file, rows, cols, nnz)}
    
    spmv_csr(y, csr_val, indices, indptr, x, {rows}); // warmup
    
    for (int i = 0; i < {BENCH_FREQ}; i++) {{
        memset(y, 0, {rows} * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_csr(y, csr_val, indices, indptr, x, {rows});
        clock_gettime(CLOCK_MONOTONIC, &t2);
        times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }}
    
    for (int i = 0; i < {BENCH_FREQ-1}; i++)
        for (int j = i+1; j < {BENCH_FREQ}; j++)
            if (times[j] < times[i]) {{ double t = times[i]; times[i] = times[j]; times[j] = t; }}
    
    printf("%.2f\\n", times[{BENCH_FREQ//2}]);
    free(y); free(x); free(csr_val); free(indptr); free(indices);
    return 0;
}}
"""


def generate_spv8_proper(csr_file, vector_file, rows, cols, nnz):
    """
    PROPERLY FIXED SpV8 with column-major transposed panel data.
    
    The key insight from the paper:
    "Step-3: Transpose all panels into column major format"
    
    This ensures cross-row access is SEQUENTIAL in memory
    """
    return f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define SIMD_WIDTH 8
#define VARIANCE_THRESHOLD 2.0

typedef struct {{
    int row_id;
    int row_length;
    int row_start;
}} RowInfo;

// TRANSPOSED panel data for efficient cross-row access
typedef struct {{
    int row_ids[SIMD_WIDTH];      // Original row IDs for output
    double *vals;                  // Column-major transposed values
    int *cols;                     // Column-major transposed column indices  
    int panel_length;              // Columns in this panel
    int allocated;                 // Memory allocated flag
}} Panel;

typedef struct {{
    RowInfo *row_info;
    Panel *panels;
    int num_panels;
    int fragment_start_idx;
    int total_rows;
}} SpV8Info;

int cmp_rows_desc(const void *a, const void *b) {{
    return ((RowInfo*)b)->row_length - ((RowInfo*)a)->row_length;
}}

int can_form_panel(const RowInfo *rows, int idx, int total) {{
    if (idx + SIMD_WIDTH > total) return 0;
    int max_len = rows[idx].row_length;
    int min_len = rows[idx + SIMD_WIDTH - 1].row_length;
    if (min_len <= 0) return 0;
    return (double)max_len / min_len <= VARIANCE_THRESHOLD;
}}

// Preprocess: sort rows, form panels, TRANSPOSE data to column-major
SpV8Info* spv8_preprocess(const int *indptr, const double *val, const int *col, int rows) {{
    SpV8Info *info = (SpV8Info*)malloc(sizeof(SpV8Info));
    info->row_info = (RowInfo*)malloc(rows * sizeof(RowInfo));
    info->total_rows = rows;
    
    for (int i = 0; i < rows; i++) {{
        info->row_info[i].row_id = i;
        info->row_info[i].row_length = indptr[i+1] - indptr[i];
        info->row_info[i].row_start = indptr[i];
    }}
    
    qsort(info->row_info, rows, sizeof(RowInfo), cmp_rows_desc);
    
    // Count panels
    int num_panels = 0;
    int idx = 0;
    while (can_form_panel(info->row_info, idx, rows)) {{
        num_panels++;
        idx += SIMD_WIDTH;
    }}
    
    info->num_panels = num_panels;
    info->fragment_start_idx = idx;
    
    if (num_panels > 0) {{
        info->panels = (Panel*)calloc(num_panels, sizeof(Panel));
        
        idx = 0;
        for (int p = 0; p < num_panels; p++) {{
            // Use minimum row length as panel length
            int panel_len = info->row_info[idx + SIMD_WIDTH - 1].row_length;
            info->panels[p].panel_length = panel_len;
            
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
                info->panels[p].row_ids[lane] = info->row_info[idx + lane].row_id;
            }}
            
            if (panel_len > 0) {{
                // Allocate and transpose to column-major format
                // Layout: vals[col * SIMD_WIDTH + lane] = value at (lane, col)
                info->panels[p].vals = (double*)aligned_alloc(64, panel_len * SIMD_WIDTH * sizeof(double));
                info->panels[p].cols = (int*)aligned_alloc(64, panel_len * SIMD_WIDTH * sizeof(int));
                info->panels[p].allocated = 1;
                
                // Transpose: copy data column by column
                for (int c = 0; c < panel_len; c++) {{
                    for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
                        int src_idx = info->row_info[idx + lane].row_start + c;
                        int dst_idx = c * SIMD_WIDTH + lane;
                        info->panels[p].vals[dst_idx] = val[src_idx];
                        info->panels[p].cols[dst_idx] = col[src_idx];
                    }}
                }}
            }} else {{
                info->panels[p].vals = NULL;
                info->panels[p].cols = NULL;
                info->panels[p].allocated = 0;
            }}
            
            idx += SIMD_WIDTH;
        }}
    }} else {{
        info->panels = NULL;
    }}
    
    return info;
}}

void spv8_free(SpV8Info *info) {{
    if (info->panels) {{
        for (int p = 0; p < info->num_panels; p++) {{
            if (info->panels[p].allocated) {{
                free(info->panels[p].vals);
                free(info->panels[p].cols);
            }}
        }}
        free(info->panels);
    }}
    free(info->row_info);
    free(info);
}}

// SpV8 kernel with proper SIMD on transposed data
void spmv_spv8(double *restrict y, const double *restrict val,
               const int *restrict col, const int *restrict ptr,
               const double *restrict x, int rows, const SpV8Info *info) {{
    
    // Phase 1: Process panels (cross-row SIMD on transposed data)
    for (int p = 0; p < info->num_panels; p++) {{
        const Panel *panel = &info->panels[p];
        int panel_len = panel->panel_length;
        
        if (panel_len == 0) {{
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
                y[panel->row_ids[lane]] = 0.0;
            }}
            continue;
        }}
        
        __m512d sums = _mm512_setzero_pd();
        
        // Cross-row processing on TRANSPOSED column-major data
        for (int c = 0; c < panel_len; c++) {{
            int base = c * SIMD_WIDTH;
            
            // Load 8 values 
            __m512d v = _mm512_load_pd(&panel->vals[base]);
            
            // Load 8 column indices 
            __m256i idx = _mm256_load_si256((__m256i*)&panel->cols[base]);
            
            // Gather x values
            __m512d xv = _mm512_i32gather_pd(idx, x, 8);
            
            // FMA
            sums = _mm512_fmadd_pd(v, xv, sums);
        }}
        
        // Store results
        double result[SIMD_WIDTH] __attribute__((aligned(64)));
        _mm512_store_pd(result, sums);
        
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
            y[panel->row_ids[lane]] = result[lane];
        }}
        
        // Handle extra elements beyond panel_length
        int row_idx = p * SIMD_WIDTH;
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
            const RowInfo *ri = &info->row_info[row_idx + lane];
            int extra_start = ri->row_start + panel_len;
            int extra_end = ri->row_start + ri->row_length;
            
            double extra = 0.0;
            for (int j = extra_start; j < extra_end; j++) {{
                extra += val[j] * x[col[j]];
            }}
            y[ri->row_id] += extra;
        }}
    }}
    
    // Phase 2: Process fragments with in-row vectorization
    for (int i = info->fragment_start_idx; i < rows; i++) {{
        const RowInfo *ri = &info->row_info[i];
        int start = ri->row_start;
        int len = ri->row_length;
        int end = start + len;
        
        if (len == 0) {{
            y[ri->row_id] = 0.0;
            continue;
        }}
        
        double sum = 0.0;
        int j = start;
        
        int vec_end = start + (len / SIMD_WIDTH) * SIMD_WIDTH;
        if (j < vec_end) {{
            __m512d sv = _mm512_setzero_pd();
            for (; j < vec_end; j += SIMD_WIDTH) {{
                __m512d v = _mm512_loadu_pd(&val[j]);
                __m256i idx = _mm256_loadu_si256((__m256i*)&col[j]);
                __m512d xv = _mm512_i32gather_pd(idx, x, 8);
                sv = _mm512_fmadd_pd(v, xv, sv);
            }}
            sum = _mm512_reduce_add_pd(sv);
        }}
        
        for (; j < end; j++) sum += val[j] * x[col[j]];
        y[ri->row_id] = sum;
    }}
}}

int main() {{
    double *y = (double*)calloc({rows}, sizeof(double));
    double *x = (double*)malloc({cols} * sizeof(double));
    double *csr_val = (double*)malloc({max(1,nnz)} * sizeof(double));
    int *indices = (int*)malloc({max(1,nnz)} * sizeof(int));
    int *indptr = (int*)malloc(({rows}+1) * sizeof(int));
    struct timespec t1, t2;
    double times[{BENCH_FREQ}];
    
    {get_file_parsing_code(csr_file, vector_file, rows, cols, nnz)}
    
    // Preprocessing (not timed - amortized over many iterations)
    SpV8Info *spv8 = spv8_preprocess(indptr, csr_val, indices, {rows});
    
    // Debug output
    fprintf(stderr, "SpV8: %d panels formed, fragment start at %d\\n", 
            spv8->num_panels, spv8->fragment_start_idx);
    
    spmv_spv8(y, csr_val, indices, indptr, x, {rows}, spv8); // warmup
    
    for (int i = 0; i < {BENCH_FREQ}; i++) {{
        memset(y, 0, {rows} * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_spv8(y, csr_val, indices, indptr, x, {rows}, spv8);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }}
    
    for (int i = 0; i < {BENCH_FREQ-1}; i++)
        for (int j = i+1; j < {BENCH_FREQ}; j++)
            if (times[j] < times[i]) {{ double t = times[i]; times[i] = times[j]; times[j] = t; }}
    
    printf("%.2f\\n", times[{BENCH_FREQ//2}]);
    
    spv8_free(spv8);
    free(y); free(x); free(csr_val); free(indptr); free(indices);
    return 0;
}}
"""


def compile_and_run(c_code, name):
    c_file = f"spmv_{name}.c"
    exe = f"spmv_{name}_exe"
    
    try:
        with open(c_file, 'w') as f:
            f.write(c_code)
        
        result = subprocess.run(
            ["gcc"] + CFLAGS + ["-o", exe, c_file, "-lm"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"Compile error ({name}): {result.stderr[:300]}")
            return None
        
        result = subprocess.run([f"./{exe}"], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Runtime error ({name}): {result.stderr[:300]}")
            return None
        
        return float(result.stdout.strip().split('\n')[0])
    except Exception as e:
        print(f"Error ({name}): {e}")
        return None


def run_tests(pattern, densities):
    print(f"\n{'='*70}")
    print(f"Pattern: {pattern.upper()}")
    print(f"{'='*70}\n")
    
    results = []
    vector_file = generate_vector(MATRIX_SIZE)
    
    for density in densities:
        print(f"  {density}%...", end=" ", flush=True)
        
        data, indices, indptr = generate_sparse_matrix(MATRIX_SIZE, pattern, density)
        nnz = len(data)
        
        if nnz == 0:
            print("SKIP")
            continue
        
        csr_file = save_csr_file(f"{pattern}_{density}.csr", data, indices, indptr)
        
        # Test both algorithms
        c_csr = generate_csr_baseline(csr_file, vector_file, MATRIX_SIZE, MATRIX_SIZE, nnz)
        time_csr = compile_and_run(c_csr, "csr")
        
        c_spv8 = generate_spv8_proper(csr_file, vector_file, MATRIX_SIZE, MATRIX_SIZE, nnz)
        time_spv8 = compile_and_run(c_spv8, "spv8")
        
        if time_csr and time_spv8:
            speedup = time_csr / time_spv8
            winner = "SpV8" if speedup > 1.0 else "CSR"
            
            print(f"NNZ:{nnz:,} | CSR:{time_csr/1000:.1f}μs, SpV8:{time_spv8/1000:.1f}μs "
                  f"({speedup:.2f}x) → {winner}")
            
            results.append({
                'density': density,
                'nnz': nnz,
                'time_csr_us': time_csr/1000,
                'time_spv8_us': time_spv8/1000,
                'speedup': speedup,
                'winner': winner
            })
        else:
            print("FAILED")
    
    return results


def save_results(pattern, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = os.path.join(RESULTS_DIR, f"{pattern}.csv")
    
    if results:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ Saved to {filename}")


def main():
    print("\n" + "="*70)
    print("SpMV: CSR vs SpV8 (PROPERLY FIXED)")
    print(f"Matrix: {MATRIX_SIZE}×{MATRIX_SIZE}, SIMD: {SIMD_WIDTH}")
    print("="*70)
    print("\nKey fix: SpV8 panel data transposed to column-major format")
    print("This makes cross-row access SEQUENTIAL in memory!\n")
    
    tests = [
        ("powerlaw", [1, 2, 5, 10, 15, 20]),
        ("random", [5, 10, 20, 30]),
        ("banded", [5, 10, 20, 30]),
    ]
    
    all_results = {}
    for pattern, densities in tests:
        results = run_tests(pattern, densities)
        save_results(pattern, results)
        all_results[pattern] = results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for pattern, results in all_results.items():
        if not results:
            continue
        spv8_wins = sum(1 for r in results if r['winner'] == 'SpV8')
        csr_wins = len(results) - spv8_wins
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"{pattern}: CSR={csr_wins} wins, SpV8={spv8_wins} wins, Avg speedup={avg_speedup:.2f}x")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()