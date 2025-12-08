def generate_spv8_algorithm(csr_file, vector_file, rows, cols, nnz):
    
    c_code = f"""
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define SIMD_WIDTH 8

typedef struct {{
    int row_id;
    int row_length;
    int row_start;
}} RowInfo;

int compare_rows(const void *a, const void *b) {{
    return ((RowInfo*)b)->row_length - ((RowInfo*)a)->row_length;
}}

// sort rows by length, identify panels
typedef struct {{
    RowInfo *row_info;
    int num_panels;
    int panel_start_idx;  // Where panels end and fragments begin
}} SpV8Info;

SpV8Info* spv8_preprocess(const int *indptr, int rows) {{
    SpV8Info *info = (SpV8Info*)malloc(sizeof(SpV8Info));
    info->row_info = (RowInfo*)malloc(rows * sizeof(RowInfo));
    
    // Build row info
    for (int i = 0; i < rows; i++) {{
        info->row_info[i].row_id = i;
        info->row_info[i].row_length = indptr[i+1] - indptr[i];
        info->row_info[i].row_start = indptr[i];
    }}
    
    // Sort by length descending
    qsort(info->row_info, rows, sizeof(RowInfo), compare_rows);
    
    // Count how many complete panels we can form
    info->num_panels = 0;
    int idx = 0;
    while (idx + SIMD_WIDTH <= rows) {{
        int panel_length = info->row_info[idx].row_length;
        int can_form = 1;
        for (int i = 1; i < SIMD_WIDTH; i++) {{
            if (info->row_info[idx + i].row_length != panel_length) {{
                can_form = 0;
                break;
            }}
        }}
        if (can_form && panel_length > 0) {{
            info->num_panels++;
            idx += SIMD_WIDTH;
        }} else {{
            break;
        }}
    }}
    info->panel_start_idx = idx;
    
    return info;
}}

void spmv_spv8(double *restrict y, const double *restrict csr_val,
               const int *restrict indices, const int *restrict indptr,
               const double *restrict x, const int rows,
               const SpV8Info *info) {{
    
    const RowInfo *row_info = info->row_info;
    
    // process groups of equal length
    int row_idx = 0;
    for (int p = 0; p < info->num_panels; p++) {{
        int panel_length = row_info[row_idx].row_length;
        double sums[SIMD_WIDTH] = {{0}};
        
        // Cross-row parallel: process SIMD_WIDTH rows together
        for (int col_idx = 0; col_idx < panel_length; col_idx++) {{
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
                int data_idx = row_info[row_idx + lane].row_start + col_idx;
                sums[lane] += csr_val[data_idx] * x[indices[data_idx]];
            }}
        }}
        
        // Write results
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {{
            y[row_info[row_idx + lane].row_id] = sums[lane];
        }}
        
        row_idx += SIMD_WIDTH;
    }}
    
    // Process fragments with in-row parallel
    for (int i = row_idx; i < rows; i++) {{
        int row_id = row_info[i].row_id;
        int row_start = row_info[i].row_start;
        int row_len = row_info[i].row_length;
        int row_end = row_start + row_len;
        
        double sum = 0.0;
        
        // vectorized part - 8 elements at a time with gather
        int j = row_start;
        int vec_end = row_start + (row_len / SIMD_WIDTH) * SIMD_WIDTH;
        
        __m512d sum_vec = _mm512_setzero_pd();
        for (; j < vec_end; j += SIMD_WIDTH) {{
            __m512d val_vec = _mm512_loadu_pd(&csr_val[j]);
            // Load 8 x 32-bit indices into 256-bit vector for gather
            __m256i idx_vec = _mm256_loadu_si256((__m256i*)&indices[j]);
            __m512d x_vec = _mm512_i32gather_pd(idx_vec, x, 8);
            sum_vec = _mm512_fmadd_pd(val_vec, x_vec, sum_vec);
        }}
        sum = _mm512_reduce_add_pd(sum_vec);
        
        // Scalar tail
        for (; j < row_end; j++) {{
            sum += csr_val[j] * x[indices[j]];
        }}
        
        y[row_id] = sum;
    }}
}}

int main() {{
    double *y = (double*)calloc({rows}, sizeof(double));
    double *x = (double*)malloc({cols} * sizeof(double));
    double *csr_val = (double*)malloc({nnz} * sizeof(double));
    int *indices = (int*)malloc({nnz} * sizeof(int));
    int *indptr = (int*)malloc(({rows} + 1) * sizeof(int));
    struct timespec t1, t2;
    double times[{BENCH_FREQ}];
    
    // Load file
    FILE *file1 = fopen("{csr_file}", "r");
    if (!file1) {{
        fprintf(stderr, "Error opening CSR file\\n");
        return 1;
    }}
    
    char c;
    int val_size = 0;
    
    // Parse indptr
    assert(fscanf(file1, "indptr=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indptr[val_size]) == 1);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1);
                val_size++;
            }} else if (c == ']') break;
        }}
    }}
    assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');
    
    // Parse indices
    val_size = 0;
    assert(fscanf(file1, "indices=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indices[val_size]) == 1);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1);
                val_size++;
            }} else if (c == ']') break;
        }}
    }}
    assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');
    
    // Parse data
    val_size = 0;
    assert(fscanf(file1, "data=[%c", &c) == 1);
    if (c != ']') {{
        ungetc(c, file1);
        assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1);
                val_size++;
            }} else if (c == ']') break;
        }}
    }}
    fclose(file1);
    
    // load vec
    FILE *file2 = fopen("{vector_file}", "r");
    if (!file2) {{
        fprintf(stderr, "Error opening vector file\\n");
        return 1;
    }}
    for (int i = 0; i < {cols}; i++) {{
        assert(fscanf(file2, "%lf,", &x[i]) == 1);
    }}
    fclose(file2);
    
    // preprocess, done once, not timed
    SpV8Info *spv8_info = spv8_preprocess(indptr, {rows});
    
    // bench
    for (int i = 0; i < {BENCH_FREQ}; i++) {{
        memset(y, 0, {rows} * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_spv8(y, csr_val, indices, indptr, x, {rows}, spv8_info);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }}
    
    // Sort times and output median
    for (int i = 0; i < {BENCH_FREQ - 1}; i++) {{
        for (int j = i + 1; j < {BENCH_FREQ}; j++) {{
            if (times[j] < times[i]) {{
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }}
        }}
    }}
    
    printf("%.2f\\n", times[{BENCH_FREQ // 2}]);
    free(spv8_info->row_info);
    free(spv8_info);
    free(y); free(x); free(csr_val); free(indptr); free(indices);
    return 0;
}}
"""
    return c_code
