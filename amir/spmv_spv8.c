
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define SIMD_WIDTH 8
#define VARIANCE_THRESHOLD 2.0

typedef struct {
    int row_id;
    int row_length;
    int row_start;
} RowInfo;

// TRANSPOSED panel data for efficient cross-row access
typedef struct {
    int row_ids[SIMD_WIDTH];      // Original row IDs for output
    double *vals;                  // Column-major transposed values
    int *cols;                     // Column-major transposed column indices  
    int panel_length;              // Columns in this panel
    int allocated;                 // Memory allocated flag
} Panel;

typedef struct {
    RowInfo *row_info;
    Panel *panels;
    int num_panels;
    int fragment_start_idx;
    int total_rows;
} SpV8Info;

int cmp_rows_desc(const void *a, const void *b) {
    return ((RowInfo*)b)->row_length - ((RowInfo*)a)->row_length;
}

int can_form_panel(const RowInfo *rows, int idx, int total) {
    if (idx + SIMD_WIDTH > total) return 0;
    int max_len = rows[idx].row_length;
    int min_len = rows[idx + SIMD_WIDTH - 1].row_length;
    if (min_len <= 0) return 0;
    return (double)max_len / min_len <= VARIANCE_THRESHOLD;
}

// Preprocess: sort rows, form panels, TRANSPOSE data to column-major
SpV8Info* spv8_preprocess(const int *indptr, const double *val, const int *col, int rows) {
    SpV8Info *info = (SpV8Info*)malloc(sizeof(SpV8Info));
    info->row_info = (RowInfo*)malloc(rows * sizeof(RowInfo));
    info->total_rows = rows;
    
    for (int i = 0; i < rows; i++) {
        info->row_info[i].row_id = i;
        info->row_info[i].row_length = indptr[i+1] - indptr[i];
        info->row_info[i].row_start = indptr[i];
    }
    
    qsort(info->row_info, rows, sizeof(RowInfo), cmp_rows_desc);
    
    // Count panels
    int num_panels = 0;
    int idx = 0;
    while (can_form_panel(info->row_info, idx, rows)) {
        num_panels++;
        idx += SIMD_WIDTH;
    }
    
    info->num_panels = num_panels;
    info->fragment_start_idx = idx;
    
    if (num_panels > 0) {
        info->panels = (Panel*)calloc(num_panels, sizeof(Panel));
        
        idx = 0;
        for (int p = 0; p < num_panels; p++) {
            // Use minimum row length as panel length
            int panel_len = info->row_info[idx + SIMD_WIDTH - 1].row_length;
            info->panels[p].panel_length = panel_len;
            
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                info->panels[p].row_ids[lane] = info->row_info[idx + lane].row_id;
            }
            
            if (panel_len > 0) {
                // IMP: Allocate and transpose to column-major format
                // Layout: vals[col * SIMD_WIDTH + lane] = value at (lane, col)
                info->panels[p].vals = (double*)aligned_alloc(64, panel_len * SIMD_WIDTH * sizeof(double));
                info->panels[p].cols = (int*)aligned_alloc(64, panel_len * SIMD_WIDTH * sizeof(int));
                info->panels[p].allocated = 1;
                
                // Transpose: copy data column by column
                for (int c = 0; c < panel_len; c++) {
                    for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                        int src_idx = info->row_info[idx + lane].row_start + c;
                        int dst_idx = c * SIMD_WIDTH + lane;
                        info->panels[p].vals[dst_idx] = val[src_idx];
                        info->panels[p].cols[dst_idx] = col[src_idx];
                    }
                }
            } else {
                info->panels[p].vals = NULL;
                info->panels[p].cols = NULL;
                info->panels[p].allocated = 0;
            }
            
            idx += SIMD_WIDTH;
        }
    } else {
        info->panels = NULL;
    }
    
    return info;
}

void spv8_free(SpV8Info *info) {
    if (info->panels) {
        for (int p = 0; p < info->num_panels; p++) {
            if (info->panels[p].allocated) {
                free(info->panels[p].vals);
                free(info->panels[p].cols);
            }
        }
        free(info->panels);
    }
    free(info->row_info);
    free(info);
}

// SpV8 kernel with proper SIMD on transposed data
void spmv_spv8(double *restrict y, const double *restrict val,
               const int *restrict col, const int *restrict ptr,
               const double *restrict x, int rows, const SpV8Info *info) {
    
    // Phase 1: Process panels (cross-row SIMD on transposed data)
    for (int p = 0; p < info->num_panels; p++) {
        const Panel *panel = &info->panels[p];
        int panel_len = panel->panel_length;
        
        if (panel_len == 0) {
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                y[panel->row_ids[lane]] = 0.0;
            }
            continue;
        }
        
        __m512d sums = _mm512_setzero_pd();
        
        // Cross-row processing on TRANSPOSED column-major data
        for (int c = 0; c < panel_len; c++) {
            int base = c * SIMD_WIDTH;
            
            // Load 8 values (contiguous after transpose)
            __m512d v = _mm512_load_pd(&panel->vals[base]);
            
            // Load 8 column indices (contiguous after transpose)
            __m256i idx = _mm256_load_si256((__m256i*)&panel->cols[base]);
            
            // Gather x values
            __m512d xv = _mm512_i32gather_pd(idx, x, 8);
            
            // FMA
            sums = _mm512_fmadd_pd(v, xv, sums);
        }
        
        // Store results
        double result[SIMD_WIDTH] __attribute__((aligned(64)));
        _mm512_store_pd(result, sums);
        
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {
            y[panel->row_ids[lane]] = result[lane];
        }
        
        // Handle extra elements beyond panel_length
        int row_idx = p * SIMD_WIDTH;
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {
            const RowInfo *ri = &info->row_info[row_idx + lane];
            int extra_start = ri->row_start + panel_len;
            int extra_end = ri->row_start + ri->row_length;
            
            double extra = 0.0;
            for (int j = extra_start; j < extra_end; j++) {
                extra += val[j] * x[col[j]];
            }
            y[ri->row_id] += extra;
        }
    }
    
    // Phase 2: Process fragments with in-row vectorization
    for (int i = info->fragment_start_idx; i < rows; i++) {
        const RowInfo *ri = &info->row_info[i];
        int start = ri->row_start;
        int len = ri->row_length;
        int end = start + len;
        
        if (len == 0) {
            y[ri->row_id] = 0.0;
            continue;
        }
        
        double sum = 0.0;
        int j = start;
        
        int vec_end = start + (len / SIMD_WIDTH) * SIMD_WIDTH;
        if (j < vec_end) {
            __m512d sv = _mm512_setzero_pd();
            for (; j < vec_end; j += SIMD_WIDTH) {
                __m512d v = _mm512_loadu_pd(&val[j]);
                __m256i idx = _mm256_loadu_si256((__m256i*)&col[j]);
                __m512d xv = _mm512_i32gather_pd(idx, x, 8);
                sv = _mm512_fmadd_pd(v, xv, sv);
            }
            sum = _mm512_reduce_add_pd(sv);
        }
        
        for (; j < end; j++) sum += val[j] * x[col[j]];
        y[ri->row_id] = sum;
    }
}

int main() {
    double *y = (double*)calloc(5000, sizeof(double));
    double *x = (double*)malloc(5000 * sizeof(double));
    double *csr_val = (double*)malloc(6941750 * sizeof(double));
    int *indices = (int*)malloc(6941750 * sizeof(int));
    int *indptr = (int*)malloc((5000+1) * sizeof(int));
    struct timespec t1, t2;
    double times[30];
    
    
    FILE *file1 = fopen("csr_data/banded_30.csr", "r");
    if (!file1) { fprintf(stderr, "CSR file error\n"); return 1; }
    
    char c;
    int val_size = 0;
    
    assert(fscanf(file1, "indptr=[%c", &c) == 1);
    if (c != ']') {
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indptr[val_size++]) == 1);
        while (fscanf(file1, "%c", &c) == 1 && c == ',')
            assert(fscanf(file1, "%d", &indptr[val_size++]) == 1);
    }
    assert(fscanf(file1, "%c", &c) == 1);
    
    val_size = 0;
    assert(fscanf(file1, "indices=[%c", &c) == 1);
    if (c != ']') {
        ungetc(c, file1);
        assert(fscanf(file1, "%d", &indices[val_size++]) == 1);
        while (fscanf(file1, "%c", &c) == 1 && c == ',')
            assert(fscanf(file1, "%d", &indices[val_size++]) == 1);
    }
    assert(fscanf(file1, "%c", &c) == 1);
    
    val_size = 0;
    assert(fscanf(file1, "data=[%c", &c) == 1);
    if (c != ']') {
        ungetc(c, file1);
        assert(fscanf(file1, "%lf", &csr_val[val_size++]) == 1);
        while (fscanf(file1, "%c", &c) == 1 && c == ',')
            assert(fscanf(file1, "%lf", &csr_val[val_size++]) == 1);
    }
    fclose(file1);
    
    FILE *file2 = fopen("vec_data/vector.txt", "r");
    if (!file2) { fprintf(stderr, "Vector file error\n"); return 1; }
    for (int i = 0; i < 5000; i++) assert(fscanf(file2, "%lf,", &x[i]) == 1);
    fclose(file2);

    
    // Preprocessing (not timed - amortized over many iterations)
    SpV8Info *spv8 = spv8_preprocess(indptr, csr_val, indices, 5000);
    
    // Debug output
    fprintf(stderr, "SpV8: %d panels formed, fragment start at %d\n", 
            spv8->num_panels, spv8->fragment_start_idx);
    
    spmv_spv8(y, csr_val, indices, indptr, x, 5000, spv8); // warmup
    
    for (int i = 0; i < 30; i++) {
        memset(y, 0, 5000 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        spmv_spv8(y, csr_val, indices, indptr, x, 5000, spv8);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }
    
    for (int i = 0; i < 29; i++)
        for (int j = i+1; j < 30; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    
    printf("%.2f\n", times[15]);
    
    spv8_free(spv8);
    free(y); free(x); free(csr_val); free(indptr); free(indices);
    return 0;
}
