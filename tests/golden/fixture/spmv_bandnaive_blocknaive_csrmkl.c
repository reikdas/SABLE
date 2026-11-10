#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <mkl.h>
#include <mkl_spblas.h>


static void skip_to_array(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF && c != '[') {}
    if (c == EOF) {
        fprintf(stderr, "Unexpected end of data file\n");
        exit(1);
    }
}

static void finish_array_line(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF && c != '\n') {}
}

static void read_double_array(FILE *file, double *out, int size) {
    skip_to_array(file);
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &out[i]) != 1) {
            fprintf(stderr, "Failed to read double array\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(file);
            if (comma != ',') {
                fprintf(stderr, "Malformed double array\n");
                exit(1);
            }
        }
    }
    finish_array_line(file);
}

static void read_int_array(FILE *file, int *out, int size) {
    skip_to_array(file);
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%d", &out[i]) != 1) {
            fprintf(stderr, "Failed to read int array\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(file);
            if (comma != ',') {
                fprintf(stderr, "Malformed int array\n");
                exit(1);
            }
        }
    }
    finish_array_line(file);
}

static void read_dense_input(FILE *file, double *out, int size) {
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &out[i]) != 1) {
            fprintf(stderr, "Failed to read dense input\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(file);
            if (comma != ',') {
                fprintf(stderr, "Malformed dense input\n");
                exit(1);
            }
        }
    }
}


#ifndef SABLE_VDIA_SPMV
#define SABLE_VDIA_SPMV
static void sable_vdia_spmv_guarded(const double *dv, long r0, long nr, long off0, long nd, long a, long b,
        long m, const double *x, double *y) {
    for (long r = a; r < b; r++) {
        double t = 0.0;
        for (long d = 0; d < nd; d++) {
            long c = r + off0 + d;
            if (c >= 0 && c < m) t += dv[d * nr + (r - r0)] * x[c];
        }
        y[r] += t;
    }
}
static void sable_vdia_spmv_segment(const double *dv, long r0, long nr, long off0, long nd, long m,
        const double *x, double *y) {
    long e = r0 + nr, lo = r0 > -off0 ? r0 : -off0, hi = m - off0 - nd + 1;
    if (hi > e) hi = e;
    if (lo > e) lo = e;
    if (hi < lo) hi = lo;
    sable_vdia_spmv_guarded(dv, r0, nr, off0, nd, r0, lo, m, x, y);
    long r = lo;
    for (; r + 32 <= hi; r += 32) {
        __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd(), a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
        const double *vp = dv + (r - r0), *xp = x + r + off0;
        for (long d = 0; d < nd; d++, vp += nr, xp++) {
            a0 = _mm512_fmadd_pd(_mm512_loadu_pd(vp), _mm512_loadu_pd(xp), a0);
            a1 = _mm512_fmadd_pd(_mm512_loadu_pd(vp + 8), _mm512_loadu_pd(xp + 8), a1);
            a2 = _mm512_fmadd_pd(_mm512_loadu_pd(vp + 16), _mm512_loadu_pd(xp + 16), a2);
            a3 = _mm512_fmadd_pd(_mm512_loadu_pd(vp + 24), _mm512_loadu_pd(xp + 24), a3);
        }
        double *yp = y + r;
        _mm512_storeu_pd(yp, _mm512_add_pd(_mm512_loadu_pd(yp), a0));
        _mm512_storeu_pd(yp + 8, _mm512_add_pd(_mm512_loadu_pd(yp + 8), a1));
        _mm512_storeu_pd(yp + 16, _mm512_add_pd(_mm512_loadu_pd(yp + 16), a2));
        _mm512_storeu_pd(yp + 24, _mm512_add_pd(_mm512_loadu_pd(yp + 24), a3));
    }
    for (; r + 8 <= hi; r += 8) {
        __m512d a0 = _mm512_setzero_pd();
        const double *vp = dv + (r - r0), *xp = x + r + off0;
        for (long d = 0; d < nd; d++, vp += nr, xp++)
            a0 = _mm512_fmadd_pd(_mm512_loadu_pd(vp), _mm512_loadu_pd(xp), a0);
        _mm512_storeu_pd(y + r, _mm512_add_pd(_mm512_loadu_pd(y + r), a0));
    }
    if (r < hi) {
        __mmask8 k = (__mmask8)((1u << (hi - r)) - 1u);
        __m512d a0 = _mm512_setzero_pd();
        const double *vp = dv + (r - r0), *xp = x + r + off0;
        for (long d = 0; d < nd; d++, vp += nr, xp++)
            a0 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(k, vp), _mm512_maskz_loadu_pd(k, xp), a0);
        _mm512_mask_storeu_pd(y + r, k, _mm512_add_pd(_mm512_maskz_loadu_pd(k, y + r), a0));
    }
    sable_vdia_spmv_guarded(dv, r0, nr, off0, nd, hi, e, m, x, y);
}
#endif

static const long vdia_val_seg_row0[] = {0};
static const long vdia_val_seg_nrows[] = {3};
static const long vdia_val_seg_off0[] = {-2};
static const long vdia_val_seg_nd[] = {5};
static const long vdia_val_seg_voff[] = {0};

static void vdia_val_spmv(double *y, const double *x, const double *val) {
    for (long s = 0; s < 1; s++)
        sable_vdia_spmv_segment(val + vdia_val_seg_voff[s], vdia_val_seg_row0[s], vdia_val_seg_nrows[s], vdia_val_seg_off0[s],
                                vdia_val_seg_nd[s], 14, x, y);
}

static void vbr_val_spmv_naive_block(double *y, const double *x, const double *vbr_val, int r0, int r1, int c0, int c1, int offset) {
for (int j = c0; j < c1; j++) {
    for (int i = r0; i < r1; i++) {
        y[i] += vbr_val[offset + (j - c0) * (r1 - r0) + (i - r0)] * x[j];
    }
}
}


int main(void) {
    double *y = (double *)calloc(14, sizeof(double));
    double *x = (double *)malloc(14 * sizeof(double));
    assert(y != NULL);
    assert(x != NULL);
    double *vdia_val = (double *)malloc(15 * sizeof(double));
    assert(vdia_val != NULL);
    int *vdia_idiag = (int *)malloc(5 * sizeof(int));
    assert(vdia_idiag != NULL);
    double *vbr_val = (double *)malloc(73 * sizeof(double));
    assert(vbr_val != NULL);
    int *csr_indptr = (int *)malloc(15 * sizeof(int));
    assert(csr_indptr != NULL);
    int *csr_indices = (int *)malloc(6 * sizeof(int));
    assert(csr_indices != NULL);
    double *csr_val = (double *)malloc(6 * sizeof(double));
    assert(csr_val != NULL);
    FILE *matrix_file = fopen("<PATH>/fixture.sabledata", "r");
    assert(matrix_file != NULL);
    read_double_array(matrix_file, vdia_val, 15);
    read_int_array(matrix_file, vdia_idiag, 5);
    read_double_array(matrix_file, vbr_val, 73);
    read_int_array(matrix_file, csr_indptr, 15);
    read_int_array(matrix_file, csr_indices, 6);
    read_double_array(matrix_file, csr_val, 6);
    fclose(matrix_file);
    FILE *rhs_file = fopen("<PATH>/x.vector", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 14);
    fclose(rhs_file);

sparse_matrix_t csr_handle;
struct matrix_descr csr_descr;
csr_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
csr_descr.mode = SPARSE_FILL_MODE_FULL;
csr_descr.diag = SPARSE_DIAG_NON_UNIT;
mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO,
    14, 14,
    csr_indptr, csr_indptr + 1,
    csr_indices, csr_val);
    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(4, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 14 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
vdia_val_spmv(y, x, vdia_val);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
vbr_val_spmv_naive_block(y, x, vbr_val, 3, 6, 3, 6, 0);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[1][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
vbr_val_spmv_naive_block(y, x, vbr_val, 6, 14, 6, 14, 9);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[2][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_handle, csr_descr, x, 1.0, y);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[3][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

mkl_sparse_destroy(csr_handle);
    printf("Dispatch 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[0][i]);
    }
    printf("\n");
    printf("Dispatch 2: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[1][i] + dispatch_part_times[2][i]);
    }
    printf("\n");
    printf("Dispatch 3: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[3][i]);
    }
    printf("\n");
    printf("Dispatch 2 Part 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[1][i]);
    }
    printf("\n");
    printf("Dispatch 2 Part 2: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[2][i]);
    }
    printf("\n");
    printf("\n");
    for (int i = 0; i < 14; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vdia_val);
    free(vdia_idiag);
    free(vbr_val);
    free(csr_indptr);
    free(csr_indices);
    free(csr_val);
    free(dispatch_part_times);
    free(x);
    free(y);
    return 0;
}
