#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>


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

static const long vdia_val_seg_row0[] = {0, 3};
static const long vdia_val_seg_nrows[] = {3, 3};
static const long vdia_val_seg_off0[] = {0, -1};
static const long vdia_val_seg_nd[] = {2, 2};
static const long vdia_val_seg_voff[] = {0, 6};

static void vdia_val_spmv(double *y, const double *x, const double *val) {
    for (long s = 0; s < 2; s++)
        sable_vdia_spmv_segment(val + vdia_val_seg_voff[s], vdia_val_seg_row0[s], vdia_val_seg_nrows[s], vdia_val_seg_off0[s],
                                vdia_val_seg_nd[s], 6, x, y);
}


int main(void) {
    double *y = (double *)calloc(6, sizeof(double));
    double *x = (double *)malloc(6 * sizeof(double));
    assert(y != NULL);
    assert(x != NULL);
    double *vdia_val = (double *)malloc(12 * sizeof(double));
    assert(vdia_val != NULL);
    int *vdia_idiag = (int *)malloc(4 * sizeof(int));
    assert(vdia_idiag != NULL);
    FILE *matrix_file = fopen("<PATH>/fixture.sabledata", "r");
    assert(matrix_file != NULL);
    read_double_array(matrix_file, vdia_val, 12);
    read_int_array(matrix_file, vdia_idiag, 4);
    fclose(matrix_file);
    FILE *rhs_file = fopen("<PATH>/x.vector", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 6);
    fclose(rhs_file);

    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(1, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 6 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
vdia_val_spmv(y, x, vdia_val);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

    printf("Dispatch 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[0][i]);
    }
    printf("\n");
    printf("\n");
    for (int i = 0; i < 6; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vdia_val);
    free(vdia_idiag);
    free(dispatch_part_times);
    free(x);
    free(y);
    return 0;
}
