#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <mkl.h>
#include <mkl_cblas.h>


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


#ifndef SABLE_VDIA_SPMM
#define SABLE_VDIA_SPMM
static void sable_vdia_tile_reuse4(const double *dv, const double *dvp, long stride, long rloc,
        long off0, long nd, long r0, long m, const double *x, double *y, long N, long NC) {
    for (long cb = 0; cb < NC; cb += 32) {
        __m512d a00 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 0), a01 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 8), a02 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 16), a03 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 24);
        __m512d a10 = _mm512_loadu_pd(y + (r0 + 1) * N + cb + 0), a11 = _mm512_loadu_pd(y + (r0 + 1) * N + cb + 8), a12 = _mm512_loadu_pd(y + (r0 + 1) * N + cb + 16), a13 = _mm512_loadu_pd(y + (r0 + 1) * N + cb + 24);
        __m512d a20 = _mm512_loadu_pd(y + (r0 + 2) * N + cb + 0), a21 = _mm512_loadu_pd(y + (r0 + 2) * N + cb + 8), a22 = _mm512_loadu_pd(y + (r0 + 2) * N + cb + 16), a23 = _mm512_loadu_pd(y + (r0 + 2) * N + cb + 24);
        __m512d a30 = _mm512_loadu_pd(y + (r0 + 3) * N + cb + 0), a31 = _mm512_loadu_pd(y + (r0 + 3) * N + cb + 8), a32 = _mm512_loadu_pd(y + (r0 + 3) * N + cb + 16), a33 = _mm512_loadu_pd(y + (r0 + 3) * N + cb + 24);
        long jlo = r0 + off0, jhi = r0 + 3 + off0 + nd - 1;
        if (jlo < 0) jlo = 0;
        if (jhi > m - 1) jhi = m - 1;
        for (long j = jlo; j <= jhi; j++) {
            const double *xr = x + j * N + cb;
            __m512d x0 = _mm512_loadu_pd(xr), x1 = _mm512_loadu_pd(xr + 8), x2 = _mm512_loadu_pd(xr + 16), x3 = _mm512_loadu_pd(xr + 24);
            const double *d0 = dvp + (j - r0 - off0 + 3) * stride + rloc;
            __m512d b;
            b = _mm512_set1_pd(d0[0 - 0 * stride]); a00 = _mm512_fmadd_pd(b, x0, a00); a01 = _mm512_fmadd_pd(b, x1, a01); a02 = _mm512_fmadd_pd(b, x2, a02); a03 = _mm512_fmadd_pd(b, x3, a03);
            b = _mm512_set1_pd(d0[1 - 1 * stride]); a10 = _mm512_fmadd_pd(b, x0, a10); a11 = _mm512_fmadd_pd(b, x1, a11); a12 = _mm512_fmadd_pd(b, x2, a12); a13 = _mm512_fmadd_pd(b, x3, a13);
            b = _mm512_set1_pd(d0[2 - 2 * stride]); a20 = _mm512_fmadd_pd(b, x0, a20); a21 = _mm512_fmadd_pd(b, x1, a21); a22 = _mm512_fmadd_pd(b, x2, a22); a23 = _mm512_fmadd_pd(b, x3, a23);
            b = _mm512_set1_pd(d0[3 - 3 * stride]); a30 = _mm512_fmadd_pd(b, x0, a30); a31 = _mm512_fmadd_pd(b, x1, a31); a32 = _mm512_fmadd_pd(b, x2, a32); a33 = _mm512_fmadd_pd(b, x3, a33);
        }
        _mm512_storeu_pd(y + (r0 + 0) * N + cb + 0, a00); _mm512_storeu_pd(y + (r0 + 0) * N + cb + 8, a01); _mm512_storeu_pd(y + (r0 + 0) * N + cb + 16, a02); _mm512_storeu_pd(y + (r0 + 0) * N + cb + 24, a03); _mm512_storeu_pd(y + (r0 + 1) * N + cb + 0, a10); _mm512_storeu_pd(y + (r0 + 1) * N + cb + 8, a11); _mm512_storeu_pd(y + (r0 + 1) * N + cb + 16, a12); _mm512_storeu_pd(y + (r0 + 1) * N + cb + 24, a13); _mm512_storeu_pd(y + (r0 + 2) * N + cb + 0, a20); _mm512_storeu_pd(y + (r0 + 2) * N + cb + 8, a21); _mm512_storeu_pd(y + (r0 + 2) * N + cb + 16, a22); _mm512_storeu_pd(y + (r0 + 2) * N + cb + 24, a23); _mm512_storeu_pd(y + (r0 + 3) * N + cb + 0, a30); _mm512_storeu_pd(y + (r0 + 3) * N + cb + 8, a31); _mm512_storeu_pd(y + (r0 + 3) * N + cb + 16, a32); _mm512_storeu_pd(y + (r0 + 3) * N + cb + 24, a33);
    }
}

static void sable_vdia_tile_direct1(const double *dv, const double *dvp, long stride, long rloc,
        long off0, long nd, long r0, long m, const double *x, double *y, long N, long NC) {
    for (long cb = 0; cb < NC; cb += 32) {
        __m512d a00 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 0), a01 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 8), a02 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 16), a03 = _mm512_loadu_pd(y + (r0 + 0) * N + cb + 24);
        for (long d = 0; d < nd; d++) {
            const double *d0 = dv + d * stride + rloc;
            long j0 = r0 + off0 + d;
            if (j0 + 0 >= 0 && j0 + 0 < m) { const double *xr = x + (j0 + 0) * N + cb; __m512d b = _mm512_set1_pd(d0[0]); a00 = _mm512_fmadd_pd(b, _mm512_loadu_pd(xr + 0), a00); a01 = _mm512_fmadd_pd(b, _mm512_loadu_pd(xr + 8), a01); a02 = _mm512_fmadd_pd(b, _mm512_loadu_pd(xr + 16), a02); a03 = _mm512_fmadd_pd(b, _mm512_loadu_pd(xr + 24), a03); }
        }
        _mm512_storeu_pd(y + (r0 + 0) * N + cb + 0, a00); _mm512_storeu_pd(y + (r0 + 0) * N + cb + 8, a01); _mm512_storeu_pd(y + (r0 + 0) * N + cb + 16, a02); _mm512_storeu_pd(y + (r0 + 0) * N + cb + 24, a03);
    }
}

static void sable_vdia_spmm_segment(const double *dv, const double *dvp, long r0, long nr, long off0, long nd,
        long m, const double *x, double *y, long N) {
    long NC = N - N % 32, r = r0;
    if (NC) {
        for (; r + 4 <= r0 + nr; r += 4)
            sable_vdia_tile_reuse4(dv, dvp, nr, r - r0, off0, nd, r, m, x, y, N, NC);
        for (; r < r0 + nr; r++)
            sable_vdia_tile_direct1(dv, dvp, nr, r - r0, off0, nd, r, m, x, y, N, NC);
    }
    if (NC < N)
        for (long row = 0; row < nr; row++)
            for (long d = 0; d < nd; d++) {
                long col = r0 + row + off0 + d;
                if (col < 0 || col >= m) continue;
                double a = dv[d * nr + row];
                for (long c = NC; c < N; c++) y[(r0 + row) * N + c] += a * x[col * N + c];
            }
}
#endif

static const long vdia_val_seg_row0[] = {0};
static const long vdia_val_seg_nrows[] = {3};
static const long vdia_val_seg_off0[] = {-2};
static const long vdia_val_seg_nd[] = {5};
static const long vdia_val_seg_voff[] = {0};

static void vdia_val_spmm(double *y, const double *x, const double *val, double *const *pad, long N) {
    for (long s = 0; s < 1; s++)
        sable_vdia_spmm_segment(val + vdia_val_seg_voff[s], pad[s], vdia_val_seg_row0[s], vdia_val_seg_nrows[s], vdia_val_seg_off0[s],
                                vdia_val_seg_nd[s], 14, x, y, N);
}

static void vbr_val_spmm_naive_block(double *y, const double *x, const double *vbr_val, int r0, int r1, int c0, int c1, int offset, int nrhs) {
for (int i = r0; i < r1; i++) {
    for (int j = c0; j < c1; j++) {
        double a = vbr_val[offset + (j - c0) * (r1 - r0) + (i - r0)];
        for (int k = 0; k < nrhs; k++) {
            y[i * nrhs + k] += a * x[j * nrhs + k];
        }
    }
}
}


int main(void) {
    double *y = (double *)calloc(7168, sizeof(double));
    double *x = (double *)malloc(7168 * sizeof(double));
    assert(y != NULL);
    assert(x != NULL);
    double *vdia_val = (double *)malloc(15 * sizeof(double));
    assert(vdia_val != NULL);
    int *vdia_idiag = (int *)malloc(5 * sizeof(int));
    assert(vdia_idiag != NULL);
    double *vbr_val = (double *)malloc(73 * sizeof(double));
    assert(vbr_val != NULL);
    FILE *matrix_file = fopen("<PATH>/fixture.sabledata", "r");
    assert(matrix_file != NULL);
    read_double_array(matrix_file, vdia_val, 15);
    read_int_array(matrix_file, vdia_idiag, 5);
    read_double_array(matrix_file, vbr_val, 73);
    fclose(matrix_file);
    FILE *rhs_file = fopen("<PATH>/x.matrix", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 7168);
    fclose(rhs_file);

double **vdia_val_pad = (double **)malloc(sizeof(double *) * 2);
assert(vdia_val_pad != NULL);
for (long s = 0; s < 1; s++) {
    size_t cells = (size_t)(vdia_val_seg_nd[s] + 6) * vdia_val_seg_nrows[s];
    vdia_val_pad[s] = (double *)aligned_alloc(64, (cells * sizeof(double) + 63) / 64 * 64);
    assert(vdia_val_pad[s] != NULL);
    memset(vdia_val_pad[s], 0, cells * sizeof(double));
    memcpy(vdia_val_pad[s] + 3 * vdia_val_seg_nrows[s], vdia_val + vdia_val_seg_voff[s],
           (size_t)vdia_val_seg_nd[s] * vdia_val_seg_nrows[s] * sizeof(double));
}
    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(3, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 7168 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
vdia_val_spmm(y, x, vdia_val, vdia_val_pad, 512);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
vbr_val_spmm_naive_block(y, x, vbr_val, 3, 6, 3, 6, 0, 512);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[1][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    14 - 6, 512, 14 - 6,
    1.0,
    &vbr_val[9], 14 - 6,
    &x[6 * 512], 512,
    1.0,
    &y[6 * 512], 512);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[2][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

for (long s = 0; s < 1; s++) free(vdia_val_pad[s]);
free(vdia_val_pad);
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
    for (int i = 0; i < 7168; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vdia_val);
    free(vdia_idiag);
    free(vbr_val);
    free(dispatch_part_times);
    free(x);
    free(y);
    return 0;
}
