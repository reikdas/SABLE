/*
 * Interpreted VDIA(+VBR+CSR) SpMM (sparse matrix x dense matrix).
 *
 * Extends interp_vbr_csr_spmm.c with a leading VDIA part.  The block structure
 * of every format is discovered at run time by walking the indirection arrays
 * read from a .sabledata file:
 *
 *   Dispatch 1 = VDIA : per segment, naive diagonal traversal
 *   Dispatch 2 = VBR  : per packed block, mixed MKL/naive (shape heuristic)
 *   Dispatch 3 = CSR  : residual, backend chosen at run time (naive | mkl)
 *
 * The dense operands x (ncols x nrhs) and y (nrows x nrhs) are row-major; nrhs
 * is inferred from the length of the dense RHS file.  VDIA SpMM uses the naive
 * per-segment diagonal loop (MKL ddiamm loses badly here due to the row-major
 * <-> column-major round-trip); VBR and CSR dispatch to cblas_dgemm /
 * mkl_sparse_d_mm as in SABLE's staged code.
 *
 * Prints, over SABLE_BENCH iterations, per-dispatch nanosecond timings:
 *   "Dispatch 1" (VDIA), "Dispatch 2" (VBR), "Dispatch 3" (CSR).
 *
 * Build:
 *   gcc -O2 -o interp_vdia_vbr_csr_spmm interp_vdia_vbr_csr_spmm.c \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt
 * Run:
 *   ./interp_vdia_vbr_csr_spmm DATA.sabledata x.matrix [csr_backend=naive|mkl] [y]
 *   (a trailing "y" prints the result vector instead of timings, for checking)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_spblas.h>

#define SABLE_BENCH 30

/* ---- dynamic .sabledata readers --------------------------------------- */

static void skip_to_array(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF && c != '[') {}
    if (c == EOF) { fprintf(stderr, "Unexpected end of data file (no '[')\n"); exit(1); }
}

static void finish_line(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {}
}

static int *read_int_array(FILE *f, int *out_n) {
    skip_to_array(f);
    int cap = 8, n = 0;
    int *buf = (int *)malloc(cap * sizeof(int));
    assert(buf != NULL);
    int c = fgetc(f);
    if (c == ']') { finish_line(f); *out_n = 0; return buf; }
    ungetc(c, f);
    for (;;) {
        int v;
        if (fscanf(f, "%d", &v) != 1) { fprintf(stderr, "Failed to read int array\n"); exit(1); }
        if (n == cap) { cap *= 2; buf = (int *)realloc(buf, cap * sizeof(int)); assert(buf); }
        buf[n++] = v;
        c = fgetc(f);
        if (c == ']') break;
        if (c != ',') { fprintf(stderr, "Malformed int array\n"); exit(1); }
    }
    finish_line(f);
    *out_n = n;
    return buf;
}

static double *read_double_array(FILE *f, int *out_n) {
    skip_to_array(f);
    int cap = 8, n = 0;
    double *buf = (double *)malloc(cap * sizeof(double));
    assert(buf != NULL);
    int c = fgetc(f);
    if (c == ']') { finish_line(f); *out_n = 0; return buf; }
    ungetc(c, f);
    for (;;) {
        double v;
        if (fscanf(f, "%lf", &v) != 1) { fprintf(stderr, "Failed to read double array\n"); exit(1); }
        if (n == cap) { cap *= 2; buf = (double *)realloc(buf, cap * sizeof(double)); assert(buf); }
        buf[n++] = v;
        c = fgetc(f);
        if (c == ']') break;
        if (c != ',') { fprintf(stderr, "Malformed double array\n"); exit(1); }
    }
    finish_line(f);
    *out_n = n;
    return buf;
}

/* Read an entire comma-separated dense RHS file (no surrounding brackets). */
static double *read_dense_all(const char *path, int *out_n) {
    FILE *f = fopen(path, "r");
    assert(f != NULL);
    int cap = 16, n = 0;
    double *buf = (double *)malloc(cap * sizeof(double));
    assert(buf != NULL);
    for (;;) {
        double v;
        if (fscanf(f, "%lf", &v) != 1) break;
        if (n == cap) { cap *= 2; buf = (double *)realloc(buf, cap * sizeof(double)); assert(buf); }
        buf[n++] = v;
        int c = fgetc(f);
        if (c != ',') break;
    }
    fclose(f);
    *out_n = n;
    return buf;
}

/* ---- VDIA: naive per-segment diagonal traversal ----------------------- */

static void run_vdia(int nsegments,
                     const int *seg_row_start, const int *seg_nrows, const int *seg_ndiags,
                     const int *seg_idiag_ptr, const int *seg_val_ptr,
                     const int *idiag, const double *val, int ncols,
                     const double *x, double *y, int nrhs) {
    for (int s = 0; s < nsegments; s++) {
        int row0 = seg_row_start[s];
        int nrows = seg_nrows[s];
        int ndiags = seg_ndiags[s];
        int idiag_off = seg_idiag_ptr[s];
        int val_off = seg_val_ptr[s];
        for (int row = 0; row < nrows; row++) {
            for (int d = 0; d < ndiags; d++) {
                int col = row0 + row + idiag[idiag_off + d];
                if (0 <= col && col < ncols) {
                    double a = val[val_off + d * nrows + row];
                    for (int k = 0; k < nrhs; k++) {
                        y[(long)(row0 + row) * nrhs + k] += a * x[(long)col * nrhs + k];
                    }
                }
            }
        }
    }
}

/* ---- VBR: mixed dispatch heuristic (mirrors MixedVBRSpmm) -------------- */
static int should_use_mkl(int rows, int cols) {
    int min_dim = rows < cols ? rows : cols;
    int max_dim = rows > cols ? rows : cols;
    if (min_dim < 8) return 0;
    return ((double)max_dim / (double)min_dim) <= 100.0;
}

static void spmm_block_naive(const double *val, int offset,
                             int r0, int r1, int c0, int c1,
                             const double *x, double *y, int nrhs) {
    int nrows = r1 - r0;
    for (int i = r0; i < r1; i++) {
        for (int j = c0; j < c1; j++) {
            double a = val[offset + (j - c0) * nrows + (i - r0)];
            for (int k = 0; k < nrhs; k++) {
                y[i * nrhs + k] += a * x[j * nrhs + k];
            }
        }
    }
}

static void spmm_block_mkl(const double *val, int offset,
                           int r0, int r1, int c0, int c1,
                           const double *x, double *y, int nrhs) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                r1 - r0, nrhs, c1 - c0, 1.0,
                &val[offset], r1 - r0,
                &x[c0 * nrhs], nrhs, 1.0,
                &y[r0 * nrhs], nrhs);
}

static void run_vbr(int n_brows,
                    const int *rpntr, const int *cpntr,
                    const int *bpntrb, const int *bindx, const int *indx,
                    const double *val, const double *x, double *y, int nrhs) {
    /* VBR stores only packed dense blocks.  bpntrb is a CSR-style row pointer:
     * block-row a's packed blocks are bindx[bpntrb[a] .. bpntrb[a+1]), and the
     * k-th packed block reads its column-major values at val + indx[k]. */
    for (int a = 0; a < n_brows; a++) {
        for (int k = bpntrb[a]; k < bpntrb[a + 1]; k++) {
            int b = bindx[k];
            int r0 = rpntr[a], r1 = rpntr[a + 1];
            int c0 = cpntr[b], c1 = cpntr[b + 1];
            if (should_use_mkl(r1 - r0, c1 - c0)) {
                spmm_block_mkl(val, indx[k], r0, r1, c0, c1, x, y, nrhs);
            } else {
                spmm_block_naive(val, indx[k], r0, r1, c0, c1, x, y, nrhs);
            }
        }
    }
}

/* ---- CSR residual ----------------------------------------------------- */

static void run_csr(int nrows, const int *csr_indptr, const int *csr_indices,
                    const double *csr_val, const double *x, double *y, int nrhs) {
    for (int i = 0; i < nrows; i++) {
        for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++) {
            int col = csr_indices[p];
            double a = csr_val[p];
            for (int k = 0; k < nrhs; k++) {
                y[i * nrhs + k] += a * x[col * nrhs + k];
            }
        }
    }
}

static double elapsed_ns(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec);
}

int main(int argc, char **argv) {
    const char *data_path = argc > 1 ? argv[1] : "fixture_vdia.sabledata";
    const char *rhs_path  = argc > 2 ? argv[2] : "x.matrix";
    int use_mkl_csr = argc > 3 && strcmp(argv[3], "mkl") == 0;
    int print_y = argc > 4 && strcmp(argv[4], "y") == 0;

    FILE *f = fopen(data_path, "r");
    assert(f != NULL);

    int n_srs, n_sn, n_snd, n_sip, n_svp, n_idiag, n_vdval;
    int *seg_row_start = read_int_array(f, &n_srs);
    int *seg_nrows     = read_int_array(f, &n_sn);
    int *seg_ndiags    = read_int_array(f, &n_snd);
    int *seg_idiag_ptr = read_int_array(f, &n_sip);
    int *seg_val_ptr   = read_int_array(f, &n_svp);
    int *vdia_idiag    = read_int_array(f, &n_idiag);
    double *vdia_val   = read_double_array(f, &n_vdval);
    int nsegments = n_srs;

    int n_rpntr, n_cpntr, n_bpntrb, n_bindx, n_indx, n_val;
    int *rpntr   = read_int_array(f, &n_rpntr);
    int *cpntr   = read_int_array(f, &n_cpntr);
    int *bpntrb  = read_int_array(f, &n_bpntrb);
    int *bindx   = read_int_array(f, &n_bindx);
    int *indx    = read_int_array(f, &n_indx);
    double *vbr_val = read_double_array(f, &n_val);

    int n_indptr, n_indices, n_csrval;
    int *csr_indptr  = read_int_array(f, &n_indptr);
    int *csr_indices = read_int_array(f, &n_indices);
    double *csr_val  = read_double_array(f, &n_csrval);
    fclose(f);

    int n_brows = n_rpntr - 1;
    int n_bcols = n_cpntr - 1;
    int nrows = rpntr[n_brows];
    int ncols = cpntr[n_bcols];

    int x_total;
    double *x = read_dense_all(rhs_path, &x_total);
    assert(ncols > 0 && x_total % ncols == 0);
    int nrhs = x_total / ncols;

    long y_size = (long)nrows * nrhs;
    double *y = (double *)calloc(y_size > 0 ? y_size : 1, sizeof(double));
    assert(y != NULL);

    /* MKL CSR: build the handle once, outside the timed loop (as SABLE does). */
    int have_mkl_csr = use_mkl_csr && n_csrval > 0;
    sparse_matrix_t csr_handle;
    struct matrix_descr csr_descr;
    if (have_mkl_csr) {
        csr_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        csr_descr.mode = SPARSE_FILL_MODE_FULL;
        csr_descr.diag = SPARSE_DIAG_NON_UNIT;
        mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO, nrows, ncols,
                                csr_indptr, csr_indptr + 1, csr_indices, csr_val);
    }

    double *t_vdia = (double *)malloc(SABLE_BENCH * sizeof(double));
    double *t_vbr  = (double *)malloc(SABLE_BENCH * sizeof(double));
    double *t_csr  = (double *)malloc(SABLE_BENCH * sizeof(double));
    assert(t_vdia && t_vbr && t_csr);
    struct timespec t1, t2;
    for (int iter = 0; iter < SABLE_BENCH; iter++) {
        memset(y, 0, (y_size > 0 ? y_size : 1) * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        run_vdia(nsegments, seg_row_start, seg_nrows, seg_ndiags, seg_idiag_ptr,
                 seg_val_ptr, vdia_idiag, vdia_val, ncols, x, y, nrhs);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        t_vdia[iter] = elapsed_ns(t1, t2);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        run_vbr(n_brows, rpntr, cpntr, bpntrb, bindx, indx, vbr_val, x, y, nrhs);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        t_vbr[iter] = elapsed_ns(t1, t2);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (have_mkl_csr) {
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_handle, csr_descr,
                            SPARSE_LAYOUT_ROW_MAJOR, x, nrhs, nrhs, 1.0, y, nrhs);
        } else {
            run_csr(nrows, csr_indptr, csr_indices, csr_val, x, y, nrhs);
        }
        clock_gettime(CLOCK_MONOTONIC, &t2);
        t_csr[iter] = elapsed_ns(t1, t2);
    }

    if (print_y) {
        for (long i = 0; i < y_size; i++) printf("%.17g\n", y[i]);
    } else {
        printf("Dispatch 1: "); for (int i = 0; i < SABLE_BENCH; i++) printf("%.0f,", t_vdia[i]); printf("\n");
        printf("Dispatch 2: "); for (int i = 0; i < SABLE_BENCH; i++) printf("%.0f,", t_vbr[i]);  printf("\n");
        printf("Dispatch 3: "); for (int i = 0; i < SABLE_BENCH; i++) printf("%.0f,", t_csr[i]);  printf("\n");
    }
    free(t_vdia); free(t_vbr); free(t_csr);
    if (have_mkl_csr) mkl_sparse_destroy(csr_handle);

    free(seg_row_start); free(seg_nrows); free(seg_ndiags); free(seg_idiag_ptr);
    free(seg_val_ptr); free(vdia_idiag); free(vdia_val);
    free(rpntr); free(cpntr); free(bpntrb); free(bindx);
    free(indx); free(vbr_val);
    free(csr_indptr); free(csr_indices); free(csr_val);
    free(x); free(y);
    return 0;
}
