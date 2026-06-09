/*
 * Interpreted VDIA(+VBR+CSR) SpMV.
 *
 * Extends interp_vbr_csr_spmv.c with a leading VDIA (variable-diagonal) part.
 * As before, the staged SABLE compiler unrolls each format's traversal at
 * code-generation time (one hard-coded segment/block call apiece); this program
 * instead reads every indirection array from a .sabledata file and walks them at
 * *run time*, dispatching:
 *
 *   Dispatch 1 = VDIA : per segment, MKL DIA (mkl_ddiamv)
 *   Dispatch 2 = VBR  : per packed block, mixed MKL/naive (shape heuristic)
 *   Dispatch 3 = CSR  : residual, backend chosen at run time (naive | mkl)
 *
 * It still walks the VDIA segment metadata and VBR block indirection at run
 * time; the per-segment / per-block compute is dispatched to the same library
 * calls SABLE's staged code uses (mkl_ddiamv, cblas_dgemv, mkl_sparse_d_mv).
 *
 * Prints, over SABLE_BENCH iterations, per-dispatch nanosecond timings:
 *   "Dispatch 1" (VDIA), "Dispatch 2" (VBR), "Dispatch 3" (CSR).
 *
 * Build:
 *   gcc -O2 -o interp_vdia_vbr_csr_spmv interp_vdia_vbr_csr_spmv.c \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt
 * Run:
 *   ./interp_vdia_vbr_csr_spmv DATA.sabledata x.vector [csr_backend=naive|mkl] [y]
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

#define SABLE_BENCH 100

/* ---- dynamic .sabledata readers -------------------------------------- */
/* Each array is one line  label=[v,v,...]; we skip to '[', read comma-separated
 * values until ']', growing the buffer, so no size has to be known up front. */

static void skip_to_array(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF && c != '[') {}
    if (c == EOF) {
        fprintf(stderr, "Unexpected end of data file (no '[')\n");
        exit(1);
    }
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

static double *read_dense_vector(const char *path, int size) {
    FILE *f = fopen(path, "r");
    assert(f != NULL);
    double *out = (double *)malloc((size > 0 ? size : 1) * sizeof(double));
    assert(out != NULL);
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%lf", &out[i]) != 1) {
            fprintf(stderr, "Failed to read dense input\n");
            exit(1);
        }
        if (i + 1 < size) {
            int comma = fgetc(f);
            if (comma != ',') { fprintf(stderr, "Malformed dense input\n"); exit(1); }
        }
    }
    fclose(f);
    return out;
}

/* ---- VDIA: per-segment MKL DIA (mirrors MKLDIASpmv) -------------------- */

static void run_vdia(int nsegments,
                     const int *seg_row_start, const int *seg_nrows, const int *seg_ndiags,
                     const int *seg_idiag_ptr, const int *seg_val_ptr,
                     const int *idiag, const double *val, int ncols,
                     const double *x, double *y) {
    char transa = 'N';
    char matdescra[6] = {'G', ' ', ' ', 'C', ' ', ' '};
    double alpha = 1.0, beta = 1.0;
    for (int s = 0; s < nsegments; s++) {
        MKL_INT m = seg_nrows[s];
        MKL_INT k = ncols;
        MKL_INT lval = seg_nrows[s];
        MKL_INT ndiag = seg_ndiags[s];
        int row0 = seg_row_start[s];
        mkl_ddiamv(&transa, &m, &k, &alpha, matdescra,
                   &val[seg_val_ptr[s]], &lval,
                   (MKL_INT *)&idiag[seg_idiag_ptr[s]], &ndiag,
                   &x[0], &beta, &y[row0]);
    }
}

/* ---- VBR: mixed dispatch heuristic (mirrors MixedVBRSpmv) -------------- */
static int should_use_mkl(int rows, int cols) {
    int min_dim = rows < cols ? rows : cols;
    int max_dim = rows > cols ? rows : cols;
    if (min_dim < 8) return 0;
    return ((double)max_dim / (double)min_dim) <= 100.0;
}

static void spmv_block_naive(const double *val, int offset,
                             int r0, int r1, int c0, int c1,
                             const double *x, double *y) {
    int nrows = r1 - r0;
    for (int i = r0; i < r1; i++) {
        for (int j = c0; j < c1; j++) {
            y[i] += val[offset + (j - c0) * nrows + (i - r0)] * x[j];
        }
    }
}

static void spmv_block_mkl(const double *val, int offset,
                           int r0, int r1, int c0, int c1,
                           const double *x, double *y) {
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                r1 - r0, c1 - c0, 1.0,
                &val[offset], r1 - r0,
                &x[c0], 1, 1.0,
                &y[r0], 1);
}

static void run_vbr(int n_brows,
                    const int *rpntr, const int *cpntr,
                    const int *bpntrb, const int *bindx, const int *indx,
                    const double *val, const double *x, double *y) {
    /* VBR stores only packed dense blocks.  bpntrb is a CSR-style row pointer:
     * block-row a's packed blocks are bindx[bpntrb[a] .. bpntrb[a+1]), and the
     * k-th packed block reads its column-major values at val + indx[k]. */
    for (int a = 0; a < n_brows; a++) {
        for (int k = bpntrb[a]; k < bpntrb[a + 1]; k++) {
            int b = bindx[k];
            int r0 = rpntr[a], r1 = rpntr[a + 1];
            int c0 = cpntr[b], c1 = cpntr[b + 1];
            if (should_use_mkl(r1 - r0, c1 - c0)) {
                spmv_block_mkl(val, indx[k], r0, r1, c0, c1, x, y);
            } else {
                spmv_block_naive(val, indx[k], r0, r1, c0, c1, x, y);
            }
        }
    }
}

/* ---- CSR residual ----------------------------------------------------- */

static void run_csr(int nrows, const int *csr_indptr, const int *csr_indices,
                    const double *csr_val, const double *x, double *y) {
    for (int i = 0; i < nrows; i++) {
        for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++) {
            y[i] += csr_val[p] * x[csr_indices[p]];
        }
    }
}

static double elapsed_ns(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec);
}

int main(int argc, char **argv) {
    const char *data_path = argc > 1 ? argv[1] : "fixture_vdia.sabledata";
    const char *rhs_path  = argc > 2 ? argv[2] : "x.vector";
    int use_mkl_csr = argc > 3 && strcmp(argv[3], "mkl") == 0;
    int print_y = argc > 4 && strcmp(argv[4], "y") == 0;

    FILE *f = fopen(data_path, "r");
    assert(f != NULL);

    /* ---- VDIA arrays ---- */
    int n_srs, n_sn, n_snd, n_sip, n_svp, n_idiag, n_vdval;
    int *seg_row_start = read_int_array(f, &n_srs);
    int *seg_nrows     = read_int_array(f, &n_sn);
    int *seg_ndiags    = read_int_array(f, &n_snd);
    int *seg_idiag_ptr = read_int_array(f, &n_sip);
    int *seg_val_ptr   = read_int_array(f, &n_svp);
    int *vdia_idiag    = read_int_array(f, &n_idiag);
    double *vdia_val   = read_double_array(f, &n_vdval);
    int nsegments = n_srs;

    /* ---- VBR arrays (only packed dense blocks; bpntrb is CSR-style) ---- */
    int n_rpntr, n_cpntr, n_bpntrb, n_bindx, n_indx, n_val;
    int *rpntr   = read_int_array(f, &n_rpntr);
    int *cpntr   = read_int_array(f, &n_cpntr);
    int *bpntrb  = read_int_array(f, &n_bpntrb);
    int *bindx   = read_int_array(f, &n_bindx);
    int *indx    = read_int_array(f, &n_indx);
    double *vbr_val = read_double_array(f, &n_val);

    /* ---- CSR arrays ---- */
    int n_indptr, n_indices, n_csrval;
    int *csr_indptr  = read_int_array(f, &n_indptr);
    int *csr_indices = read_int_array(f, &n_indices);
    double *csr_val  = read_double_array(f, &n_csrval);
    fclose(f);

    /* Matrix shape implied by the VBR partition pointers. */
    int n_brows = n_rpntr - 1;
    int n_bcols = n_cpntr - 1;
    int nrows = rpntr[n_brows];
    int ncols = cpntr[n_bcols];

    double *x = read_dense_vector(rhs_path, ncols);
    double *y = (double *)calloc(nrows > 0 ? nrows : 1, sizeof(double));
    assert(y != NULL);

    /* mkl_ddiamv treats idiag relative to the segment's own (0,0); SABLE stores
     * segment-relative diagonals (the naive path adds row0).  Build a row0-shifted
     * copy once, outside the timed loop (NOT written to .sabledata). */
    int *idiag_mkl = (int *)malloc((n_idiag > 0 ? n_idiag : 1) * sizeof(int));
    assert(idiag_mkl != NULL);
    for (int s = 0; s < nsegments; s++)
        for (int d = 0; d < seg_ndiags[s]; d++)
            idiag_mkl[seg_idiag_ptr[s] + d] = vdia_idiag[seg_idiag_ptr[s] + d] + seg_row_start[s];

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
        memset(y, 0, (nrows > 0 ? nrows : 1) * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        run_vdia(nsegments, seg_row_start, seg_nrows, seg_ndiags, seg_idiag_ptr,
                 seg_val_ptr, idiag_mkl, vdia_val, ncols, x, y);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        t_vdia[iter] = elapsed_ns(t1, t2);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        run_vbr(n_brows, rpntr, cpntr, bpntrb, bindx, indx, vbr_val, x, y);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        t_vbr[iter] = elapsed_ns(t1, t2);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (have_mkl_csr) {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_handle, csr_descr, x, 1.0, y);
        } else {
            run_csr(nrows, csr_indptr, csr_indices, csr_val, x, y);
        }
        clock_gettime(CLOCK_MONOTONIC, &t2);
        t_csr[iter] = elapsed_ns(t1, t2);
    }
    if (print_y) {
        for (int i = 0; i < nrows; i++) printf("%.17g\n", y[i]);
    } else {
        printf("Dispatch 1: "); for (int i = 0; i < SABLE_BENCH; i++) printf("%.0f,", t_vdia[i]); printf("\n");
        printf("Dispatch 2: "); for (int i = 0; i < SABLE_BENCH; i++) printf("%.0f,", t_vbr[i]);  printf("\n");
        printf("Dispatch 3: "); for (int i = 0; i < SABLE_BENCH; i++) printf("%.0f,", t_csr[i]);  printf("\n");
    }
    free(t_vdia); free(t_vbr); free(t_csr);
    if (have_mkl_csr) mkl_sparse_destroy(csr_handle);

    free(seg_row_start); free(seg_nrows); free(seg_ndiags); free(seg_idiag_ptr);
    free(seg_val_ptr); free(vdia_idiag); free(vdia_val); free(idiag_mkl);
    free(rpntr); free(cpntr); free(bpntrb); free(bindx);
    free(indx); free(vbr_val);
    free(csr_indptr); free(csr_indices); free(csr_val);
    free(x); free(y);
    return 0;
}
