#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "spf_structure.h"
#include "spf_executors.h"


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


static void write_csr_matrix_market(
    const char *path,
    int nrows,
    int ncols,
    int nnz,
    const int *indptr,
    const int *indices,
    const double *values
) {
    FILE *file = fopen(path, "w");
    assert(file != NULL);
    fprintf(file, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(file, "%d %d %d\n", nrows, ncols, nnz);
    for (int row = 0; row < nrows; row++) {
        for (int p = indptr[row]; p < indptr[row + 1]; p++) {
            fprintf(file, "%d %d %.17g\n", row + 1, indices[p] + 1, values[p]);
        }
    }
    fclose(file);
}


int main(void) {
    double *y = (double *)calloc(6, sizeof(double));
    double *x = (double *)malloc(6 * sizeof(double));
    assert(y != NULL);
    assert(x != NULL);
    double *vbr_val = (double *)malloc(9 * sizeof(double));
    assert(vbr_val != NULL);
    int *vbr_indx = (int *)malloc(2 * sizeof(int));
    assert(vbr_indx != NULL);
    int *vbr_bindx = (int *)malloc(2 * sizeof(int));
    assert(vbr_bindx != NULL);
    int *vbr_rpntr = (int *)malloc(3 * sizeof(int));
    assert(vbr_rpntr != NULL);
    int *vbr_cpntr = (int *)malloc(3 * sizeof(int));
    assert(vbr_cpntr != NULL);
    int *vbr_bpntrb = (int *)malloc(2 * sizeof(int));
    assert(vbr_bpntrb != NULL);
    int *vbr_bpntre = (int *)malloc(2 * sizeof(int));
    assert(vbr_bpntre != NULL);
    int *vbr_ublocks = (int *)malloc(1 * sizeof(int));
    assert(vbr_ublocks != NULL);
    int *csr_indptr = (int *)malloc(7 * sizeof(int));
    assert(csr_indptr != NULL);
    int *csr_indices = (int *)malloc(3 * sizeof(int));
    assert(csr_indices != NULL);
    double *csr_val = (double *)malloc(3 * sizeof(double));
    assert(csr_val != NULL);
    FILE *matrix_file = fopen("<PATH>/fixture.sabledata", "r");
    assert(matrix_file != NULL);
    read_double_array(matrix_file, vbr_val, 9);
    read_int_array(matrix_file, vbr_indx, 2);
    read_int_array(matrix_file, vbr_bindx, 2);
    read_int_array(matrix_file, vbr_rpntr, 3);
    read_int_array(matrix_file, vbr_cpntr, 3);
    read_int_array(matrix_file, vbr_bpntrb, 2);
    read_int_array(matrix_file, vbr_bpntre, 2);
    read_int_array(matrix_file, vbr_ublocks, 1);
    read_int_array(matrix_file, csr_indptr, 7);
    read_int_array(matrix_file, csr_indices, 3);
    read_double_array(matrix_file, csr_val, 3);
    fclose(matrix_file);
    FILE *rhs_file = fopen("<PATH>/x.vector", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 6);
    fclose(rhs_file);

char csr_indptr_mtx_path[] = "csr_indptr_input_396053e0547b.mtx";
char csr_indptr_uzp_dir[] = "csr_indptr_tmp_396053e0547b";
write_csr_matrix_market(csr_indptr_mtx_path, 6, 6, 3,
    csr_indptr, csr_indices, csr_val);
char csr_indptr_cmd[4096];
snprintf(csr_indptr_cmd, sizeof(csr_indptr_cmd),
    "\"<REPO>/uzp_prepare.sh\" \"%s\" \"%s\"",
    csr_indptr_mtx_path, csr_indptr_uzp_dir);
int csr_indptr_rc = system(csr_indptr_cmd);
assert(csr_indptr_rc == 0);
char csr_indptr_path[1024];
snprintf(csr_indptr_path, sizeof(csr_indptr_path), "%s/csr_indptr_input_396053e0547b.tuned.uzp",
    csr_indptr_uzp_dir);
s_spf_structure_t *csr_indptr_spf_mat = spf_matrix_read_from_file(csr_indptr_path);
assert(csr_indptr_spf_mat != NULL);
    struct timespec t1, t2;
    double *sparse_times = (double *)calloc(1, sizeof(double));
    double *dense_times = (double *)calloc(1, sizeof(double));
    double (*dense_block_times)[1] = (double (*)[1])calloc(1, 1 * sizeof(double));
    assert(sparse_times != NULL);
    assert(dense_times != NULL);
    assert(dense_block_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 6 * sizeof(double));
        double iter_sparse_ns = 0.0;
        double iter_dense_ns = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &t1);
for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
        y[i] += vbr_val[0 + (j - 0) * (3 - 0) + (i - 0)] * x[j];
    }
}
        clock_gettime(CLOCK_MONOTONIC, &t2);
        iter_dense_ns += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        dense_block_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
spf_executors_spf_matrix_dense_vector_product(csr_indptr_spf_mat, x, y, 6, 6, 0);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        iter_sparse_ns += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        sparse_times[iter] = iter_sparse_ns;
        dense_times[iter] = iter_dense_ns;
    }

    printf("Sparse: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", sparse_times[i]);
    }
    printf("\n");
    printf("Dense: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dense_times[i]);
    }
    printf("\n");
    printf("Dense Block 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dense_block_times[0][i]);
    }
    printf("\n");
    printf("\n");
    for (int i = 0; i < 6; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vbr_val);
    free(vbr_indx);
    free(vbr_bindx);
    free(vbr_rpntr);
    free(vbr_cpntr);
    free(vbr_bpntrb);
    free(vbr_bpntre);
    free(vbr_ublocks);
    free(csr_indptr);
    free(csr_indices);
    free(csr_val);
    free(dense_block_times);
    free(dense_times);
    free(sparse_times);
    free(x);
    free(y);
    return 0;
}
