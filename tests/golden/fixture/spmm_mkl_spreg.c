#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include <mkl_cblas.h>
#ifdef __cplusplus
#ifndef restrict
#define restrict __restrict__
#endif /* restrict */
#endif /* __cplusplus */
#include "spmm_spreg_wrapper.h"


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


int main(void) {
    double *y = (double *)calloc(3072, sizeof(double));
    double *x = (double *)malloc(3072 * sizeof(double));
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
    FILE *rhs_file = fopen("<PATH>/x.matrix", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 3072);
    fclose(rhs_file);

void *csr_indptr_spreg_handle = spmm_spreg_init(csr_val, csr_indices, csr_indptr,
    6, 6, 512, 1);
if (csr_indptr_spreg_handle == NULL) {
    fprintf(stderr, "Failed to initialize sparse-register-tiling executor\n");
    return 1;
}
double *csr_indptr_spreg_y = (double *)calloc(3072, sizeof(double));
assert(csr_indptr_spreg_y != NULL);
    struct timespec t1, t2;
    double *sparse_times = (double *)calloc(1, sizeof(double));
    double *dense_times = (double *)calloc(1, sizeof(double));
    double (*dense_block_times)[1] = (double (*)[1])calloc(1, 1 * sizeof(double));
    assert(sparse_times != NULL);
    assert(dense_times != NULL);
    assert(dense_block_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 3072 * sizeof(double));
        double iter_sparse_ns = 0.0;
        double iter_dense_ns = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &t1);
cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    3 - 0, 512, 3 - 0,
    1.0,
    &vbr_val[0], 3 - 0,
    &x[0 * 512], 512,
    1.0,
    &y[0 * 512], 512);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        iter_dense_ns += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        dense_block_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
spmm_spreg_execute(csr_indptr_spreg_handle, csr_indptr_spreg_y, x);
for (int i = 0; i < 3072; i++) {
    y[i] += csr_indptr_spreg_y[i];
}
        clock_gettime(CLOCK_MONOTONIC, &t2);
        iter_sparse_ns += (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        sparse_times[iter] = iter_sparse_ns;
        dense_times[iter] = iter_dense_ns;
    }

spmm_spreg_cleanup(csr_indptr_spreg_handle);
free(csr_indptr_spreg_y);
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
    for (int i = 0; i < 3072; i++) {
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
