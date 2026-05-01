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

static void vdia_data_spmm_naive_segment(double *y, const double *x, const double *vdia_data, int ncols, int diag_center, int row0, int row1, int lower, int upper, int base, int nrhs) {
int rows = row1 - row0;
int ndiags = lower + upper + 1;
int diag_start = diag_center - lower;
for (int local_diag = 0; local_diag < ndiags; local_diag++) {
    int diag = diag_start + local_diag;
    for (int row = row0; row < row1; row++) {
        int col = row + diag;
        if (0 <= col && col < ncols) {
            double a = vdia_data[base + local_diag * rows + (row - row0)];
            for (int rhs_col = 0; rhs_col < nrhs; rhs_col++) {
                y[row * nrhs + rhs_col] += a * x[col * nrhs + rhs_col];
            }
        }
    }
}
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
    double *vdia_data = (double *)malloc(15 * sizeof(double));
    assert(vdia_data != NULL);
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
    read_double_array(matrix_file, vdia_data, 15);
    read_double_array(matrix_file, vbr_val, 73);
    read_int_array(matrix_file, csr_indptr, 15);
    read_int_array(matrix_file, csr_indices, 6);
    read_double_array(matrix_file, csr_val, 6);
    fclose(matrix_file);
    FILE *rhs_file = fopen("<PATH>/x.matrix", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 7168);
    fclose(rhs_file);

void *csr_indptr_spreg_handle = spmm_spreg_init(csr_val, csr_indices, csr_indptr,
    14, 14, 512, 1);
if (csr_indptr_spreg_handle == NULL) {
    fprintf(stderr, "Failed to initialize sparse-register-tiling executor\n");
    return 1;
}
double *csr_indptr_spreg_y = (double *)calloc(7168, sizeof(double));
assert(csr_indptr_spreg_y != NULL);
    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(4, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 7168 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
vdia_data_spmm_naive_segment(y, x, vdia_data, 14, 0, 0, 3, 2, 2, 0, 512);
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
        clock_gettime(CLOCK_MONOTONIC, &t1);
spmm_spreg_execute(csr_indptr_spreg_handle, csr_indptr_spreg_y, x);
for (int i = 0; i < 7168; i++) {
    y[i] += csr_indptr_spreg_y[i];
}
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[3][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

spmm_spreg_cleanup(csr_indptr_spreg_handle);
free(csr_indptr_spreg_y);
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
    for (int i = 0; i < 7168; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vdia_data);
    free(vbr_val);
    free(csr_indptr);
    free(csr_indices);
    free(csr_val);
    free(dispatch_part_times);
    free(x);
    free(y);
    return 0;
}
