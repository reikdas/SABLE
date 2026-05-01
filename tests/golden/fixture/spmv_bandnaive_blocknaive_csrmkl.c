#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

static void vdia_data_spmv_naive_segment(double *y, const double *x, const double *vdia_data, int ncols, int diag_center, int row0, int row1, int lower, int upper, int base) {
int rows = row1 - row0;
int ndiags = lower + upper + 1;
int diag_start = diag_center - lower;
for (int local_diag = 0; local_diag < ndiags; local_diag++) {
    int diag = diag_start + local_diag;
    for (int row = row0; row < row1; row++) {
        int col = row + diag;
        if (0 <= col && col < ncols) {
            y[row] += vdia_data[base + local_diag * rows + (row - row0)] * x[col];
        }
    }
}
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
vdia_data_spmv_naive_segment(y, x, vdia_data, 14, 0, 0, 3, 2, 2, 0);
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
