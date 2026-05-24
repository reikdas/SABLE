#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

static void vdia_val_spmv_naive_segment(double *y, const double *x, const double *vdia_val, const int *vdia_idiag, int row0, int nrows, int ndiags, int idiag_off, int val_off) {
for (int d = 0; d < ndiags; d++) {
    int diag = vdia_idiag[idiag_off + d];
    for (int row = 0; row < nrows; row++) {
        int col = row0 + row + diag;
        if (0 <= col && col < 14) {
            y[row0 + row] += vdia_val[val_off + d * nrows + row] * x[col];
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
    FILE *rhs_file = fopen("<PATH>/x.vector", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 14);
    fclose(rhs_file);

    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(3, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 14 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
vdia_val_spmv_naive_segment(y, x, vdia_val, vdia_idiag, 0, 3, 5, 0, 0);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
vbr_val_spmv_naive_block(y, x, vbr_val, 3, 6, 3, 6, 0);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[1][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
cblas_dgemv(CblasColMajor, CblasNoTrans,
    14 - 6, 14 - 6,
    1.0,
    &vbr_val[9], 14 - 6,
    &x[6], 1,
    1.0,
    &y[6], 1);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[2][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

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
    for (int i = 0; i < 14; i++) {
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
