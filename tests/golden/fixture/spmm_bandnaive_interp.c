#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


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

static void vdia_val_spmm_naive_segment(double *y, const double *x, const double *vdia_val, const int *vdia_idiag, int row0, int nrows, int ndiags, int idiag_off, int val_off, int nrhs) {
for (int row = 0; row < nrows; row++) {
    for (int d = 0; d < ndiags; d++) {
        int col = row0 + row + vdia_idiag[idiag_off + d];
        if (0 <= col && col < 14) {
            double a = vdia_val[val_off + d * nrows + row];
            for (int rhs_col = 0; rhs_col < nrhs; rhs_col++) {
                y[(row0 + row) * nrhs + rhs_col] += a * x[col * nrhs + rhs_col];
            }
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
    int *_interp_vdia_val_spmm_naive_segment_row0 = (int *)malloc(1 * sizeof(int));
    assert(_interp_vdia_val_spmm_naive_segment_row0 != NULL);
    int *_interp_vdia_val_spmm_naive_segment_nrows = (int *)malloc(1 * sizeof(int));
    assert(_interp_vdia_val_spmm_naive_segment_nrows != NULL);
    int *_interp_vdia_val_spmm_naive_segment_ndiags = (int *)malloc(1 * sizeof(int));
    assert(_interp_vdia_val_spmm_naive_segment_ndiags != NULL);
    int *_interp_vdia_val_spmm_naive_segment_idiag_off = (int *)malloc(1 * sizeof(int));
    assert(_interp_vdia_val_spmm_naive_segment_idiag_off != NULL);
    int *_interp_vdia_val_spmm_naive_segment_val_off = (int *)malloc(1 * sizeof(int));
    assert(_interp_vdia_val_spmm_naive_segment_val_off != NULL);
    int *_interp_vdia_val_spmm_naive_segment_nrhs = (int *)malloc(1 * sizeof(int));
    assert(_interp_vdia_val_spmm_naive_segment_nrhs != NULL);
    FILE *matrix_file = fopen("<PATH>/fixture.sabledata", "r");
    assert(matrix_file != NULL);
    read_double_array(matrix_file, vdia_val, 15);
    read_int_array(matrix_file, vdia_idiag, 5);
    read_int_array(matrix_file, _interp_vdia_val_spmm_naive_segment_row0, 1);
    read_int_array(matrix_file, _interp_vdia_val_spmm_naive_segment_nrows, 1);
    read_int_array(matrix_file, _interp_vdia_val_spmm_naive_segment_ndiags, 1);
    read_int_array(matrix_file, _interp_vdia_val_spmm_naive_segment_idiag_off, 1);
    read_int_array(matrix_file, _interp_vdia_val_spmm_naive_segment_val_off, 1);
    read_int_array(matrix_file, _interp_vdia_val_spmm_naive_segment_nrhs, 1);
    fclose(matrix_file);
    FILE *rhs_file = fopen("<PATH>/x.matrix", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 7168);
    fclose(rhs_file);

    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(1, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 7168 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
for (int _interp_s = 0; _interp_s < 1; _interp_s++) {
    vdia_val_spmm_naive_segment(y, x, vdia_val, vdia_idiag, _interp_vdia_val_spmm_naive_segment_row0[_interp_s], _interp_vdia_val_spmm_naive_segment_nrows[_interp_s], _interp_vdia_val_spmm_naive_segment_ndiags[_interp_s], _interp_vdia_val_spmm_naive_segment_idiag_off[_interp_s], _interp_vdia_val_spmm_naive_segment_val_off[_interp_s], _interp_vdia_val_spmm_naive_segment_nrhs[_interp_s]);
}
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

    printf("Dispatch 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[0][i]);
    }
    printf("\n");
    printf("\n");
    for (int i = 0; i < 7168; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vdia_val);
    free(vdia_idiag);
    free(_interp_vdia_val_spmm_naive_segment_row0);
    free(_interp_vdia_val_spmm_naive_segment_nrows);
    free(_interp_vdia_val_spmm_naive_segment_ndiags);
    free(_interp_vdia_val_spmm_naive_segment_idiag_off);
    free(_interp_vdia_val_spmm_naive_segment_val_off);
    free(_interp_vdia_val_spmm_naive_segment_nrhs);
    free(dispatch_part_times);
    free(x);
    free(y);
    return 0;
}
