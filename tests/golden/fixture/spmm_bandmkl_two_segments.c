#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>


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

static const MKL_INT vdia_val_mkl_diag[] = {0, 1, 2, 3};


int main(void) {
    double *y = (double *)calloc(24, sizeof(double));
    double *x = (double *)malloc(24 * sizeof(double));
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
    FILE *rhs_file = fopen("<PATH>/x.matrix", "r");
    assert(rhs_file != NULL);
    read_dense_input(rhs_file, x, 24);
    fclose(rhs_file);

double *vdia_val_mkl_xc = (double *)malloc((long)6 * 4 * sizeof(double));
double *vdia_val_mkl_yc = (double *)calloc((long)6 * 4, sizeof(double));
assert(vdia_val_mkl_xc != NULL && vdia_val_mkl_yc != NULL);
for (int _c = 0; _c < 6; _c++)
    for (int _r = 0; _r < 4; _r++)
        vdia_val_mkl_xc[_c + (long)_r * 6] = x[(long)_c * 4 + _r];
    struct timespec t1, t2;
    double (*dispatch_part_times)[1] = (double (*)[1])calloc(2, 1 * sizeof(double));
    assert(dispatch_part_times != NULL);
    for (int iter = 0; iter < 1; iter++) {
        memset(y, 0, 24 * sizeof(double));
        clock_gettime(CLOCK_MONOTONIC, &t1);
{
MKL_INT mkl_m = 3;
MKL_INT mkl_n = 4;
MKL_INT mkl_k = 6;
MKL_INT mkl_lval = 3;
MKL_INT mkl_ndiag = 2;
MKL_INT mkl_ldb = 6;
MKL_INT mkl_ldc = 6;
double mkl_alpha = 1.0;
double mkl_beta = 0.0;
char mkl_transa = 'N';
char mkl_matdescra[6] = {'G', ' ', ' ', 'C', ' ', ' '};
mkl_ddiamm(&mkl_transa, &mkl_m, &mkl_n, &mkl_k, &mkl_alpha, mkl_matdescra,
           &vdia_val[0], &mkl_lval, (MKL_INT *)&vdia_val_mkl_diag[0], &mkl_ndiag,
           &vdia_val_mkl_xc[0], &mkl_ldb, &mkl_beta, &vdia_val_mkl_yc[0], &mkl_ldc);
}
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[0][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t1);
{
MKL_INT mkl_m = 3;
MKL_INT mkl_n = 4;
MKL_INT mkl_k = 6;
MKL_INT mkl_lval = 3;
MKL_INT mkl_ndiag = 2;
MKL_INT mkl_ldb = 6;
MKL_INT mkl_ldc = 6;
double mkl_alpha = 1.0;
double mkl_beta = 0.0;
char mkl_transa = 'N';
char mkl_matdescra[6] = {'G', ' ', ' ', 'C', ' ', ' '};
mkl_ddiamm(&mkl_transa, &mkl_m, &mkl_n, &mkl_k, &mkl_alpha, mkl_matdescra,
           &vdia_val[6], &mkl_lval, (MKL_INT *)&vdia_val_mkl_diag[2], &mkl_ndiag,
           &vdia_val_mkl_xc[0], &mkl_ldb, &mkl_beta, &vdia_val_mkl_yc[3], &mkl_ldc);
}
        clock_gettime(CLOCK_MONOTONIC, &t2);
        dispatch_part_times[1][iter] = (t2.tv_sec - t1.tv_sec) * 1000000000.0 + (t2.tv_nsec - t1.tv_nsec);
    }

for (int _row = 0; _row < 6; _row++)
    for (int _r = 0; _r < 4; _r++)
        y[(long)_row * 4 + _r] += vdia_val_mkl_yc[(long)_row + (long)_r * 6];
free(vdia_val_mkl_xc);
free(vdia_val_mkl_yc);
    printf("Dispatch 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[0][i] + dispatch_part_times[1][i]);
    }
    printf("\n");
    printf("Dispatch 1 Part 1: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[0][i]);
    }
    printf("\n");
    printf("Dispatch 1 Part 2: ");
    for (int i = 0; i < 1; i++) {
        printf("%.0f,", dispatch_part_times[1][i]);
    }
    printf("\n");
    printf("\n");
    for (int i = 0; i < 24; i++) {
        printf("%.17g\n", y[i]);
    }
    free(vdia_val);
    free(vdia_idiag);
    free(dispatch_part_times);
    free(x);
    free(y);
    return 0;
}
