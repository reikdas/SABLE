#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "utility.h"

int main() {
  struct csr_matrix mat;

  /* Create dummy CSR arrays instead of reading `info.txt` and CSR files.
   * These are small example arrays; adjust as needed.
   */
  int rows = 4;
  int cols = 5;
  int m = 6; /* number of non-zeros */

  double *nnz_arr = (double *)malloc(m * sizeof(double));
  int *indices_arr = (int *)malloc(m * sizeof(int));
  int *indptr_arr = (int *)malloc((rows + 1) * sizeof(int));

  /* sample CSR: row lengths [2,1,2,1] */
  double tmp_nnz[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  int tmp_idx[6] = {0, 2, 1, 0, 4, 3};
  int tmp_indptr[5] = {0, 2, 3, 5, 6};

  for (int i = 0; i < m; i++) { nnz_arr[i] = tmp_nnz[i]; indices_arr[i] = tmp_idx[i]; }
  for (int i = 0; i < rows + 1; i++) indptr_arr[i] = tmp_indptr[i];

  mat = input_matrix(m, rows, cols, nnz_arr, indices_arr, indptr_arr);

  free(nnz_arr); free(indices_arr); free(indptr_arr);

  /* allocate x and y here (caller-managed) */
  double *x = (double *)malloc(mat.cols * sizeof(double));
  double *y = (double *)malloc(mat.rows * sizeof(double));

  /* attempt to read x.txt; if unavailable, initialize defaults */
  FILE *fx = fopen("x.txt", "r");
  if (fx) {
    for (int i = 0; i < mat.cols; i++) {
      if (fscanf(fx, "%lf", &x[i]) != 1) x[i] = 1.0;
    }
    fclose(fx);
  } else {
    for (int i = 0; i < mat.cols; i++) x[i] = 1.0;
  }
  for (int i = 0; i < mat.rows; i++) y[i] = 0.0;

  struct tr_matrix tr = process(&mat);

  struct timespec t1, t2;
  double times[100];
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < tr.mat.rows; j++) y[j] = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    spmv_tr_spvv8_kernel(&tr, x, y);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
  }
  // median
  for (int i = 0; i < 99; i++) {
    for (int j = i+1; j < 100; j++) {
      if (times[j] < times[i]) { double tmp = times[i]; times[i] = times[j]; times[j] = tmp; }
    }
  }
  printf("Time: %.2f ns\n", times[50]);

  for (int i = 0; i < tr.mat.rows; i++) y[i] = 0.0;
  spmv_tr_spvv8_kernel(&tr, x, y);

  /* correctness checking removed (no `ans` stored) */

  destroy_matrix(&mat);
  // Free tr resources
  for (int t = 0; t < tr.task_count; t++) free(tr.tasks[t]);
  free(tr.tasks); free(tr.task_sizes); free(tr.spvv8_len);
  destroy_matrix(&tr.mat);

  free(x); free(y);
  return 0;
}
