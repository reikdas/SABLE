#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define always_inline __inline__ __attribute__((always_inline))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

typedef unsigned int u32;
typedef unsigned long long u64;

struct csr_matrix {
  int m, rows, cols;
  double *nnz;
  int *col, *rowb, *rowe;
  int *tstart;
  int *tend;
};
struct csr_matrix input_matrix(int m, int rows, int cols, double *nnz_arr, int *indices_arr, int *indptr_arr);
void destroy_matrix(struct csr_matrix *mat);
struct csr_matrix apply_order(struct csr_matrix *mat, int **tasks, int *task_sizes, int task_count);

/* TR (task-reordered) matrix and exported runtime routines used by
   the generated matrix kernels. Moved here so other C files can
   include a single shared header to access these symbols. */
struct tr_matrix {
  struct csr_matrix mat;
  int *spvv8_len;    /* per-task bulk length */
  int **tasks;       /* array of int* (each task's row indices) */
  int *task_sizes;   /* sizes for each task */
  int task_count;
};

/* Reorder/process CSR matrix into internal TR representation. */
struct tr_matrix process(struct csr_matrix *mat);

/* SpMV kernel that consumes a TR matrix and performs y += A*x. */
void spmv_tr_spvv8_kernel(struct tr_matrix *tr, double *x, double *y);

#ifdef __cplusplus
}
#endif
