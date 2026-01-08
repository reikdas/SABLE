#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "utility.h"

// Helper to allocate 64-byte aligned memory (aligned_alloc requires
// the size to be a multiple of alignment)
// Note: replaced aligned allocations with plain malloc calls below.

/* `struct tr_matrix` is defined in `utility.h`; remove duplicate definition here. */

// ...existing code...

// apply_order: create a new csr matrix that contains rows in the order
// specified by tasks (tasks is array of int* with sizes task_sizes)
struct csr_matrix apply_order(struct csr_matrix *mat, int **tasks, int *task_sizes, int task_count) {
  struct csr_matrix ret;
  ret.m = mat->m;
  ret.rows = mat->rows;
  ret.cols = mat->cols;
  ret.nnz = (double *)malloc(mat->m * sizeof(double));
  ret.col = (int *)malloc(mat->m * sizeof(int));
  ret.rowb = (int *)malloc(mat->rows * sizeof(int));
  ret.rowe = (int *)malloc(mat->rows * sizeof(int));
  ret.tstart = (int *)malloc(task_count * sizeof(int));
  ret.tend = (int *)malloc(task_count * sizeof(int));
  

  int npos = 0;
  int pos = 0;
  int start = 0;
  for (int t = 0; t < task_count; t++) {
    ret.tstart[t] = start;
    ret.tend[t] = start + task_sizes[t];
    start += task_sizes[t];
    for (int k = 0; k < task_sizes[t]; k++) {
      int row = tasks[t][k];
      int b = mat->rowb[row];
      int e = mat->rowe[row];
      ret.rowb[pos] = npos;
      ret.rowe[pos++] = npos + (e - b);
      for (int i = b; i < e; i++) {
        ret.nnz[npos] = mat->nnz[i];
        ret.col[npos++] = mat->col[i];
      }
    }
  }

  return ret;
}

struct csr_matrix input_matrix(int m, int rows, int cols, double *nnz_arr, int *indices_arr, int *indptr_arr) {
  struct csr_matrix mat;

  /* initialize sizes in mat */
  mat.m = m;
  mat.rows = rows;
  mat.cols = cols;

  mat.nnz = (double *)malloc(mat.m * sizeof(double));
  mat.col = (int *)malloc(mat.m * sizeof(int));
  mat.rowb = (int *)malloc(mat.rows * sizeof(int));
  mat.rowe = (int *)malloc(mat.rows * sizeof(int));
  mat.tstart = NULL;
  mat.tend = NULL;

  /* copy nnz and indices */
  for (int i = 0; i < mat.m; i++) mat.nnz[i] = nnz_arr[i];
  for (int i = 0; i < mat.m; i++) mat.col[i] = indices_arr[i];

  /* copy indptr -> rowb/rowe */
  for (int r = 0; r < mat.rows; r++) {
    mat.rowb[r] = indptr_arr[r];
    mat.rowe[r] = indptr_arr[r+1];
  }

  /* ans removed; caller no longer stores expected answers in matrix */

  return mat;
}

void destroy_matrix(struct csr_matrix *mat) {
  if (mat->nnz) free(mat->nnz);
  if (mat->col) free(mat->col);
  if (mat->rowb) free(mat->rowb);
  if (mat->rowe) free(mat->rowe);
  if (mat->tstart) free(mat->tstart);
  if (mat->tend) free(mat->tend);
  /* ans removed; nothing to free here */
}

// Reorder and group rows similar to original C++ implementation.
struct tr_matrix tr_reorder(struct csr_matrix *mat, int **tasks, int *task_sizes, int task_count) {
  struct tr_matrix tr;
  tr.task_count = task_count;
  tr.tasks = (int **)malloc(task_count * sizeof(int*));
  tr.task_sizes = (int *)malloc(task_count * sizeof(int));
  tr.spvv8_len = (int *)malloc(task_count * sizeof(int));

  for (int t = 0; t < task_count; t++) {
    int *task = tasks[t];
    int tsize = task_sizes[t];

    // compute row lengths and find max
    int *rowlen = (int *)malloc(tsize * sizeof(int));
    int maxlen = 0;
    for (int i = 0; i < tsize; i++) {
      int r = task[i];
      int rl = mat->rowe[r] - mat->rowb[r];
      rowlen[i] = rl;
      if (rl > maxlen) maxlen = rl;
    }

    // create buckets by rowlen
    int bucket_count = maxlen + 1;
    int *counts = (int *)calloc(bucket_count, sizeof(int));
    for (int i = 0; i < tsize; i++) counts[rowlen[i]]++;

    int **buckets = (int **)malloc(bucket_count * sizeof(int*));
    for (int i = 0; i < bucket_count; i++) {
      if (counts[i]) buckets[i] = (int *)malloc(counts[i] * sizeof(int));
      else buckets[i] = NULL;
      counts[i] = 0; // reuse to track fill
    }

    for (int i = 0; i < tsize; i++) {
      int rl = rowlen[i];
      buckets[rl][counts[rl]++] = task[i];
    }

    // build order and remain lists
    int *order = (int *)malloc(tsize * sizeof(int));
    int order_pos = 0;
    int *remain = (int *)malloc(tsize * sizeof(int));
    int remain_pos = 0;

    for (int k = 0; k <= maxlen; k++) {
      if (!buckets[k]) continue;
      int size = counts[k];
      int left = size % 8;
      if (k > 32) left = size;
      int bulk = size - left;
      for (int i = 0; i < bulk; i++) order[order_pos++] = buckets[k][i];
      for (int i = bulk; i < size; i++) remain[remain_pos++] = buckets[k][i];
    }

    tr.spvv8_len[t] = order_pos;

    // assemble new task: order then remain
    tr.task_sizes[t] = order_pos + remain_pos;
    tr.tasks[t] = (int *)malloc(tr.task_sizes[t] * sizeof(int));
    memcpy(tr.tasks[t], order, order_pos * sizeof(int));
    memcpy(tr.tasks[t] + order_pos, remain, remain_pos * sizeof(int));

    // free temporary
    free(order); free(remain);
    for (int k = 0; k <= maxlen; k++) if (buckets[k]) free(buckets[k]);
    free(buckets); free(counts); free(rowlen);
  }

  // apply ordering to matrix
  tr.mat = apply_order(mat, tr.tasks, tr.task_sizes, task_count);

  return tr;
}

int is_banded(struct csr_matrix *mat, int band_size) {
  if (band_size == -1) band_size = mat->cols / 64;
  int band_count = 0;
  for (int r = 0; r < mat->rows; r++) {
    int rb = mat->rowb[r];
    int re = mat->rowe[r];
    for (int i = rb; i < re; i++) {
      int col = mat->col[i];
      if (abs(col - r) <= band_size) band_count++;
    }
  }
  if ((double)band_count / mat->m >= 0.3) return 1;
  return 0;
}

struct tr_matrix process(struct csr_matrix *mat) {
  // allocate tasks
  int banded = is_banded(mat, -1);
  int panel_count = mat->rows / 2000;
  if (panel_count < 4) panel_count = 4;
  if (banded) {
    panel_count = mat->rows / 10000;
    if (panel_count < 4) panel_count = 4;
  }
  int panel = panel_count;
  int **tasks = (int **)malloc(panel * sizeof(int*));
  int *task_sizes = (int *)calloc(panel, sizeof(int));
  int *task_caps = (int *)malloc(panel * sizeof(int));
  for (int i = 0; i < panel; i++) { task_caps[i] = 128; tasks[i] = (int *)malloc(task_caps[i] * sizeof(int)); }

  int pos = 0;
  int len = mat->m / panel_count;
  int limit = mat->rows - 7;
  int i;
  int count = 0;
  for (i = 0; i < limit; i += 8) {
    for (int j = 0; j < 8; j++) {
      if (i+j >= mat->rows) break;
      int rowlen = mat->rowe[i + j] - mat->rowb[i + j];
      if (rowlen > 0) {
        if (task_sizes[pos] + 1 > task_caps[pos]) {
          task_caps[pos] *= 2;
          tasks[pos] = (int *)realloc(tasks[pos], task_caps[pos] * sizeof(int));
        }
        tasks[pos][task_sizes[pos]++] = i + j;
        count += rowlen;
      }
    }

    if (count >= len) {
      if (pos + 1 < panel) { pos += 1; count = 0; }
    }
  }

  if (i < mat->rows) {
    for (; i < mat->rows; i++) {
      if (task_sizes[pos] + 1 > task_caps[pos]) {
        task_caps[pos] *= 2;
        tasks[pos] = (int *)realloc(tasks[pos], task_caps[pos] * sizeof(int));
      }
      tasks[pos][task_sizes[pos]++] = i;
    }
  }

  struct tr_matrix ret = tr_reorder(mat, tasks, task_sizes, panel);

  // free temporary tasks used for building (tr now owns copies per task)
  for (int t = 0; t < panel; t++) free(tasks[t]);
  free(tasks); free(task_sizes); free(task_caps);

  return ret;
}

always_inline double avx512_fma_spvv_kernel(int *col, double *nnz, int rowlen, double *x) {
  int limit = rowlen - 7;
  int *col_p;
  double *nnz_p;
  double sum = 0;
  __m256i c1;
  __m512d v1, v2, s;
  s = _mm512_setzero_pd();
  int i;

  for (i = 0; i < limit; i += 8) {
    col_p = col + i;
    nnz_p = nnz + i;
    c1 = _mm256_loadu_si256((const __m256i *) col_p);
    v2 = _mm512_i32gather_pd(c1, x, 8);
    v1 = _mm512_loadu_pd(nnz_p);
    s = _mm512_fmadd_pd(v1, v2, s);
  }

  sum += _mm512_reduce_add_pd(s);
  for (; i < rowlen; i++) sum += nnz[i] * x[col[i]];

  return sum;
}

always_inline void avx512_spvv8_kernel_tr(const int *rows, int *rowb, int *rowe, int *col, double *nnz, double *x, double *y) {
  __m256i rs = _mm256_loadu_si256((const __m256i *) rows);
  __m512d acc = _mm512_setzero_pd();

  int rowlen = *rowe - *rowb;
  int base = *rowb;

  // prefetch y for the 8 rows
  {
    int idx0 = rows[0];
    int idx1 = rows[1];
    int idx2 = rows[2];
    int idx3 = rows[3];
    int idx4 = rows[4];
    int idx5 = rows[5];
    int idx6 = rows[6];
    int idx7 = rows[7];

    _m_prefetchw(y + idx0);
    _m_prefetchw(y + idx1);
    _m_prefetchw(y + idx2);
    _m_prefetchw(y + idx3);
    _m_prefetchw(y + idx4);
    _m_prefetchw(y + idx5);
    _m_prefetchw(y + idx6);
    _m_prefetchw(y + idx7);
  }

  for (int c = 0; c < rowlen; c++) {
    int offset = base + c * 8;
    __m256i cc = _mm256_loadu_si256((const __m256i *) (col + offset));
    __m512d nz = _mm512_loadu_pd(nnz + offset);
    __m512d xx = _mm512_i32gather_pd(cc, x, 8);
    acc = _mm512_fmadd_pd(nz, xx, acc);
  }

  _mm512_i32scatter_pd(y, rs, acc, 8);
}

void spmv_tr_spvv8_kernel(struct tr_matrix *tr, double *x, double *y) {
  int size = tr->task_count;
  for (int tid = 0; tid < size; tid++) {
    int *task = tr->tasks[tid];
    struct csr_matrix *mat = &tr->mat;
    int start = mat->tstart[tid];
    int end = mat->tend[tid];
    int limit = tr->spvv8_len[tid];
    int p, c;
    for (p = start, c = 0; c < limit; p += 8, c += 8) {
      avx512_spvv8_kernel_tr(task + c, mat->rowb + p, mat->rowe + p, mat->col, mat->nnz, x, y);
    }
    for (; p < end; p++) {
      int r = task[p - start];
      int begin = mat->rowb[p];
      int en = mat->rowe[p];
      int rowlen = en - begin;
      _mm_prefetch(y + r, _MM_HINT_ET1);
      y[r] = avx512_fma_spvv_kernel(mat->col + begin, mat->nnz + begin, rowlen, x);
    }
  }
}

/* `main` moved to `src/spmv_spv8_main.c` so this compilation unit can be
   built into an object that does not define `main`. */
