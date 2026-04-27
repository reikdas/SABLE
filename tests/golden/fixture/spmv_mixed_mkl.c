#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <assert.h>

void spmv_kernel(double *restrict y, const double *restrict x, const double *restrict val, const int i_start, const int i_end, const int j_start, const int j_end, const int val_offset) {
	for (int j = j_start; j < j_end; j++) {
		for (int i = i_start; i < i_end; i++) {
			y[i] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[j]);
		}
	}
}

void spmv_kernel_2(double *restrict y, const double *restrict x, const double *restrict val, const int i_start, const int j_start, const int j_end, const int val_offset) {
	for (int j = j_start; j < j_end; j++) {
		y[i_start] += ((&val[val_offset])[(((j-j_start)))] * x[j]);
	}
}

void spmv_kernel_3(double *restrict y, const double *restrict x, const double *restrict val, const int i_start, const int i_end, const int j_start, const int val_offset) {
	double xj = x[j_start];
	for (int i = i_start; i < i_end; i++) {
		y[i] += ((&val[val_offset])[(i-i_start)] * xj);
	}
}



void spmv_kernel_mkl(double *restrict y, const double *restrict x, const double *restrict val,
                      const int i_start, const int i_end, const int j_start, const int j_end,
                      const int val_offset) {
    const int m = i_end - i_start;  // block_rows
    const int n = j_end - j_start;  // block_cols

    // A is stored column-major (m x n)
    // y = alpha * A * x + beta * y
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n,
                1.0,                        // alpha
                &val[val_offset], m,        // A, lda
                &x[j_start], 1,             // x, incx
                1.0,                        // beta
                &y[i_start], 1);            // y, incy
}


int main() {
	mkl_set_num_threads(1);

	double *y = (double*)malloc(6 * sizeof(double));
	double *x = (double*)malloc(6 * sizeof(double));
	double *val = (double*)malloc(9 * sizeof(double));
	double *csr_val = (double*)malloc(3 * sizeof(double));
	int *indptr = (int*)malloc(7 * sizeof(int));
	int *indices = (int*)malloc(3 * sizeof(int));
	if (!csr_val || !indptr || !indices) {
		printf("Memory allocation failed for csr_val/indptr/indices\n");
		return 1;
	}
	struct timespec t1, t2;
	long sparse_times[1];
	long (*dense_block_times)[1] = (long(*)[1])malloc(1 * 1 * sizeof(long));
	for (int i=0; i<1; i++) {
		sparse_times[i] = 0;
		for (int j=0; j<1; j++) {
			dense_block_times[j][i] = 0;
		}
	}
	for (int i=0; i<1; i++) {
	FILE *file1 = fopen("<PATH>/fixture.vbrc", "r");
	if (file1 == NULL) { printf("Error opening file1"); return 1; }
	FILE *file2 = fopen("<PATH>/generated_vector_6.vector", "r");
	if (file2 == NULL) { printf("Error opening file2"); return 1; }
		memset(y, 0, sizeof(double)*6);
		memset(val, 0, 9 * sizeof(double));
		memset(csr_val, 0, 3 * sizeof(double));
		memset(indptr, 0, 7 * sizeof(int));
		memset(indices, 0, 3 * sizeof(int));
		char c;
		int x_size=0, val_size=0;
		val_size=0;
		assert(fscanf(file1, "val=[%c", &c) == 1);
		if (c != ']') {
			ungetc(c, file1);
			assert(fscanf(file1, "%lf", &val[val_size]) == 1.0);
			val_size++;
			while (1) {
				assert(fscanf(file1, "%c", &c) == 1);
				if (c == ',') {
					assert(fscanf(file1, "%lf", &val[val_size]) == 1.0);
					val_size++;
				} else if (c == ']') {
					break;
				} else {
					assert(0);
				}
			}
		}
		if(fscanf(file1, "%c", &c));
		assert(c=='\n');
		val_size=0;
		assert(fscanf(file1, "csr_val=[%c", &c) == 1);
		if (c != ']') {
			ungetc(c, file1);
			assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1.0);
			val_size++;
			while (1) {
				assert(fscanf(file1, "%c", &c) == 1);
				if (c == ',') {
					assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1.0);
					val_size++;
				} else if (c == ']') {
					break;
				} else {
					assert(0);
				}
			}
		}
		if(fscanf(file1, "%c", &c));
		assert(c=='\n');
		val_size=0;
		assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
		val_size++;
		while (1) {
			assert(fscanf(file1, "%c", &c) == 1);
			if (c == ',') {
				assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
				val_size++;
			} else if (c == ']') {
				break;
			} else {
				assert(0);
			}
		}
		if(fscanf(file1, "%c", &c));
		assert(c=='\n');
		val_size=0;
		assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
		val_size++;
		while (1) {
			assert(fscanf(file1, "%c", &c) == 1);
			if (c == ',') {
				assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
				val_size++;
			} else if (c == ']') {
				break;
			} else {
				assert(0);
			}
		}
		if(fscanf(file1, "%c", &c));
		assert(c=='\n');
		fclose(file1);
		while (x_size < 6 && fscanf(file2, "%lf,", &x[x_size]) == 1) {
            x_size++;
        }
        fclose(file2);
		sparse_matrix_t A;
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, 6, 6, indptr, indptr+1, indices, csr_val);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
		clock_gettime(CLOCK_MONOTONIC, &t1);
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
		clock_gettime(CLOCK_MONOTONIC, &t2);
		sparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		spmv_kernel(y, x, val, 0, 3, 0, 3, 0);
		clock_gettime(CLOCK_MONOTONIC, &t2);
		dense_block_times[0][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
	}
	printf("Sparse: ");
	for (int i=0; i<1; i++) {
		printf("%lu,", sparse_times[i]);
	}
	printf("\n");
	printf("Dense: ");
	for (int i=0; i<1; i++) {
		long total_dense = 0;
		for (int j=0; j<1; j++) {
			total_dense += dense_block_times[j][i];
		}
		printf("%lu,", total_dense);
	}
	printf("\n");
	for (int j=0; j<1; j++) {
		printf("Dense Block %d: ", j+1);
		for (int i=0; i<1; i++) {
			printf("%lu,", dense_block_times[j][i]);
		}
		printf("\n");
	}
	printf("\n");
	for (int i=0; i<6; i++) {
		printf("%lf\n", y[i]);
	}
	free(dense_block_times);
	free(y);
	free(x);
	free(val);
	free(csr_val);
	free(indptr);
	free(indices);
}
