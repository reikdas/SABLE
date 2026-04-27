#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


void spmm_kernel(
    double *restrict Y,
    const double *restrict X,
    const double *restrict val,
    const int i_start, const int i_end,
    const int j_start, const int j_end,
    const int val_offset) 
{
    const int block_i = 16;
    const int block_j = 16;

    for (int ii = i_start; ii < i_end; ii += block_i) {
        int i_max = (ii + block_i < i_end) ? (ii + block_i) : i_end;
        for (int jj = j_start; jj < j_end; jj += block_j) {
            int j_max = (jj + block_j < j_end) ? (jj + block_j) : j_end;

            for (int i = ii; i < i_max; i++) {
                for (int j = jj; j < j_max; j++) {
                    double a = (&val[val_offset])[
                        ((j - j_start) * (i_end - i_start)) + (i - i_start)
                    ];

                    double *y_row = &Y[i * 512];
                    const double *x_row = &X[j * 512];

                    for (int k = 0; k < 512; k++) {
                        y_row[k] += a * x_row[k];
                    }
                }
            }
        }
    }
}


void spmm_kernel_2(
    double *restrict Y,
    const double *restrict X,
    const double *restrict val,
    const int i_start,
    const int j_start, const int j_end,
    const int val_offset) 
{
    double *y_row = &Y[i_start * 512];
    const double *block_val = &val[val_offset];
    
    for (int j = j_start; j < j_end; j++) {
        double a = block_val[j - j_start];
        const double *x_row = &X[j * 512];
        
        for (int k = 0; k < 512; k++) {
            y_row[k] += a * x_row[k];
        }
    }
}


void spmm_kernel_3(
    double *restrict Y,
    const double *restrict X,
    const double *restrict val,
    const int i_start, const int i_end,
    const int j_start,
    const int val_offset) 
{
    const double *x_row = &X[j_start * 512];
    const double *block_val = &val[val_offset];
    
    for (int i = i_start; i < i_end; i++) {
        double a = block_val[i - i_start];
        double *y_row = &Y[i * 512];
        
        for (int k = 0; k < 512; k++) {
            y_row[k] += a * x_row[k];
        }
    }
}


void spmm_kernel_4(
    double *restrict Y,
    const double *restrict X,
    const double *restrict val,
    const int i_start, const int i_end,
    const int j_start, const int j_end,
    const int val_offset) 
{
    const int block_i = 16;
    
    for (int ii = i_start; ii < i_end; ii += block_i) {
        int i_max = (ii + block_i < i_end) ? (ii + block_i) : i_end;
        
        for (int j = j_start; j < j_end; j++) {
            for (int i = ii; i < i_max; i++) {
                double a = (&val[val_offset])[
                    ((j - j_start) * (i_end - i_start)) + (i - i_start)
                ];
                
                double *y_row = &Y[i * 512];
                const double *x_row = &X[j * 512];
                
                for (int k = 0; k < 512; k++) {
                    y_row[k] += a * x_row[k];
                }
            }
        }
    }
}


void spmm_kernel_5(
    double *restrict Y,
    const double *restrict X,
    const double *restrict val,
    const int i_start, const int i_end,
    const int j_start, const int j_end,
    const int val_offset) 
{
    const int block_j = 16;
    
    for (int jj = j_start; jj < j_end; jj += block_j) {
        int j_max = (jj + block_j < j_end) ? (jj + block_j) : j_end;
        
        for (int i = i_start; i < i_end; i++) {
            for (int j = jj; j < j_max; j++) {
                double a = (&val[val_offset])[
                    ((j - j_start) * (i_end - i_start)) + (i - i_start)
                ];
                
                double *y_row = &Y[i * 512];
                const double *x_row = &X[j * 512];
                
                for (int k = 0; k < 512; k++) {
                    y_row[k] += a * x_row[k];
                }
            }
        }
    }
}


void spmm_sparse(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int sparse_rows) {
    for (int i = 0; i < sparse_rows; i++) {
        for (int p = indptr[i]; p < indptr[i+1]; p++) {
            int col = indices[p];
            double val = csr_val[p];
            for (int j = 0; j < 512; ++j) {
                y[i * 512 + j] += val * x[col * 512 + j];
            }
        }
    }
}

int main() {
	double *y = (double*)malloc(3072 * sizeof(double));
	double *x = (double*)malloc(3072 * sizeof(double));
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
	FILE *file2 = fopen("<PATH>/generated_matrix_6x512.matrix", "r");
	if (file2 == NULL) { printf("Error opening file2"); return 1; }
		memset(y, 0, sizeof(double)*3072);
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
		while (x_size < 3072 && fscanf(file2, "%lf,", &x[x_size]) == 1) {
            x_size++;
        }
        fclose(file2);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		spmm_sparse(y, csr_val, indices, indptr, x, 6);
		clock_gettime(CLOCK_MONOTONIC, &t2);
		sparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		spmm_kernel_4(y, x, val, 0, 3, 0, 3, 0);
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
	for (int i=0; i<3072; i++) {
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
