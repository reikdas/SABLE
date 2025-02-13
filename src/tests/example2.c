#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void spmv_kernel(float *y, const float* x, const float* val, int i_start, int i_end, int j_start, int j_end, int val_offset) {
	for (int j = j_start; j < j_end; j++) {
		for (int i = i_start; i < i_end; i++) {
			y[i] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[j]);
		}
	}
}

int main() {
	long times[5];
	FILE *file1 = fopen("/Users/amir/Documents/Pratyush/repo/2/SABLE/src/tests/example2.vbrc", "r");
	if (file1 == NULL) { printf("Error opening file1"); return 1; }
	FILE *file2 = fopen("/Users/amir/Documents/Pratyush/repo/2/SABLE/Generated_dense_tensors/generated_vector_11.vector", "r");
	if (file2 == NULL) { printf("Error opening file2"); return 1; }
	float* y = (float*)calloc(11, sizeof(float));
	float* x = (float*)calloc(12, sizeof(float));
	float* val = (float*)calloc(46, sizeof(float));
	char c;
	int x_size=0, val_size=0;
	assert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
    val_size++;
    while (1) {
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {
            assert(fscanf(file1, "%f", &val[val_size]) == 1.0);
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

    while (x_size < 11 && fscanf(file2, "%f,", &x[x_size]) == 1) {
        x_size++;
    }
    fclose(file2);
	struct timeval t1;

	struct timeval t2;
	for (int i=0; i<6; i++) {
		memset(y, 0, sizeof(float)*11);
		gettimeofday(&t1, NULL);
		long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
		spmv_kernel(y, x, val, 0, 2, 0, 2, 0);
		spmv_kernel(y, x, val, 0, 2, 5, 6, 4);

		y[0] += val[6] * x[10];		y[1] += val[7] * x[10];
		spmv_kernel(y, x, val, 2, 5, 2, 5, 8);

		y[2] += val[17] * x[5];		y[4] += val[18] * x[5];
		spmv_kernel(y, x, val, 5, 6, 0, 2, 19);
		spmv_kernel(y, x, val, 5, 6, 2, 5, 21);
		spmv_kernel(y, x, val, 5, 6, 5, 6, 24);
		spmv_kernel(y, x, val, 5, 6, 6, 9, 25);

		y[6] += val[28] * x[5];		y[7] += val[29] * x[5];
		spmv_kernel(y, x, val, 6, 9, 6, 9, 30);
		spmv_kernel(y, x, val, 9, 11, 0, 2, 39);

		y[9] += val[43] * x[10];		y[10] += val[44] * x[10];
		gettimeofday(&t2, NULL);
		long t2s = t2.tv_sec * 1000000L + t2.tv_usec;
		if (i!=0)
			times[i-1] = t2s-t1s;
	}
	printf("example2 = ");
	for (int i=0; i<5; i++) {
		printf("%lu,", times[i]);
	}
	printf("\n");
	for (int i=0; i<11; i++) {
		printf("%f\n", y[i]);
	}
}
