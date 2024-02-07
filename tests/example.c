#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>

int main() {
	FILE *file1 = fopen("/home/reikdas/VBR-SpMV/tests/example.vbr", "r");
	if (file1 == NULL) { printf("Error opening file1"); return 1; }
	FILE *file2 = fopen("/home/reikdas/VBR-SpMV/generated_matrix_11x11.matrix", "r");
	if (file2 == NULL) { printf("Error opening file2"); return 1; }
	float** y = (float**)calloc(11, sizeof(float*));
	for (int i=0; i<11; i++) {
		y[i] = (float*)calloc(11, sizeof(float));
	}
	float** x = (float**)calloc(11, sizeof(float*));
	for (int i=0; i<11; i++) {
		x[i] = (float*)calloc(11, sizeof(float));
	}
	float* val = (float*)calloc(52, sizeof(float));
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
    fclose(file1);	for(int i = 0; i < 11; i++) {
        for(int j = 0; j < 11; j++) {
            assert(fscanf(file2, "%f,", &x[i][j]) == 1);
        }
    }
    fclose(file2);	int count = 0;
	struct timeval t1;
	gettimeofday(&t1, NULL);
	long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
	for (int i=0; i<2; i++) {
		for (int j=0; j<11; j++) {
			for (int k=0; k<2; k++) {
				y[i][j] += val[0+ (k-0)*2 + (i-0)] * x[k][j];
			}
		}
	}
	for (int i=0; i<2; i++) {
		for (int j=0; j<11; j++) {
			for (int k=5; k<6; k++) {
				y[i][j] += val[4+ (k-5)*2 + (i-0)] * x[k][j];
			}
		}
	}
	for (int i=0; i<2; i++) {
		for (int j=0; j<11; j++) {
			for (int k=9; k<11; k++) {
				y[i][j] += val[6+ (k-9)*2 + (i-0)] * x[k][j];
			}
		}
	}
	for (int i=2; i<5; i++) {
		for (int j=0; j<11; j++) {
			for (int k=2; k<5; k++) {
				y[i][j] += val[10+ (k-2)*3 + (i-2)] * x[k][j];
			}
		}
	}
	for (int i=2; i<5; i++) {
		for (int j=0; j<11; j++) {
			for (int k=5; k<6; k++) {
				y[i][j] += val[19+ (k-5)*3 + (i-2)] * x[k][j];
			}
		}
	}
	for (int i=5; i<6; i++) {
		for (int j=0; j<11; j++) {
			for (int k=0; k<2; k++) {
				y[i][j] += val[22+ (k-0)*1 + (i-5)] * x[k][j];
			}
		}
	}
	for (int i=5; i<6; i++) {
		for (int j=0; j<11; j++) {
			for (int k=2; k<5; k++) {
				y[i][j] += val[24+ (k-2)*1 + (i-5)] * x[k][j];
			}
		}
	}
	for (int i=5; i<6; i++) {
		for (int j=0; j<11; j++) {
			for (int k=5; k<6; k++) {
				y[i][j] += val[27+ (k-5)*1 + (i-5)] * x[k][j];
			}
		}
	}
	for (int i=5; i<6; i++) {
		for (int j=0; j<11; j++) {
			for (int k=6; k<9; k++) {
				y[i][j] += val[28+ (k-6)*1 + (i-5)] * x[k][j];
			}
		}
	}
	for (int i=6; i<9; i++) {
		for (int j=0; j<11; j++) {
			for (int k=5; k<6; k++) {
				y[i][j] += val[31+ (k-5)*3 + (i-6)] * x[k][j];
			}
		}
	}
	for (int i=6; i<9; i++) {
		for (int j=0; j<11; j++) {
			for (int k=6; k<9; k++) {
				y[i][j] += val[34+ (k-6)*3 + (i-6)] * x[k][j];
			}
		}
	}
	for (int i=9; i<11; i++) {
		for (int j=0; j<11; j++) {
			for (int k=0; k<2; k++) {
				y[i][j] += val[43+ (k-0)*2 + (i-9)] * x[k][j];
			}
		}
	}
	for (int i=9; i<11; i++) {
		for (int j=0; j<11; j++) {
			for (int k=9; k<11; k++) {
				y[i][j] += val[47+ (k-9)*2 + (i-9)] * x[k][j];
			}
		}
	}
	struct timeval t2;
	gettimeofday(&t2, NULL);
	long t2s = t2.tv_sec * 1000000L + t2.tv_usec;
	printf("example = %lu\n", t2s-t1s);
	for (int i=0; i<11; i++) {
		for (int j=0; j<11; j++) {
			printf("%f\n", y[i][j]);
		}
	}
}
