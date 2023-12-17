#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>

double* loadMatrixMarket(const char *filename, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening the file.\n");
        exit(1);
    }

    char line[1024];

    // Skip lines that begin with %
    while (fgets(line, sizeof(line), file) != NULL && line[0] == '%');

    // Read matrix size
    int nnz;
    sscanf(line, "%d %d %d\n", rows, cols, &nnz);
    // printf("Matrix size: %d x %d %d\n", *rows, *cols, nnz);

    double* values = (double *)calloc((*rows) * (*cols), sizeof(double));
    if (values == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    int x, y;
    double val;
    while (fscanf(file, "%d %d %lf\n", &x, &y, &val) == 3) {
        // printf("%d %d %lf\n", x, y, val);
        values[((x-1) * ((*cols) - 1)) + y] = val;
    }

    fclose(file);
    return values;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
        exit(1);
    }

    int rows, cols;
    double *vector, *result;

    // Replace "your_matrix_market_file.mtx" with the actual file name
    double *matrix = loadMatrixMarket(argv[1], &rows, &cols);

    // Create a vector (array of size number of columns of the matrix)
    vector = (double *)malloc(cols * sizeof(double));
    if (vector == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        free(matrix);
        exit(1);
    }

    // Initialize the vector (you should replace this with your own values)
    for (int i = 0; i < cols; i++) {
        vector[i] = 1.0; // or any other value
    }

    // Allocate memory for the result vector
    result = (double *)malloc(rows * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        free(matrix);
        free(vector);
        exit(1);
    }

    struct timeval t1;
    gettimeofday(&t1, NULL);
    long t1s = t1.tv_sec * 1000000L + t1.tv_usec;

    // Perform matrix-vector multiplication using BLAS
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, matrix, cols, vector, 1, 0.0, result, 1);

    struct timeval t2;
    gettimeofday(&t2, NULL);
    long t2s = t2.tv_sec * 1000000L + t2.tv_usec;

    printf("%s = %lu\n", argv[1], t2s-t1s);

    // Print the result vector
    for (int i = 0; i < rows; i++) {
        printf("%lf\n", argv[1], result[i]);
    }

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result);

    return 0;
}
