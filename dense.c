#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

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
        printf("%d %d %lf\n", x, y, val);
        values[((x-1) * ((*cols) - 1)) + y] = val;
    }

    fclose(file);
    return values;
}

int main() {
    int rows, cols;
    double *vector, *result;

    // Replace "your_matrix_market_file.mtx" with the actual file name
    double *matrix = loadMatrixMarket("Generated_Matrices/Matrix_50_50_2000_1000000z.mtx", &rows, &cols);

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

    // Perform matrix-vector multiplication using BLAS
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, matrix, cols, vector, 1, 0.0, result, 1);

    // Print the result vector
    // printf("Result vector:\n");
    // for (int i = 0; i < rows; i++) {
    //     printf("%lf\n", result[i]);
    // }

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result);

    return 0;
}
