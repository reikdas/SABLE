#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <omp.h>

struct COO {
    int rows, cols, nnz;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;

    // Sort by (row, column)
    void sort() {
        std::vector<int> indices(row_indices.size());
        for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return (row_indices[a] < row_indices[b]) || 
                   (row_indices[a] == row_indices[b] && col_indices[a] < col_indices[b]);
        });

        std::vector<int> sorted_row(row_indices.size()), sorted_col(col_indices.size());
        std::vector<float> sorted_vals(values.size());

        for (size_t i = 0; i < indices.size(); i++) {
            sorted_row[i] = row_indices[indices[i]];
            sorted_col[i] = col_indices[indices[i]];
            sorted_vals[i] = values[indices[i]];
        }

        row_indices = std::move(sorted_row);
        col_indices = std::move(sorted_col);
        values = std::move(sorted_vals);
    }
};

// Define CSR struct without using vectors
struct CSR {
    int rows, cols, nnz;
    int *row_ptrs;
    int *col_indices;
    float *values;
};

// add a function to convert COO to CSR
CSR coo_to_csr(const COO &coo) {
    CSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;

    csr.row_ptrs = new int[csr.rows + 1];
    csr.col_indices = new int[csr.nnz];
    csr.values = new float[csr.nnz];

    int row = 0;
    int row_start = 0;
    csr.row_ptrs[0] = 0;

    for (int i = 0; i < csr.nnz; i++) {
        while (coo.row_indices[i] != row) {
            row++;
            csr.row_ptrs[row] = row_start;
        }

        csr.col_indices[i] = coo.col_indices[i];
        csr.values[i] = coo.values[i];
        row_start++;
    }

    while (row < csr.rows) {
        row++;
        csr.row_ptrs[row] = row_start;
    }

    return csr;
}

// add enum to keep track of the matrix data type
enum class MatrixType {
    REAL,
    INTEGER
};

// add enum to keep track of the matrix format general, symmetric
enum class MatrixFormat {
    GENERAL,
    SYMMETRIC
};

COO readMTXtoCOO(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    MatrixFormat matrixFormat;
    MatrixType matrixType;

    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        // line 0 should contain MatrixMarket header, the matrix type and the matrix format
        if (i == 0) {
            if (line.find("MatrixMarket") != std::string::npos) {
                if (line.find("real") != std::string::npos) {
                    matrixType = MatrixType::REAL;
                } else if (line.find("integer") != std::string::npos) {
                    matrixType = MatrixType::INTEGER;
                }

                if (line.find("general") != std::string::npos) {
                    matrixFormat = MatrixFormat::GENERAL;
                } else if (line.find("symmetric") != std::string::npos) {
                    matrixFormat = MatrixFormat::SYMMETRIC;
                }
                i++;
            } else {
                std::cerr << "Error: Invalid MatrixMarket header\n";
                exit(1);
            }
            continue;
        }
        // Skip comments (lines starting with '%')
        else if (line[0] == '%') {
            // std::cout << line << std::endl;
            i++;
            continue;
        }
        break; // First non-comment line contains matrix size
    }

    std::istringstream iss(line);
    int M, N, nz;
    if (!(iss >> M >> N >> nz)) {
        std::cerr << "Error: Invalid matrix size format\n";
        exit(1);
    }

    if (M != N && matrixFormat == MatrixFormat::SYMMETRIC) {
        std::cerr << "Error: Symmetric matrix must be square\n";
        exit(1);
    }


    COO coo;
    coo.rows = M;
    coo.cols = N;

    if (matrixFormat == MatrixFormat::SYMMETRIC) {
        coo.row_indices.reserve(nz*2);
        coo.col_indices.reserve(nz*2);
        coo.values.reserve(nz*2);
        coo.nnz = nz*2;
    } else {
        coo.row_indices.reserve(nz);
        coo.col_indices.reserve(nz);
        coo.values.reserve(nz);
        coo.nnz = nz;
    }

    // Read matrix entries
    for (int i = 0; i < nz; i++) {
        int row, col;
        float value;

        if (!(file >> row >> col >> value)) {
            std::cerr << "Error: Invalid matrix entry format\n";
            exit(1);
        }

        coo.row_indices.push_back(row - 1); // Convert 1-based to 0-based
        coo.col_indices.push_back(col - 1);
        coo.values.push_back(value);

        if (matrixFormat == MatrixFormat::SYMMETRIC && row != col) {
            coo.row_indices.push_back(col - 1);
            coo.col_indices.push_back(row - 1);
            coo.values.push_back(value);
        }
    }

    // sort the COO matrix by row and col indices
    coo.sort();

    file.close();
    return coo;
}

// add a function to calculate the matrix-vector product
void spmv(const CSR &csr, const float *x, float *y) {

    // add pragma omp parallel for only if a flag is defined at compile time
    #ifdef OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < csr.rows; i++) {
        // y[i] = 0;
        float sum = 0;
        for (int j = csr.row_ptrs[i]; j < csr.row_ptrs[i + 1]; j++) {
            sum += csr.values[j] * x[csr.col_indices[j]];
        }
        y[i] = sum;
    }
}

// Example usage
int main(int argc, char *argv[]) {


    // std::string filename = "matrix.mtx";
    // read the filename from the command line
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        exit(1);
    }
    std::string filename = argv[1];
    int num_threads = std::stoi(argv[2]);

    // set the number of threads
    omp_set_num_threads(num_threads);

    COO cooMatrix = readMTXtoCOO(filename);

    // std::cout << "Matrix dimensions: " << cooMatrix.rows << "x" << cooMatrix.cols << std::endl;
    // std::cout << "Number of non-zeros: " << cooMatrix.row_indices.size() << std::endl;
    // std::cout << "Number of non-zeros: " << cooMatrix.nnz << std::endl;

    CSR csrMatrix = coo_to_csr(cooMatrix);

    // create an array with the same size as the number of cols in the csrMatrix
    float *x = new float[csrMatrix.cols];

    // initialize the array with 0-1 random values
    for (int i = 0; i < csrMatrix.cols; i++) {
        x[i] = (float) rand() / RAND_MAX;
    }

    // create an array with the same size as the number of rows in the csrMatrix
    float *y = new float[csrMatrix.rows];

    // calculate the matrix-vector product and time it
    float* exec_time = new float[32];
    spmv(csrMatrix, x, y);
    for (int i = 0; i < 32; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        spmv(csrMatrix, x, y);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        exec_time[i] = elapsed.count();
    }
    
    // get the median of the execution time
    std::sort(exec_time, exec_time + 32);
    std::cout << "Time: " << exec_time[16] * 1e6 << " us" << std::endl;
    

    // print time in us
    // std::cout << "Time: " << elapsed.count() * 1e6 << " us" << std::endl;


    return 0;
}
