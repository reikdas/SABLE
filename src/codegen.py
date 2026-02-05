import copy
import os
import pathlib
import time

from utils.fileio import read_vbrc

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")


def split_chunks(values, num_chunks):
    if len([v for v in values if v != 0]) < num_chunks:
        num_chunks = len([v for v in values if v!=0])

    # Create a list of (value, index) tuples, excluding zeros
    indexed_values = [(index, value) for index, value in enumerate(values) if value != 0]

    # Sort by value in descending order
    sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)

    # Initialize chunks
    chunks = [[] for _ in range(num_chunks)]
    chunk_sums = [0] * num_chunks

    # Distribute values
    for index, value in sorted_indexed_values:
        # Find the chunk with the smallest sum
        min_sum_index = chunk_sums.index(min(chunk_sums))
        chunks[min_sum_index].append(index)
        chunk_sums[min_sum_index] += value

    return chunks

# Thresholds for MKL vs naive dispatch for dense blocks
# Based on empirical analysis: MKL struggles with thin/vector-like blocks
MKL_MIN_DIM_THRESHOLD = 8       # Minimum dimension for MKL to be beneficial
MKL_MAX_ASPECT_RATIO = 100      # Maximum aspect ratio for MKL to be beneficial


def should_use_mkl_for_block(rows: int, cols: int) -> bool:
    """Determine whether to use MKL or naive kernel for a dense block.
    
    MKL (cblas_dgemm) is better for "chunky" blocks where both dimensions
    are reasonably sized. For thin/vector-like blocks (one dimension very small)
    or blocks with extreme aspect ratios, the naive handwritten kernel has
    less overhead and performs better.
    
    Args:
        rows: Number of rows in the dense block
        cols: Number of columns in the dense block
        
    Returns:
        True if MKL should be used, False if naive kernel should be used
    """
    min_dim = min(rows, cols)
    max_dim = max(rows, cols)
    
    # Use naive for thin blocks (vector-like)
    if min_dim < MKL_MIN_DIM_THRESHOLD:
        return False
    
    # Use naive for extreme aspect ratios
    aspect_ratio = max_dim / min_dim if min_dim > 0 else float('inf')
    if aspect_ratio > MKL_MAX_ASPECT_RATIO:
        return False
    
    # Use MKL for chunky blocks
    return True


def get_best_naive_spmm_kernel_call(num_rows: int, num_cols: int, i_start: int, i_end: int, 
                                     j_start: int, j_end: int, val_offset: int) -> str:
    """Select the best naive SpMM kernel based on block shape.
    
    Different kernels are optimized for different block shapes:
    - spmm_kernel_2: Single row block (num_rows == 1)
    - spmm_kernel_3: Single column block (num_cols == 1)
    - spmm_kernel_4: Few columns (num_cols < 16) - blocks only in rows
    - spmm_kernel_5: Few rows (num_rows < 16) - blocks only in columns
    - spmm_kernel: General case with 16x16 blocking
    
    Args:
        num_rows, num_cols: Block dimensions
        i_start, i_end: Row range in output matrix
        j_start, j_end: Column range in input matrix
        val_offset: Offset into val array
        
    Returns:
        C function call string for the best kernel
    """
    if num_rows == 1:
        # Single row block - use kernel optimized for accumulating into one Y row
        return f"spmm_kernel_2(y, x, val, {i_start}, {j_start}, {j_end}, {val_offset});"
    elif num_cols == 1:
        # Single column block - use kernel optimized for broadcasting one X row
        return f"spmm_kernel_3(y, x, val, {i_start}, {i_end}, {j_start}, {val_offset});"
    elif num_cols < 16:
        # Few columns - block only in rows, iterate through all columns
        return f"spmm_kernel_4(y, x, val, {i_start}, {i_end}, {j_start}, {j_end}, {val_offset});"
    elif num_rows < 16:
        # Few rows - block only in columns, iterate through all rows
        return f"spmm_kernel_5(y, x, val, {i_start}, {i_end}, {j_start}, {j_end}, {val_offset});"
    else:
        # General case - use blocked kernel (16x16 blocking)
        return f"spmm_kernel(y, x, val, {i_start}, {i_end}, {j_start}, {j_end}, {val_offset});"

def spmv_kernel():
    code = []
    code.append("void spmv_kernel(double *restrict y, const double *restrict x, const double *restrict val, const int i_start, const int i_end, const int j_start, const int j_end, const int val_offset) {\n")
    code.append("\tfor (int j = j_start; j < j_end; j++) {\n")
    code.append("\t\tfor (int i = i_start; i < i_end; i++) {\n")
    code.append("\t\t\ty[i] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n\n")
    return "".join(code)

def spmv_kernel_2():
    code = []
    code.append("void spmv_kernel_2(double *restrict y, const double *restrict x, const double *restrict val, const int i_start, const int j_start, const int j_end, const int val_offset) {\n")
    code.append("\tfor (int j = j_start; j < j_end; j++) {\n")
    code.append("\t\ty[i_start] += ((&val[val_offset])[(((j-j_start)))] * x[j]);\n")
    code.append("\t}\n")
    code.append("}\n\n")
    return "".join(code)

def spmv_kernel_3():
    code = []
    code.append("void spmv_kernel_3(double *restrict y, const double *restrict x, const double *restrict val, const int i_start, const int i_end, const int j_start, const int val_offset) {\n")
    code.append("\tdouble xj = x[j_start];\n")
    code.append("\tfor (int i = i_start; i < i_end; i++) {\n")
    code.append("\t\ty[i] += ((&val[val_offset])[(i-i_start)] * xj);\n")
    code.append("\t}\n")
    code.append("}\n\n")
    return "".join(code)

def spmv_kernel_blas():
    """SpMV kernel using MKL's cblas_dgemv for dense blocks.

    Dense block values are stored column-major: val[(j-j_start)*block_rows + (i-i_start)]
    With CblasColMajor + CblasNoTrans, this matches our storage layout directly.

    y[block_rows] += A[block_rows x block_cols] * x[block_cols]
    """
    code = """
void spmv_kernel_blas(double *restrict y, const double *restrict x, const double *restrict val,
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

"""
    return code

def spmv_sparse():
    code = []
    code.append("void spmv_sparse(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int rpntr_size) {\n")
    code.append("\tfor (int i = 0; i < rpntr_size; i++) {\n")
    code.append("\t\tint row_start = indptr[i];\n")
    code.append("\t\tint row_end = indptr[i + 1];\n")
    code.append("\t\tfor (int j = row_start; j < row_end; j++) {\n")
    code.append("\t\t\ty[i] += csr_val[j] * x[indices[j]];\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n\n")
    return "".join(code)

def spmv_sparse_naive():
    """Simple 3-loop CSR SpMV kernel similar to csr-spmv.cpp - used as baseline."""
    code = []
    code.append("void spmv_sparse_naive(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int rpntr_size) {\n")
    code.append("\tfor (int i = 0; i < rpntr_size; i++) {\n")
    code.append("\t\tdouble sum = 0.0;\n")
    code.append("\t\tfor (int j = indptr[i]; j < indptr[i + 1]; j++) {\n")
    code.append("\t\t\tsum += csr_val[j] * x[indices[j]];\n")
    code.append("\t\t}\n")
    code.append("\t\ty[i] += sum;\n")
    code.append("\t}\n")
    code.append("}\n\n")
    return "".join(code)


def spmv_sparse_naive_parallel():
    """OpenMP-parallelized CSR SpMV kernel."""
    code = []
    code.append("void spmv_sparse_naive_parallel(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int rpntr_size) {\n")
    code.append("\t#pragma omp parallel for schedule(dynamic, 64)\n")
    code.append("\tfor (int i = 0; i < rpntr_size; i++) {\n")
    code.append("\t\tdouble sum = 0.0;\n")
    code.append("\t\tfor (int j = indptr[i]; j < indptr[i + 1]; j++) {\n")
    code.append("\t\t\tsum += csr_val[j] * x[indices[j]];\n")
    code.append("\t\t}\n")
    code.append("\t\ty[i] += sum;\n")
    code.append("\t}\n")
    code.append("}\n\n")
    return "".join(code)

def spmm_kernel():
    """General SpMM kernel with 2D blocking for arbitrary block shapes."""
    code = """
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
"""
    return code

def spmm_kernel_2():
    """SpMM kernel optimized for single-row blocks (1 row, many columns).
    
    When there's only one output row, we can keep the Y row in cache
    and iterate through all X columns, accumulating into that single row.
    """
    code = """
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
"""
    return code

def spmm_kernel_3():
    """SpMM kernel optimized for single-column blocks (many rows, 1 column).
    
    When there's only one input column, we load the X row once and
    broadcast it to all output rows with different coefficients.
    """
    code = """
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
"""
    return code

def spmm_kernel_4():
    """SpMM kernel optimized for blocks with cols < 16 (many rows, few columns).
    
    When there are few columns, blocking in the column dimension doesn't help.
    We block only in the row dimension and iterate through all columns.
    This handles cases like 3606x3 efficiently.
    """
    code = """
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
"""
    return code

def spmm_kernel_5():
    """SpMM kernel optimized for blocks with rows < 16 (few rows, many columns).
    
    When there are few rows, blocking in the row dimension doesn't help.
    We block only in the column dimension and iterate through all rows.
    """
    code = """
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
"""
    return code

def spmm_sparse():
    code = """
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
"""
    return code


def spmm_sparse_parallel():
    """OpenMP-parallelized CSR SpMM kernel."""
    code = """
void spmm_sparse_parallel(double *restrict y, const double *restrict csr_val, const int *restrict indices, const int *restrict indptr, const double *restrict x, const int sparse_rows) {
    #pragma omp parallel for schedule(dynamic, 64)
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
"""
    return code


def spmm_kernel_mkl():
    """SpMM kernel using MKL's cblas_dgemm for dense blocks.

    Dense block values are stored column-major: val[(j-j_start)*block_rows + (i-i_start)]
    With CblasRowMajor + CblasTrans, we treat the column-major A as transposed row-major.

    Y[block_rows x 512] += A[block_rows x block_cols] * X[block_cols x 512]
    """
    code = """
void spmm_kernel_mkl(
    double *restrict Y,
    const double *restrict X,
    const double *restrict val,
    const int i_start, const int i_end,
    const int j_start, const int j_end,
    const int val_offset)
{
    const int block_rows = i_end - i_start;
    const int block_cols = j_end - j_start;

    // A is stored column-major (block_rows x block_cols)
    // X is row-major (K x 512), we use submatrix starting at row j_start
    // Y is row-major (M x 512), we update submatrix starting at row i_start
    //
    // With CblasRowMajor + CblasTrans:
    // - A is treated as (block_cols x block_rows) row-major, then transposed to (block_rows x block_cols)
    // - This matches our column-major storage
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                block_rows, 512, block_cols,     // M, N, K
                1.0,                              // alpha
                &val[val_offset], block_rows,     // A: column-major = row-major transposed, lda = block_rows
                &X[j_start * 512], 512,           // B: row-major (block_cols x 512), ldb = 512
                1.0,                              // beta
                &Y[i_start * 512], 512);          // C: row-major (block_rows x 512), ldc = 512
}
"""
    return code


def gen_single_threaded_spmm_naive_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with handwritten dense kernels + handwritten sparse kernel.

    Args:
        threads: Number of threads for OpenMP parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")

    # Use parallel kernel when threads > 1
    use_parallel = threads > 1
    sparse_kernel_name = "spmm_sparse_parallel" if use_parallel else "spmm_sparse"
    sparse_kernel_func = spmm_sparse_parallel() if use_parallel else spmm_sparse()

    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    if threads > 1:
        code.append("#include <omp.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append(spmm_kernel())
    code.append("\n")
    code.append(spmm_kernel_2())
    code.append("\n")
    code.append(spmm_kernel_3())
    code.append("\n")
    code.append(spmm_kernel_4())
    code.append("\n")
    code.append(spmm_kernel_5())
    code.append("\n")
    code.append(sparse_kernel_func)
    code.append("\n")
    code.append("int main() {\n")
    if threads > 1:
        code.append(f"\tomp_set_num_threads({threads});\n\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1] * 512} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1] * 512} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        if (len(indptr) > 0):
            code.append(f"\tint *indptr = (int*)malloc({len(indptr)} * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc({len(indices)} * sizeof(int));\n")
            code.append(f"\tif (!indptr || !indices) {{\n")
            code.append(f"\t\tprintf(\"Memory allocation failed for indptr/indices\\n\");\n")
            code.append(f"\t\treturn 1;\n")
            code.append(f"\t}}\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1] * 512))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(int));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(int));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
        code.append(f"""\t\tval_size=0;
        assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))
    
    if (len(ublocks) > 0):
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append(f"\t\t{sparse_kernel_name}(y, csr_val, indices, indptr, x, {rpntr[-1]});\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
    count = 0
    nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    # Dispatch to best naive kernel based on block shape
                    num_rows = rpntr[a+1] - rpntr[a]
                    num_cols = cpntr[b+1] - cpntr[b]
                    kernel_call = get_best_naive_spmm_kernel_call(
                        num_rows, num_cols,
                        rpntr[a], rpntr[a+1], cpntr[b], cpntr[b+1], indx[count]
                    )
                    code.append(f"\t\t{kernel_call}\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")
    
    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]*512}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")
    
    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1


def gen_single_threaded_spmm_mkl_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with MKL cblas_dgemm dense kernel + handwritten sparse kernel.

    Uses MKL BLAS for dense blocks instead of hand-written kernels.
    This should provide significantly better performance for larger blocks
    while potentially having overhead for very small blocks.

    Args:
        threads: Number of threads for MKL parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")

    # Use parallel kernel when threads > 1
    use_parallel = threads > 1
    sparse_kernel_name = "spmm_sparse_parallel" if use_parallel else "spmm_sparse"
    sparse_kernel_func = spmm_sparse_parallel() if use_parallel else spmm_sparse()

    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    code.append("#include <mkl.h>\n")
    code.append("#include <mkl_cblas.h>\n")
    if threads > 1:
        code.append("#include <omp.h>\n")
    code.append("\n")
    code.append(spmm_kernel_mkl())
    code.append("\n")
    # Include all naive kernel variants for thin/vector-like blocks
    code.append(spmm_kernel())    # General 16x16 blocked kernel
    code.append("\n")
    code.append(spmm_kernel_2())  # Single row (num_rows == 1)
    code.append("\n")
    code.append(spmm_kernel_3())  # Single column (num_cols == 1)
    code.append("\n")
    code.append(spmm_kernel_4())  # Few columns (num_cols < 16)
    code.append("\n")
    code.append(spmm_kernel_5())  # Few rows (num_rows < 16)
    code.append("\n")
    code.append(sparse_kernel_func)
    code.append("\n")
    code.append("int main() {\n")
    # Set thread counts
    if threads > 1:
        code.append(f"\tomp_set_num_threads({threads});\n")
    code.append(f"\tmkl_set_num_threads({threads});\n\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1] * 512} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1] * 512} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        if (len(indptr) > 0):
            code.append(f"\tint *indptr = (int*)malloc({len(indptr)} * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc({len(indices)} * sizeof(int));\n")
            code.append(f"\tif (!indptr || !indices) {{\n")
            code.append(f"\t\tprintf(\"Memory allocation failed for indptr/indices\\n\");\n")
            code.append(f"\t\treturn 1;\n")
            code.append(f"\t}}\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1] * 512))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(int));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(int));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
        code.append(f"""\t\tval_size=0;
        assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))

    if (len(ublocks) > 0):
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append(f"\t\t{sparse_kernel_name}(y, csr_val, indices, indptr, x, {rpntr[-1]});\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
    count = 0
    nnz_block = 0
    mkl_blocks = 0
    naive_blocks = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    block_rows = rpntr[a+1] - rpntr[a]
                    block_cols = cpntr[b+1] - cpntr[b]
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    # Dispatch to MKL or best naive kernel based on block shape
                    if should_use_mkl_for_block(block_rows, block_cols):
                        code.append(f"\t\tspmm_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        mkl_blocks += 1
                    else:
                        # Use the best naive kernel for this block shape
                        kernel_call = get_best_naive_spmm_kernel_call(
                            block_rows, block_cols,
                            rpntr[a], rpntr[a+1], cpntr[b], cpntr[b+1], indx[count]
                        )
                        code.append(f"\t\t{kernel_call}\n")
                        naive_blocks += 1
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]*512}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    # Log dispatch statistics
    print(f"  [mkl_naive] Block dispatch: {mkl_blocks} MKL, {naive_blocks} naive (of {mkl_blocks + naive_blocks} total dense blocks)")
    return time2-time1


def _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_kernel_include, sparse_kernel_function, sparse_kernel_call, dense_first=False, threads=1, use_mkl=False):
    """Common code generation for single-threaded SpMV functions.

    Args:
        sparse_kernel_include: String with additional includes (e.g., '#include "utility.h"\\n' or '#include <mkl.h>\\n')
        sparse_kernel_function: String with additional kernel function definitions (e.g., spmv_sparse_naive function)
        sparse_kernel_call: Function that generates the sparse kernel call code, takes (code, ublocks, csr_val, rpntr, cpntr, indptr, indices) as args
        dense_first: If True, execute dense kernel before sparse kernel. If False, execute sparse kernel first.
        threads: Number of threads to use for parallelization (default: 1)
        use_mkl: If True, set MKL thread count as well (default: False)
    """
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    if threads > 1:
        code.append("#include <omp.h>\n")
    if sparse_kernel_include:
        code.append(sparse_kernel_include)
    code.append("#include <assert.h>\n\n")
    code.append(spmv_kernel())
    code.append("\n")
    code.append(spmv_kernel_2())
    code.append("\n")
    code.append(spmv_kernel_3())
    code.append("\n")
    if sparse_kernel_function:
        code.append(sparse_kernel_function)
        code.append("\n")
    code.append("int main() {\n")
    # Set up thread counts if using parallelization
    if threads > 1:
        code.append(f"\tomp_set_num_threads({threads});\n")
    if use_mkl:
        code.append(f"\tmkl_set_num_threads({threads});\n")
    if threads > 1 or use_mkl:
        code.append("\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1]} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1]} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if (len(ublocks) > 0):
        if (len(indptr) > 0):
            code.append(f"\tint *indptr = (int*)malloc({len(indptr)} * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc({len(indices)} * sizeof(int));\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1]))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(int));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(int));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
    if (len(indptr) > 0):
        code.append(f"""\t\tval_size=0;
        assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1]))

    # Helper to generate dense block kernel calls
    def gen_dense_blocks():
        count = 0
        nnz_block = 0
        for a in range(len(rpntr)-1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if nnz_block not in ublocks:
                        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                        if (rpntr[a+1] - rpntr[a]) == 1:
                            code.append(f"\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        elif (cpntr[b+1] - cpntr[b]) == 1:
                            code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                        else:
                            code.append(f"\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                        code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                        count+=1
                    nnz_block += 1
        return count

    # Dispatch kernels in the appropriate order
    if dense_first:
        # Dense kernel first, then sparse kernel
        count = gen_dense_blocks()
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
    else:
        # Sparse kernel first, then dense kernel (original behavior)
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
        count = gen_dense_blocks()

    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")    

    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    return code

def _gen_single_threaded_spmv_common_blas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_kernel_include, sparse_kernel_function, sparse_kernel_call, dense_first=False, threads=1):
    """Common code generation for single-threaded SpMV functions with BLAS dense kernels.

    Uses cblas_dgemv for dense blocks, but keeps spmv_kernel_3 for single-column (Nx1) blocks
    since GCC-vectorized code is slightly faster for those.

    Args:
        sparse_kernel_include: String with additional includes (e.g., '#include "utility.h"\\n' or '#include <mkl.h>\\n')
        sparse_kernel_function: String with additional kernel function definitions (e.g., spmv_sparse_naive function)
        sparse_kernel_call: Function that generates the sparse kernel call code, takes (code, ublocks, csr_val, rpntr, cpntr, indptr, indices) as args
        dense_first: If True, execute dense kernel before sparse kernel. If False, execute sparse kernel first.
        threads: Number of threads for MKL parallelization (default: 1)
    """
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <mkl.h>\n")  # Always include MKL for cblas_dgemv
    if threads > 1:
        code.append("#include <omp.h>\n")
    if sparse_kernel_include:
        code.append(sparse_kernel_include)
    code.append("#include <assert.h>\n\n")

    # Include all dense kernels for threshold-based dispatch
    code.append(spmv_kernel())      # General naive kernel
    code.append(spmv_kernel_2())    # 1xM (single row) kernel
    code.append(spmv_kernel_3())    # Nx1 (single column) kernel
    code.append("\n")
    code.append(spmv_kernel_blas())
    code.append("\n")
    if sparse_kernel_function:
        code.append(sparse_kernel_function)
        code.append("\n")
    code.append("int main() {\n")
    # Set up thread counts
    if threads > 1:
        code.append(f"\tomp_set_num_threads({threads});\n")
    code.append(f"\tmkl_set_num_threads({threads});\n")  # MKL threading
    code.append("\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1]} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1]} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if (len(ublocks) > 0):
        if (len(indptr) > 0):
            code.append(f"\tint *indptr = (int*)malloc({len(indptr)} * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc({len(indices)} * sizeof(int));\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1]))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(int));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(int));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
    if (len(indptr) > 0):
        code.append(f"""\t\tval_size=0;
        assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1]))

    # Helper to generate dense block kernel calls with BLAS dispatch
    def gen_dense_blocks_blas():
        count = 0
        nnz_block = 0
        for a in range(len(rpntr)-1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if nnz_block not in ublocks:
                        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                        # Threshold-based dispatch: use BLAS only for large, reasonably-shaped blocks
                        # Based on benchmarks: handwritten kernels are faster for small-to-medium
                        # square blocks (up to ~100x100), while BLAS wins for larger blocks.
                        # CRITICAL: Skinny blocks (e.g., 2xN or Nx2) should NEVER use BLAS -
                        # BLAS has too much call overhead for these, handwritten kernels are 2-3x faster.
                        rows = rpntr[a+1] - rpntr[a]
                        cols = cpntr[b+1] - cpntr[b]
                        BLAS_AREA_THRESHOLD = 10000  # Use BLAS for blocks with area >= 10000 (~100x100)
                        MIN_DIM_FOR_BLAS = 16  # Minimum dimension to consider BLAS (avoids skinny blocks)
                        if cols == 1:
                            # Nx1 (single column) - use kernel_3
                            code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                        elif rows == 1:
                            # 1xM (single row) - use kernel_2 (dot product)
                            code.append(f"\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        elif rows * cols < BLAS_AREA_THRESHOLD or min(rows, cols) < MIN_DIM_FOR_BLAS:
                            # Small/medium block OR skinny block - use general naive kernel (faster due to lower overhead)
                            code.append(f"\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        else:
                            # Large block with reasonable aspect ratio - use BLAS dgemv
                            code.append(f"\t\tspmv_kernel_blas(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                        code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                        count+=1
                    nnz_block += 1
        return count

    # Dispatch kernels in the appropriate order
    if dense_first:
        # Dense kernel first, then sparse kernel
        count = gen_dense_blocks_blas()
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
    else:
        # Sparse kernel first, then dense kernel (original behavior)
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
        count = gen_dense_blocks_blas()

    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    return code

def gen_single_threaded_spmv_naive_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using naive dense kernels + SpV8 sparse kernel.
    
    When sparse dispatch is spv8, dense dispatch comes before sparse dispatch.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # When sparse dispatch is spv8, dense dispatch comes before sparse dispatch
    dense_first = True

    def sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
        if len(ublocks) > 0:
            code.append(f"\t\tstruct csr_matrix mat = input_matrix({len(csr_val)}, {rpntr[-1]}, {cpntr[-1]}, csr_val, indices, indptr);\n")
            code.append("\t\tstruct tr_matrix tr = process(&mat);\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
            code.append("\t\tspmv_tr_spvv8_kernel(&tr, x, y);\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
            code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

    sparse_include = '#include "utility.h"\n' if len(ublocks) > 0 else ''
    # SpV8 kernel has internal OpenMP parallelization, just pass threads for omp_set_num_threads
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, dense_first=dense_first, threads=threads)

    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_naive_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using naive dense kernels + naive 3-loop CSR SpMV kernel for sparse part.

    This function can handle both split matrices (with dense blocks) and fully sparse matrices.
    Dense blocks are handled using the standard dense kernels (spmv_kernel, spmv_kernel_2, spmv_kernel_3).

    When sparse dispatch is naive, sparse dispatch comes before dense dispatch.

    Args:
        threads: Number of threads to use for parallelization (default: 1)
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # Use parallel kernel when threads > 1
    use_parallel = threads > 1
    kernel_name = "spmv_sparse_naive_parallel" if use_parallel else "spmv_sparse_naive"
    kernel_func = spmv_sparse_naive_parallel() if use_parallel else spmv_sparse_naive()

    def sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
        if len(ublocks) > 0:
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
            code.append(f"\t\t{kernel_name}(y, csr_val, indices, indptr, x, {rpntr[-1]});\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
            code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

    # When sparse dispatch is naive, sparse dispatch comes before dense dispatch
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, '', kernel_func, sparse_kernel_call, dense_first=False, threads=threads)

    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_naive_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using naive dense kernels + MKL sparse kernel.

    When sparse dispatch is mkl, sparse dispatch comes before dense dispatch.

    Args:
        threads: Number of threads to use for MKL parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # MKL sparse kernel benefits from running first (clean cache), unlike SpV8.
    # Dense-first was found to hurt MKL sparse performance due to cache pollution.
    dense_first = False
    beta = "1.0" if dense_first else "0.0"

    def sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
        if len(ublocks) > 0:
            # Note: mkl_set_num_threads is set once in main() via _gen_single_threaded_spmv_common
            code.append(f"""\t\tsparse_matrix_t A;
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;\n""")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
            code.append(f"\t\tmkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, {beta}, y);\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
            code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

    sparse_include = ''
    if len(ublocks) > 0:
        sparse_include = "#include <mkl.h>\n#include <mkl_spblas.h>\n"
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, dense_first=dense_first, threads=threads, use_mkl=True)

    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_naive_uzp(
    val,
    indx,
    bindx,
    rpntr,
    cpntr,
    bpntrb,
    bpntre,
    ublocks,
    indptr,
    indices,
    csr_val,
    dir_name,
    filename,
    vbr_dir,
    bench: int = 5,
    sparse_mtx_path: str = "",
    threads: int = 1,
) -> int:
    """Generate code using UZP for the sparse (CSR) part.

    UZP preparation (z_polyhedrator + spf_aggregator) runs ONCE via a shell
    script outside the timing loop.  Only the UZP kernel execution is timed.

    Args:
        sparse_mtx_path: Absolute path to the .mtx file containing just the
            sparse remainder of this matrix.  Generated offline by
            ``generate_sparse_mtx.py``.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")

    uzp_prepare_script = os.path.join(BASE_PATH, "uzp_prepare.sh")

    code: list[str] = []
    code.append("#include <assert.h>\n")
    code.append("#include <stdio.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <unistd.h>\n\n")

    # UZP headers (compile needs -I uzp-genex)
    code.append("#include <spf_structure.h>\n")
    code.append("#include <spf_executors.h>\n\n")

    # Dense kernels (same as other backends)
    code.append(spmv_kernel())
    code.append("\n")
    code.append(spmv_kernel_2())
    code.append("\n")
    code.append(spmv_kernel_3())
    code.append("\n")

    code.append("int main() {\n")
    code.append(f"\tconst int nrows = {rpntr[-1]};\n")
    code.append(f"\tconst int ncols = {cpntr[-1]};\n")
    code.append(f"\tconst int bench = {bench};\n\n")

    code.append("\t// Allocate vectors\n")
    code.append("\tdouble *y = (double*)malloc((size_t)nrows * sizeof(double));\n")
    code.append("\tdouble *x = (double*)malloc((size_t)ncols * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append("\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    code.append("\tassert(y && x && val);\n\n")

    code.append("\t// Read dense block values from VBRC (outside timing loop)\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening vbrc file\\n\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening vector file\\n\"); return 1; }\n")
    code.append("\tmemset(x, 0, (size_t)ncols * sizeof(double));\n")
    code.append("\tmemset(val, 0, (size_t)({0}) * sizeof(double));\n".format(len(val) if len(val) > 0 else 1))
    code.append("\tchar c;\n")
    code.append("\tint x_size = 0, val_size = 0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%c", &c) == 1);
\tif (c != ']') {
\t\tungetc(c, file1);
\t\tassert(fscanf(file1, "%lf", &val[val_size]) == 1);
\t\tval_size++;
\t\twhile (1) {
\t\t\tassert(fscanf(file1, "%c", &c) == 1);
\t\t\tif (c == ',') {
\t\t\t\tassert(fscanf(file1, "%lf", &val[val_size]) == 1);
\t\t\t\tval_size++;
\t\t\t} else if (c == ']') {
\t\t\t\tbreak;
\t\t\t} else {
\t\t\t\tassert(0);
\t\t\t}
\t\t}
\t}
\tassert(fscanf(file1, "%c", &c) == 1 && c == '\\n');
''')
    code.append("\tfclose(file1);\n")
    code.append('''\twhile (x_size < ncols && fscanf(file2, "%lf,", &x[x_size]) == 1) {
\t\tx_size++;
\t}
\tfclose(file2);
''')
    code.append("\n")

    # Run UZP preparation shell script ONCE (outside timing loop)
    if len(ublocks) > 0:
        mtx_abs = os.path.abspath(sparse_mtx_path)
        mtx_basename = os.path.splitext(os.path.basename(sparse_mtx_path))[0]
        uzp_tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_UZP_tmp', filename))
        code.append("\t// Prepare UZP files from offline-generated .mtx (outside timing loop)\n")
        code.append(f"\tchar uzp_dir[] = \"{uzp_tmp_dir}\";\n")
        code.append("\tchar cmd[2048];\n")
        code.append(f"\tsnprintf(cmd, sizeof(cmd), \"\\\"{os.path.abspath(uzp_prepare_script)}\\\" \\\"{mtx_abs}\\\" \\\"%s\\\"\", uzp_dir);\n")
        code.append("\tint rc = system(cmd);\n")
        code.append("\tassert(rc == 0);\n")
        code.append("\tchar uzp_path[600];\n")
        code.append(f"\tsnprintf(uzp_path, sizeof(uzp_path), \"%s/{mtx_basename}.tuned.uzp\", uzp_dir);\n")
        code.append("\ts_spf_structure_t* spf_mat = spf_matrix_read_from_file(uzp_path);\n")
        code.append("\tassert(spf_mat);\n\n")

    # Timing state
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")

    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n\n")

    code.append("\t// Benchmark loop: only UZP kernel execution is timed for sparse part\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tmemset(y, 0, (size_t)nrows * sizeof(double));\n")

    if len(ublocks) > 0:
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append("\t\tspf_executors_spf_matrix_dense_vector_product(spf_mat, x, y, ncols, nrows, 0);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);\n")

    count = 0
    nnz_block = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    if (rpntr[a + 1] - rpntr[a]) == 1:
                        code.append(f"\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    elif (cpntr[b + 1] - cpntr[b]) == 1:
                        code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                    else:
                        code.append(f"\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);\n")
                    count += 1
                nnz_block += 1
    code.append("\t}\n\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print result vector
    code.append("\tprintf(\"\\n\");\n")
    code.append("\tfor (int i=0; i<nrows; i++) {\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    # Cleanup
    code.append("\tfree(dense_block_times);\n")
    code.append("\tfree(y);\n")
    code.append("\tfree(x);\n")
    code.append("\tfree(val);\n")
    code.append("\treturn 0;\n")
    code.append("}\n")

    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)

    time2 = time.time_ns() // 1_000_000
    return time2 - time1

def gen_single_threaded_spmv_blas_uzp(
    val,
    indx,
    bindx,
    rpntr,
    cpntr,
    bpntrb,
    bpntre,
    ublocks,
    indptr,
    indices,
    csr_val,
    dir_name,
    filename,
    vbr_dir,
    bench: int = 5,
    sparse_mtx_path: str = "",
    threads: int = 1,
) -> int:
    """UZP sparse dispatch + BLAS dense blocks (cblas_dgemv).

    UZP preparation (z_polyhedrator + spf_aggregator) runs ONCE via a shell
    script outside the timing loop.  Only the UZP kernel execution is timed.
    Dense blocks use BLAS (``cblas_dgemv``) except Nx1 blocks which keep
    ``spmv_kernel_3``.

    Args:
        sparse_mtx_path: Absolute path to the .mtx file containing just the
            sparse remainder of this matrix.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")

    uzp_prepare_script = os.path.join(BASE_PATH, "uzp_prepare.sh")

    code: list[str] = []
    code.append("#include <assert.h>\n")
    code.append("#include <stdio.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <unistd.h>\n")
    code.append("#include <mkl.h>\n\n")

    code.append("#include <spf_structure.h>\n")
    code.append("#include <spf_executors.h>\n\n")

    code.append(spmv_kernel_3())
    code.append("\n")
    code.append(spmv_kernel_blas())
    code.append("\n")

    code.append("int main() {\n")
    code.append(f"\tconst int nrows = {rpntr[-1]};\n")
    code.append(f"\tconst int ncols = {cpntr[-1]};\n")
    code.append(f"\tconst int bench = {bench};\n\n")

    code.append("\t// Allocate vectors\n")
    code.append("\tdouble *y = (double*)malloc((size_t)nrows * sizeof(double));\n")
    code.append("\tdouble *x = (double*)malloc((size_t)ncols * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append("\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    code.append("\tassert(y && x && val);\n\n")

    code.append("\t// Read dense block values from VBRC (outside timing loop)\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening vbrc file\\n\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening vector file\\n\"); return 1; }\n")
    code.append("\tmemset(x, 0, (size_t)ncols * sizeof(double));\n")
    code.append("\tmemset(val, 0, (size_t)({0}) * sizeof(double));\n".format(len(val) if len(val) > 0 else 1))
    code.append("\tchar c;\n")
    code.append("\tint x_size = 0, val_size = 0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%c", &c) == 1);
\tif (c != ']') {
\t\tungetc(c, file1);
\t\tassert(fscanf(file1, "%lf", &val[val_size]) == 1);
\t\tval_size++;
\t\twhile (1) {
\t\t\tassert(fscanf(file1, "%c", &c) == 1);
\t\t\tif (c == ',') {
\t\t\t\tassert(fscanf(file1, "%lf", &val[val_size]) == 1);
\t\t\t\tval_size++;
\t\t\t} else if (c == ']') {
\t\t\t\tbreak;
\t\t\t} else {
\t\t\t\tassert(0);
\t\t\t}
\t\t}
\t}
\tassert(fscanf(file1, "%c", &c) == 1 && c == '\\n');
''')
    code.append("\tfclose(file1);\n")
    code.append('''\twhile (x_size < ncols && fscanf(file2, "%lf,", &x[x_size]) == 1) {
\t\tx_size++;
\t}
\tfclose(file2);
''')
    code.append("\n")

    # Run UZP preparation shell script ONCE (outside timing loop)
    if len(ublocks) > 0:
        mtx_abs = os.path.abspath(sparse_mtx_path)
        mtx_basename = os.path.splitext(os.path.basename(sparse_mtx_path))[0]
        uzp_tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_UZP_tmp', filename))
        code.append("\t// Prepare UZP files from offline-generated .mtx (outside timing loop)\n")
        code.append(f"\tchar uzp_dir[] = \"{uzp_tmp_dir}\";\n")
        code.append("\tchar cmd[2048];\n")
        code.append(f"\tsnprintf(cmd, sizeof(cmd), \"\\\"{os.path.abspath(uzp_prepare_script)}\\\" \\\"{mtx_abs}\\\" \\\"%s\\\"\", uzp_dir);\n")
        code.append("\tint rc = system(cmd);\n")
        code.append("\tassert(rc == 0);\n")
        code.append("\tchar uzp_path[600];\n")
        code.append(f"\tsnprintf(uzp_path, sizeof(uzp_path), \"%s/{mtx_basename}.tuned.uzp\", uzp_dir);\n")
        code.append("\ts_spf_structure_t* spf_mat = spf_matrix_read_from_file(uzp_path);\n")
        code.append("\tassert(spf_mat);\n\n")

    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")

    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n\n")

    code.append("\tfor (int i=0; i<bench; i++) {\n")
    code.append("\t\tmemset(y, 0, (size_t)nrows * sizeof(double));\n")

    if len(ublocks) > 0:
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append("\t\tspf_executors_spf_matrix_dense_vector_product(spf_mat, x, y, ncols, nrows, 0);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);\n")

    count = 0
    nnz_block = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    if (cpntr[b + 1] - cpntr[b]) == 1:
                        code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                    else:
                        code.append(f"\t\tspmv_kernel_blas(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);\n")
                    count += 1
                nnz_block += 1
    code.append("\t}\n\n")

    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    code.append("\tprintf(\"\\n\");\n")
    code.append("\tfor (int i=0; i<nrows; i++) {\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    code.append("\tfree(dense_block_times);\n")
    code.append("\tfree(y);\n")
    code.append("\tfree(x);\n")
    code.append("\tfree(val);\n")
    code.append("\treturn 0;\n")
    code.append("}\n")

    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)

    time2 = time.time_ns() // 1_000_000
    return time2 - time1

def gen_single_threaded_spmv_blas_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using BLAS dense kernels + SpV8 sparse kernel.

    When sparse dispatch is spv8, dense dispatch comes before sparse dispatch.

    Args:
        threads: Number of threads for parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # When sparse dispatch is spv8, dense dispatch comes before sparse dispatch
    dense_first = True

    def sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
        if len(ublocks) > 0:
            code.append(f"\t\tstruct csr_matrix mat = input_matrix({len(csr_val)}, {rpntr[-1]}, {cpntr[-1]}, csr_val, indices, indptr);\n")
            code.append("\t\tstruct tr_matrix tr = process(&mat);\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
            code.append("\t\tspmv_tr_spvv8_kernel(&tr, x, y);\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
            code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

    sparse_include = '#include "utility.h"\n' if len(ublocks) > 0 else ''
    code = _gen_single_threaded_spmv_common_blas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, dense_first=dense_first, threads=threads)

    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_blas_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using BLAS dense kernels + naive 3-loop CSR sparse kernel.

    When sparse dispatch is naive, sparse dispatch comes before dense dispatch.

    Args:
        threads: Number of threads for parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # Use parallel kernel when threads > 1
    use_parallel = threads > 1
    kernel_name = "spmv_sparse_naive_parallel" if use_parallel else "spmv_sparse_naive"
    kernel_func = spmv_sparse_naive_parallel() if use_parallel else spmv_sparse_naive()

    def sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
        if len(ublocks) > 0:
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
            code.append(f"\t\t{kernel_name}(y, csr_val, indices, indptr, x, {rpntr[-1]});\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
            code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

    # When sparse dispatch is naive, sparse dispatch comes before dense dispatch
    code = _gen_single_threaded_spmv_common_blas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, '', kernel_func, sparse_kernel_call, dense_first=False, threads=threads)

    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_blas_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using BLAS dense kernels + MKL sparse kernel.

    When sparse dispatch is mkl, sparse dispatch comes before dense dispatch.

    Args:
        threads: Number of threads for MKL parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # MKL sparse kernel benefits from running first (clean cache), unlike SpV8.
    # Dense-first was found to hurt MKL sparse performance due to cache pollution.
    dense_first = False
    beta = "1.0" if dense_first else "0.0"

    def sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
        if len(ublocks) > 0:
            # Note: mkl_set_num_threads is set once in main() via _gen_single_threaded_spmv_common_blas
            code.append(f"""\t\tsparse_matrix_t A;
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;\n""")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
            code.append(f"\t\tmkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, {beta}, y);\n")
            code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
            code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

    # MKL sparse functions need mkl_spblas.h (mkl.h is already included by _gen_single_threaded_spmv_common_blas)
    sparse_include = ''
    if len(ublocks) > 0:
        sparse_include = "#include <mkl_spblas.h>\n"
    code = _gen_single_threaded_spmv_common_blas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, dense_first=dense_first, threads=threads)

    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name: str, filename: str, vbr_dir: str, bench:int=5) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n")
        f.write("#include <string.h>\n")
        f.write("#include <mkl.h>\n")
        f.write("#include <mkl_spblas.h>\n")
        f.write("#include <omp.h>\n\n")
        # Allocate memory dynamically instead of on stack
        f.write(f"double *y = (double*)malloc({rpntr[-1]} * sizeof(double));\n")
        f.write(f"double *x = (double*)malloc({cpntr[-1]} * sizeof(double));\n")
        if len(val) > 0:
            f.write(f"double *val = (double*)malloc({len(val)} * sizeof(double));\n")
        else:
            f.write(f"double *val = (double*)malloc(1 * sizeof(double));\n")
        if len(ublocks) > 0:
            f.write(f"double *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
        
        # Check if allocation succeeded
        f.write(f"if (!y || !x || !val || (csr_val && !csr_val)) {{\n")
        f.write(f"\tprintf(\"Memory allocation failed\\n\");\n")
        f.write(f"\treturn 1;\n")
        f.write(f"}}\n")
        
        # Initialize arrays
        f.write(f"memset(y, 0, {rpntr[-1]} * sizeof(double));\n")
        f.write(f"memset(x, 0, {cpntr[-1]} * sizeof(double));\n")
        f.write(f"memset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
        if len(ublocks) > 0:
            f.write(f"memset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        f.write("\n")
        f.write(spmv_kernel())
        f.write(spmv_kernel_2())
        f.write(spmv_kernel_3())
        f.write("\n")
        work_per_br = [0]*(len(rpntr)-1)
        count = 0
        for a in range(len(rpntr) - 1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if count not in ublocks:
                        work_per_br[a] += (rpntr[a+1] - rpntr[a])*(cpntr[b+1] - cpntr[b])
                    count += 1
        count2 = 0
        thread_br_map = split_chunks(work_per_br, threads)
        funcount = 0
        num_working_threads = len(thread_br_map)
        f.write("\n")
        f.write("int main() {\n")
        f.write(f"\tlong times[{bench}];\n")
        f.write(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
        f.write("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
        f.write(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
        f.write("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
        f.write("\tchar c;\n")
        f.write(f"\tint x_size=0, val_size=0;\n")
        f.write('''\tassert(fscanf(file1, "val=[%c", &c) == 1);
    if (c != ']') {
        ungetc(c, file1);
        assert(fscanf(file1, "%lf", &val[val_size]) == 1);
        val_size++;
        while (1) {
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {
                assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                val_size++;
            } else if (c == ']') {
                break;
            } else {
                assert(0);
            }
        }
    }
    assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
        if (len(ublocks) > 0):
            f.write('''\tval_size=0;
    assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
    if(fscanf(file1, "%c", &c));
    assert(c=='\\n');\n''')
        if (len(indptr) > 0):
            f.write(f"""\tint indptr[{len(indptr)}] = {{0}};
    int indices[{len(indices)}] = {{0}};
    val_size=0;
    assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
    val_size++;
    while (1) {{
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
            val_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    if(fscanf(file1, "%c", &c));
    assert(c=='\\n');
    val_size=0;
    assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
    val_size++;
    while (1) {{
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
            val_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    if(fscanf(file1, "%c", &c));
    assert(c=='\\n');\n""")
        f.write("\tfclose(file1);\n")
        f.write('''\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
        if len(ublocks) > 0:
            f.write(f"""\tsparse_matrix_t A;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_set_num_threads({threads});\n""")
        f.write("\t#pragma omp parallel\n")
        f.write("\t{\n")
        f.write(f"\tomp_set_num_threads({threads});\n")
        f.write("\t}\n")
        f.write("\tstruct timespec t1;\n")
        f.write("\tstruct timespec t2;\n")
        f.write(f"\tfor (int i=0; i<{bench+1}; i++) {{\n")
        f.write("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1]))
        f.write("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        if (len(ublocks) > 0):
            f.write("\t\tmkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);\n")
        if len(thread_br_map) > 0:
            f.write("\t\t#pragma omp parallel sections\n")
            f.write("\t\t{\n")
        for br_list in thread_br_map:
            f.write("\t\t#pragma omp section\n")
            f.write("\t\t{\n")
            for a in br_list:
                if bpntrb[a] == -1:
                    continue
                ublocks_count = copy.copy(bpntrb[a])
                valid_cols = bindx[bpntrb[a]:bpntre[a]]
                count = 0
                # find num_ublocks before this block
                idx_offset = 0
                for ub in ublocks:
                    if ub < bpntrb[a]:
                        idx_offset += 1
                    if ub > bpntrb[a]:
                        break
                indx_start = bpntrb[a] - idx_offset
                for b in range(len(cpntr)-1):
                    if b in valid_cols:
                        if ublocks_count not in ublocks:
                            if (rpntr[a+1] - rpntr[a]) == 1:
                                f.write(f"\t\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                            elif (cpntr[b+1] - cpntr[b]) == 1:
                                f.write(f"\t\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                            else:
                                f.write(f"\t\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[indx_start+count]});\n")
                            count += 1
                        ublocks_count += 1
            f.write("}\n")
            funcount += 1
        if len(thread_br_map) > 0:
            f.write("\t\t}\n")
        f.write("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        f.write("\t\tif (i!=0)\n")
        f.write("\t\t\ttimes[i-1] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
        f.write("\t}\n")
        f.write('\tprintf("{0} = ");\n'.format(filename))
        f.write("\tfor (int i=0; i<{0}; i++) {{\n".format(bench))
        f.write("\t\tprintf(\"%lu,\", times[i]);\n")
        f.write("\t}\n")
        f.write("\tprintf(\"\\n\");\n")
        f.write(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
        f.write("\t\tprintf(\"%.2f\\n\", y[i]);\n")
        f.write("\t}\n")
        
        # Free allocated memory
        f.write(f"\tfree(y);\n")
        f.write(f"\tfree(x);\n")
        f.write(f"\tfree(val);\n")
        if len(ublocks) > 0:
            f.write(f"\tfree(csr_val);\n")
        
        f.write("}\n")

def vbr_spmv_codegen(filename: str, dir_name: str, vbr_dir: str, threads: int, mkl: bool, bench: int = 5)->int:
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbr_path)
    time1 = time.time_ns() // 1_000_000
    if mkl:
        # gen_single_threaded_spmv_dgemv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
        gen_single_threaded_spmv_naive_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, threads)
    else:
        # Use single-threaded codegen with threads parameter - SpV8 kernel has internal OpenMP support
        gen_single_threaded_spmv_naive_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, threads)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmm_naive_spreg(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with handwritten dense kernels + sparse-register-tiling sparse kernel.

    Uses separate init/execute/cleanup pattern for spreg:
    - spmm_spreg_init: Called once before timing loop (inspection + packing)
    - spmm_spreg_execute: Called inside timing loop (actual SpMM)
    - spmm_spreg_cleanup: Called after timing loop

    Args:
        threads: Number of threads for sparse-register-tiling parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    # When compiling with C++ compiler, restrict needs to be defined
    code.append("#ifdef __cplusplus\n")
    code.append("#ifndef restrict\n")
    code.append("#define restrict __restrict__\n")
    code.append("#endif\n")
    code.append("#endif\n")
    # Include sparse-register-tiling wrapper
    code.append('#include "spmm_spreg_wrapper.h"\n')
    code.append("\n")
    code.append(spmm_kernel())
    code.append("\n")
    code.append(spmm_kernel_2())
    code.append("\n")
    code.append(spmm_kernel_3())
    code.append("\n")
    code.append(spmm_kernel_4())
    code.append("\n")
    code.append(spmm_kernel_5())
    code.append("\n")
    code.append("int main() {\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1] * 512} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1] * 512} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        if (len(indptr) > 0):
            code.append(f"\tint *indptr = (int*)malloc({len(indptr)} * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc({len(indices)} * sizeof(int));\n")
            code.append(f"\tif (!indptr || !indices) {{\n")
            code.append(f"\t\tprintf(\"Memory allocation failed for indptr/indices\\n\");\n")
            code.append(f"\t\treturn 1;\n")
            code.append(f"\t}}\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    
    # === INITIALIZE SPARSE-REGISTER-TILING EXECUTOR (once, needs initial data load) ===
    # For spreg, we need to load CSR data once to initialize the executor
    # Then in the benchmark loop, we'll reload data for consistency with naive version
    if (len(ublocks) > 0):
        code.append(f"\n\t// Initial data load for spreg executor initialization\n")
        code.append(f"\tFILE *init_file = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
        code.append("\tif (init_file == NULL) { printf(\"Error opening init_file\"); return 1; }\n")
        code.append(f"\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\tmemset(indptr, 0, {len(indptr)} * sizeof(int));\n")
        code.append(f"\tmemset(indices, 0, {len(indices)} * sizeof(int));\n")
        code.append("\tchar init_c;\n")
        code.append("\tint init_val_size=0;\n")
        # Skip val line
        code.append('''\tassert(fscanf(init_file, "val=[%c", &init_c) == 1);
        while (init_c != '\\n') { assert(fscanf(init_file, "%c", &init_c) == 1); }\n''')
        # Read csr_val
        code.append('''\tinit_val_size=0;
        assert(fscanf(init_file, "csr_val=[%lf", &csr_val[init_val_size]) == 1.0);
        init_val_size++;
        while (1) {
            assert(fscanf(init_file, "%c", &init_c) == 1);
            if (init_c == ',') {
                assert(fscanf(init_file, "%lf", &csr_val[init_val_size]) == 1.0);
                init_val_size++;
            } else if (init_c == ']') {
                break;
            } else {
                assert(0);
            }
        }
        if(fscanf(init_file, "%c", &init_c));
        assert(init_c=='\\n');\n''')
        # Read indptr and indices
        code.append(f"""\tinit_val_size=0;
        assert(fscanf(init_file, "indptr=[%d", &indptr[init_val_size]) == 1.0);
        init_val_size++;
        while (1) {{
            assert(fscanf(init_file, "%c", &init_c) == 1);
            if (init_c == ',') {{
                assert(fscanf(init_file, "%d", &indptr[init_val_size]) == 1.0);
                init_val_size++;
            }} else if (init_c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(init_file, "%c", &init_c));
        assert(init_c=='\\n');
        init_val_size=0;
        assert(fscanf(init_file, "indices=[%d", &indices[init_val_size]) == 1.0);
        init_val_size++;
        while (1) {{
            assert(fscanf(init_file, "%c", &init_c) == 1);
            if (init_c == ',') {{
                assert(fscanf(init_file, "%d", &indices[init_val_size]) == 1.0);
                init_val_size++;
            }} else if (init_c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(init_file, "%c", &init_c));
        assert(init_c=='\\n');\n""")
        code.append("\tfclose(init_file);\n")
        code.append(f"\n\t// Initialize sparse-register-tiling executor (not timed)\n")
        code.append(f"\tvoid *spreg_handle = spmm_spreg_init(csr_val, indices, indptr, {rpntr[-1]}, {cpntr[-1]}, 512, {threads});\n")
        code.append("\tif (spreg_handle == NULL) {\n")
        code.append("\t\tprintf(\"Failed to initialize sparse-register-tiling executor\\n\");\n")
        code.append("\t\treturn 1;\n")
        code.append("\t}\n")

    # === BENCHMARK LOOP - load data each iteration (outside timing), like SpMV ===
    code.append(f"\n\t// Benchmark loop - load data each iteration (outside timing)\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1] * 512))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(int));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(int));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
        code.append(f"""\t\tval_size=0;
        assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))
    
    if (len(ublocks) > 0):
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        # Use sparse-register-tiling execute (only execution, no init)
        code.append(f"\t\tspmm_spreg_execute(spreg_handle, y, x);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
    count = 0
    nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    # Dispatch to best naive kernel based on block shape
                    num_rows = rpntr[a+1] - rpntr[a]
                    num_cols = cpntr[b+1] - cpntr[b]
                    kernel_call = get_best_naive_spmm_kernel_call(
                        num_rows, num_cols,
                        rpntr[a], rpntr[a+1], cpntr[b], cpntr[b+1], indx[count]
                    )
                    code.append(f"\t\t{kernel_call}\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")
    
    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]*512}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")
    
    # Cleanup sparse-register-tiling executor
    if len(ublocks) > 0:
        code.append("\n\t// Cleanup sparse-register-tiling executor\n")
        code.append("\tspmm_spreg_cleanup(spreg_handle);\n")
    
    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")
    
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1


def gen_single_threaded_spmm_mkl_spreg(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with MKL cblas_dgemm dense kernel + sparse-register-tiling sparse kernel.

    Uses MKL BLAS for dense blocks and sparse-register-tiling for sparse part.
    Uses separate init/execute/cleanup pattern for spreg:
    - spmm_spreg_init: Called once before timing loop (inspection + packing)
    - spmm_spreg_execute: Called inside timing loop (actual SpMM)
    - spmm_spreg_cleanup: Called after timing loop

    Args:
        threads: Number of threads for parallelization (MKL + spreg) (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    code.append("#include <mkl.h>\n")
    code.append("#include <mkl_cblas.h>\n")
    # When compiling with C++ compiler, restrict needs to be defined
    code.append("#ifdef __cplusplus\n")
    code.append("#ifndef restrict\n")
    code.append("#define restrict __restrict__\n")
    code.append("#endif\n")
    code.append("#endif\n")
    # Include sparse-register-tiling wrapper
    code.append('#include "spmm_spreg_wrapper.h"\n')
    code.append("\n")
    code.append(spmm_kernel_mkl())
    code.append("\n")
    # Include all naive kernel variants for thin/vector-like blocks
    code.append(spmm_kernel())    # General 16x16 blocked kernel
    code.append("\n")
    code.append(spmm_kernel_2())  # Single row (num_rows == 1)
    code.append("\n")
    code.append(spmm_kernel_3())  # Single column (num_cols == 1)
    code.append("\n")
    code.append(spmm_kernel_4())  # Few columns (num_cols < 16)
    code.append("\n")
    code.append(spmm_kernel_5())  # Few rows (num_rows < 16)
    code.append("\n")
    code.append("int main() {\n")
    # Set thread counts
    code.append(f"\tmkl_set_num_threads({threads});\n\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1] * 512} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1] * 512} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        # When ublocks > 0, we need CSR data for spreg. Validate that we have it.
        # If csr_val is empty but ublocks > 0, this is an error - we can't use spreg without CSR data.
        if len(csr_val) == 0 or len(indptr) == 0 or len(indices) == 0:
            code.append(f'\tprintf("Error: ublocks > 0 but CSR data is empty (csr_val=%d, indptr=%d, indices=%d)\\n", {len(csr_val)}, {len(indptr)}, {len(indices)});\n')
            code.append(f'\treturn 1;\n')
        # Allocate arrays with their actual sizes (they should all be > 0 at this point)
        if len(csr_val) == 0:
            code.append(f"\tdouble *csr_val = (double*)malloc(1 * sizeof(double));\n")
        # Always allocate indptr and indices when ublocks > 0 (needed for spreg init)
        if len(indptr) > 0:
            code.append(f"\tint *indptr = (int*)malloc({len(indptr)} * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc({len(indices)} * sizeof(int));\n")
        else:
            code.append(f"\tint *indptr = (int*)malloc(1 * sizeof(int));\n")
            code.append(f"\tint *indices = (int*)malloc(1 * sizeof(int));\n")
        code.append(f"\tif (!indptr || !indices) {{\n")
        code.append(f"\t\tprintf(\"Memory allocation failed for indptr/indices\\n\");\n")
        code.append(f"\t\treturn 1;\n")
        code.append(f"\t}}\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")

    # === INITIALIZE SPARSE-REGISTER-TILING EXECUTOR (once, needs initial data load) ===
    if (len(ublocks) > 0):
        code.append(f"\n\t// Initial data load for spreg executor initialization\n")
        code.append(f"\tFILE *init_file = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
        code.append("\tif (init_file == NULL) { printf(\"Error opening init_file\"); return 1; }\n")
        code.append(f"\tmemset(csr_val, 0, {len(csr_val) if len(csr_val) > 0 else 1} * sizeof(double));\n")
        code.append(f"\tmemset(indptr, 0, {len(indptr) if len(indptr) > 0 else 1} * sizeof(int));\n")
        code.append(f"\tmemset(indices, 0, {len(indices) if len(indices) > 0 else 1} * sizeof(int));\n")

        code.append("\tchar init_c;\n")
        code.append("\tint init_val_size=0;\n")
        # Skip val line
        code.append('''\tassert(fscanf(init_file, "val=[%c", &init_c) == 1);
        while (init_c != '\\n') { assert(fscanf(init_file, "%c", &init_c) == 1); }\n''')
        # Read csr_val with bounds checking
        code.append(f'''\tinit_val_size=0;
        int csr_val_max = {len(csr_val) if len(csr_val) > 0 else 1};
        assert(fscanf(init_file, "csr_val=[%lf", &csr_val[init_val_size]) == 1.0);
        init_val_size++;
        while (1) {{
            assert(fscanf(init_file, "%c", &init_c) == 1);
            if (init_c == ',') {{
                if (init_val_size >= csr_val_max) {{
                    printf("Error: csr_val buffer overflow! Read %d elements but allocated %d\\n", init_val_size, csr_val_max);
                    return 1;
                }}
                assert(fscanf(init_file, "%lf", &csr_val[init_val_size]) == 1.0);
                init_val_size++;
            }} else if (init_c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(init_file, "%c", &init_c));
        assert(init_c=='\\n');\n''')
        # Read indptr and indices with bounds checking
        code.append(f"""\tinit_val_size=0;
        int indptr_max = {len(indptr) if len(indptr) > 0 else 1};
        assert(fscanf(init_file, "indptr=[%d", &indptr[init_val_size]) == 1.0);
        init_val_size++;
        while (1) {{
            assert(fscanf(init_file, "%c", &init_c) == 1);
            if (init_c == ',') {{
                if (init_val_size >= indptr_max) {{
                    printf("Error: indptr buffer overflow! Read %d elements but allocated %d\\n", init_val_size, indptr_max);
                    return 1;
                }}
                assert(fscanf(init_file, "%d", &indptr[init_val_size]) == 1.0);
                init_val_size++;
            }} else if (init_c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(init_file, "%c", &init_c));
        assert(init_c=='\\n');
        init_val_size=0;
        int indices_max = {len(indices) if len(indices) > 0 else 1};
        assert(fscanf(init_file, "indices=[%d", &indices[init_val_size]) == 1.0);
        init_val_size++;
        while (1) {{
            assert(fscanf(init_file, "%c", &init_c) == 1);
            if (init_c == ',') {{
                if (init_val_size >= indices_max) {{
                    printf("Error: indices buffer overflow! Read %d elements but allocated %d\\n", init_val_size, indices_max);
                    return 1;
                }}
                assert(fscanf(init_file, "%d", &indices[init_val_size]) == 1.0);
                init_val_size++;
            }} else if (init_c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(init_file, "%c", &init_c));
        assert(init_c=='\\n');\n""")
        code.append("\tfclose(init_file);\n")
        code.append(f"\n\t// Initialize sparse-register-tiling executor (not timed)\n")
        code.append(f"\tvoid *spreg_handle = spmm_spreg_init(csr_val, indices, indptr, {rpntr[-1]}, {cpntr[-1]}, 512, {threads});\n")
        code.append("\tif (spreg_handle == NULL) {\n")
        code.append("\t\tprintf(\"Failed to initialize sparse-register-tiling executor\\n\");\n")
        code.append("\t\treturn 1;\n")
        code.append("\t}\n")

    code.append(f"\n\t// Benchmark loop - load data each iteration (outside timing)\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1] * 512))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        # Only memset indptr and indices if they're allocated (when ublocks > 0)
        code.append(f"\t\tmemset(indptr, 0, {len(indptr) if len(indptr) > 0 else 1} * sizeof(int));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices) if len(indices) > 0 else 1} * sizeof(int));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        code.append(f'''\t\tval_size=0;
        int csr_val_max_loop = {len(csr_val) if len(csr_val) > 0 else 1};
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                if (val_size >= csr_val_max_loop) {{
                    printf("Error: csr_val buffer overflow in loop iteration %d! Read %d elements but allocated %d\\n", i, val_size, csr_val_max_loop);
                    return 1;
                }}
                assert(fscanf(file1, "%lf", &csr_val[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
        code.append(f"""\t\tval_size=0;
        int indptr_max_loop = {len(indptr) if len(indptr) > 0 else 1};
        assert(fscanf(file1, "indptr=[%d", &indptr[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                if (val_size >= indptr_max_loop) {{
                    printf("Error: indptr buffer overflow in loop iteration %d! Read %d elements but allocated %d\\n", i, val_size, indptr_max_loop);
                    return 1;
                }}
                assert(fscanf(file1, "%d", &indptr[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        int indices_max_loop = {len(indices) if len(indices) > 0 else 1};
        assert(fscanf(file1, "indices=[%d", &indices[val_size]) == 1.0);
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                if (val_size >= indices_max_loop) {{
                    printf("Error: indices buffer overflow in loop iteration %d! Read %d elements but allocated %d\\n", i, val_size, indices_max_loop);
                    return 1;
                }}
                assert(fscanf(file1, "%d", &indices[val_size]) == 1.0);
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))
    if (len(ublocks) > 0):
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        # Use sparse-register-tiling execute (only execution, no init)
        code.append(f"\t\tspmm_spreg_execute(spreg_handle, y, x);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
    count = 0
    nnz_block = 0
    mkl_blocks = 0
    naive_blocks = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    block_rows = rpntr[a+1] - rpntr[a]
                    block_cols = cpntr[b+1] - cpntr[b]
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    # Dispatch to MKL or best naive kernel based on block shape
                    if should_use_mkl_for_block(block_rows, block_cols):
                        code.append(f"\t\tspmm_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        mkl_blocks += 1
                    else:
                        # Use the best naive kernel for this block shape
                        kernel_call = get_best_naive_spmm_kernel_call(
                            block_rows, block_cols,
                            rpntr[a], rpntr[a+1], cpntr[b], cpntr[b+1], indx[count]
                        )
                        code.append(f"\t\t{kernel_call}\n")
                        naive_blocks += 1
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]*512}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    # Cleanup sparse-register-tiling executor
    if len(ublocks) > 0:
        code.append("\n\t// Cleanup sparse-register-tiling executor\n")
        code.append("\tspmm_spreg_cleanup(spreg_handle);\n")

    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    # Log dispatch statistics
    print(f"  [mkl_spreg] Block dispatch: {mkl_blocks} MKL, {naive_blocks} naive (of {mkl_blocks + naive_blocks} total dense blocks)")
    return time2-time1


def gen_single_threaded_spmm_naive_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with handwritten dense kernels + MKL sparse kernel (mkl_sparse_d_mm).

    Uses handwritten kernels for dense blocks and MKL's mkl_sparse_d_mm for the sparse CSR part.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    # MKL headers for sparse mm
    if len(ublocks) > 0:
        code.append("#include <mkl.h>\n")
        code.append("#include <mkl_spblas.h>\n")
    code.append("\n")
    code.append(spmm_kernel())
    code.append("\n")
    code.append(spmm_kernel_2())
    code.append("\n")
    code.append(spmm_kernel_3())
    code.append("\n")
    code.append(spmm_kernel_4())
    code.append("\n")
    code.append(spmm_kernel_5())
    code.append("\n")
    code.append("int main() {\n")
    # Set MKL to single-threaded
    if len(ublocks) > 0:
        code.append("\tmkl_set_num_threads(1);\n\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1] * 512} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1] * 512} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        if (len(indptr) > 0):
            code.append(f"\tMKL_INT *indptr = (MKL_INT*)malloc({len(indptr)} * sizeof(MKL_INT));\n")
            code.append(f"\tMKL_INT *indices = (MKL_INT*)malloc({len(indices)} * sizeof(MKL_INT));\n")
            code.append(f"\tif (!indptr || !indices) {{\n")
            code.append(f"\t\tprintf(\"Memory allocation failed for indptr/indices\\n\");\n")
            code.append(f"\t\treturn 1;\n")
            code.append(f"\t}}\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1] * 512))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(MKL_INT));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(MKL_INT));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        # Use MKL_INT for indptr and indices
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
        code.append(f"""\t\tval_size=0;
        {{ int tmp; assert(fscanf(file1, "indptr=[%d", &tmp) == 1); indptr[val_size] = tmp; }}
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                int tmp; assert(fscanf(file1, "%d", &tmp) == 1); indptr[val_size] = tmp;
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        {{ int tmp; assert(fscanf(file1, "indices=[%d", &tmp) == 1); indices[val_size] = tmp; }}
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                int tmp; assert(fscanf(file1, "%d", &tmp) == 1); indices[val_size] = tmp;
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))

    if (len(ublocks) > 0):
        # Use MKL sparse mm for the sparse part
        code.append(f"""\t\tsparse_matrix_t A;
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;\n""")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append(f"\t\tmkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, SPARSE_LAYOUT_ROW_MAJOR, x, 512, 512, 0.0, y, 512);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
        code.append("\t\tmkl_sparse_destroy(A);\n")
    count = 0
    nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    # Dispatch to best naive kernel based on block shape
                    num_rows = rpntr[a+1] - rpntr[a]
                    num_cols = cpntr[b+1] - cpntr[b]
                    kernel_call = get_best_naive_spmm_kernel_call(
                        num_rows, num_cols,
                        rpntr[a], rpntr[a+1], cpntr[b], cpntr[b+1], indx[count]
                    )
                    code.append(f"\t\t{kernel_call}\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]*512}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1


def gen_single_threaded_spmm_mkl_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with MKL cblas_dgemm dense kernel + MKL sparse kernel (mkl_sparse_d_mm).

    Uses MKL BLAS for dense blocks and MKL's mkl_sparse_d_mm for the sparse CSR part.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    code.append("#include <mkl.h>\n")
    code.append("#include <mkl_cblas.h>\n")
    code.append("#include <mkl_spblas.h>\n\n")
    code.append(spmm_kernel_mkl())
    code.append("\n")
    # Include all naive kernel variants for thin/vector-like blocks
    code.append(spmm_kernel())    # General 16x16 blocked kernel
    code.append("\n")
    code.append(spmm_kernel_2())  # Single row (num_rows == 1)
    code.append("\n")
    code.append(spmm_kernel_3())  # Single column (num_cols == 1)
    code.append("\n")
    code.append(spmm_kernel_4())  # Few columns (num_cols < 16)
    code.append("\n")
    code.append(spmm_kernel_5())  # Few rows (num_rows < 16)
    code.append("\n")
    code.append("int main() {\n")
    # Set thread counts
    code.append(f"\tmkl_set_num_threads({threads});\n\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1] * 512} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1] * 512} * sizeof(double));\n")
    if len(val) > 0:
        code.append(f"\tdouble *val = (double*)malloc({len(val)} * sizeof(double));\n")
    else:
        code.append(f"\tdouble *val = (double*)malloc(1 * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({len(csr_val)} * sizeof(double));\n")
    if len(ublocks) > 0:
        if (len(indptr) > 0):
            code.append(f"\tMKL_INT *indptr = (MKL_INT*)malloc({len(indptr)} * sizeof(MKL_INT));\n")
            code.append(f"\tMKL_INT *indices = (MKL_INT*)malloc({len(indices)} * sizeof(MKL_INT));\n")
            code.append(f"\tif (!indptr || !indices) {{\n")
            code.append(f"\t\tprintf(\"Memory allocation failed for indptr/indices\\n\");\n")
            code.append(f"\t\treturn 1;\n")
            code.append(f"\t}}\n")
    code.append("\tstruct timespec t1, t2;\n")
    code.append(f"\tlong sparse_times[{bench}];\n")
    prev_count = 0
    prev_nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if prev_nnz_block not in ublocks:
                    prev_count += 1
                prev_nnz_block += 1
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({prev_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1] * 512))
    code.append(f"\t\tmemset(val, 0, {len(val) if len(val) > 0 else 1} * sizeof(double));\n")
    if len(csr_val) > 0:
        code.append(f"\t\tmemset(csr_val, 0, {len(csr_val)} * sizeof(double));\n")
        code.append(f"\t\tmemset(indptr, 0, {len(indptr)} * sizeof(MKL_INT));\n")
        code.append(f"\t\tmemset(indices, 0, {len(indices)} * sizeof(MKL_INT));\n")
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    code.append('''\t\tassert(fscanf(file1, "val=[%c", &c) == 1);
        if (c != ']') {
            ungetc(c, file1);
            assert(fscanf(file1, "%lf", &val[val_size]) == 1);
            val_size++;
            while (1) {
                assert(fscanf(file1, "%c", &c) == 1);
                if (c == ',') {
                    assert(fscanf(file1, "%lf", &val[val_size]) == 1);
                    val_size++;
                } else if (c == ']') {
                    break;
                } else {
                    assert(0);
                }
            }
        }
        assert(fscanf(file1, "%c", &c) == 1 && c == '\\n');\n''')
    if (len(ublocks) > 0):
        # Use MKL_INT for indptr and indices
        code.append('''\t\tval_size=0;
        assert(fscanf(file1, "csr_val=[%lf", &csr_val[val_size]) == 1.0);
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
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n''')
        code.append(f"""\t\tval_size=0;
        {{ int tmp; assert(fscanf(file1, "indptr=[%d", &tmp) == 1); indptr[val_size] = tmp; }}
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                int tmp; assert(fscanf(file1, "%d", &tmp) == 1); indptr[val_size] = tmp;
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');
        val_size=0;
        {{ int tmp; assert(fscanf(file1, "indices=[%d", &tmp) == 1); indices[val_size] = tmp; }}
        val_size++;
        while (1) {{
            assert(fscanf(file1, "%c", &c) == 1);
            if (c == ',') {{
                int tmp; assert(fscanf(file1, "%d", &tmp) == 1); indices[val_size] = tmp;
                val_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        if(fscanf(file1, "%c", &c));
        assert(c=='\\n');\n""")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))

    if (len(ublocks) > 0):
        # Use MKL sparse mm for the sparse part
        code.append(f"""\t\tsparse_matrix_t A;
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;\n""")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append(f"\t\tmkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, SPARSE_LAYOUT_ROW_MAJOR, x, 512, 512, 0.0, y, 512);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
        code.append("\t\tmkl_sparse_destroy(A);\n")
    count = 0
    nnz_block = 0
    mkl_blocks = 0
    naive_blocks = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    block_rows = rpntr[a+1] - rpntr[a]
                    block_cols = cpntr[b+1] - cpntr[b]
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    # Dispatch to MKL or best naive kernel based on block shape
                    if should_use_mkl_for_block(block_rows, block_cols):
                        code.append(f"\t\tspmm_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        mkl_blocks += 1
                    else:
                        # Use the best naive kernel for this block shape
                        kernel_call = get_best_naive_spmm_kernel_call(
                            block_rows, block_cols,
                            rpntr[a], rpntr[a+1], cpntr[b], cpntr[b+1], indx[count]
                        )
                        code.append(f"\t\t{kernel_call}\n")
                        naive_blocks += 1
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    # Print sparse timings
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print dense timings
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")

    # Print individual dense block timings
    code.append(f"\tfor (int j=0; j<{count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

    # Print original filename output
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]*512}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

    # Free allocated memory
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if len(csr_val) > 0:
        code.append(f"\tfree(csr_val);\n")
    if len(indptr) > 0:
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    # Log dispatch statistics
    print(f"  [mkl_mkl] Block dispatch: {mkl_blocks} MKL, {naive_blocks} naive (of {mkl_blocks + naive_blocks} total dense blocks)")
    return time2-time1


def vbr_spmm_codegen(filename: str, dir_name: str, vbr_dir: str, threads: int, bench: int = 5, mkl: bool = False)->int:
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbr_path)
    time1 = time.time_ns() // 1_000_000
    # Use single-threaded codegen with threads parameter - it has internal OpenMP support
    gen_single_threaded_spmm_naive_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, threads)
    time2 = time.time_ns() // 1_000_000
    return time2-time1