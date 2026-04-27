import copy
import os
import pathlib
import time
from functools import partial

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

def _write_generated_code(dir_name: str, filename: str, code: list[str]) -> None:
    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)

def _count_dense_blocks(
    rpntr: list[int],
    cpntr: list[int],
    bpntrb: list[int],
    bpntre: list[int],
    bindx: list[int],
    ublocks: list[int],
) -> int:
    count = 0
    nnz_block = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr) - 1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    count += 1
                nnz_block += 1
    return count

def _append_timing_workspace(code: list[str], bench: int, dense_block_count: int) -> None:
    code.append(f"\tlong sparse_times[{bench}];\n")
    code.append(f"\tlong (*dense_block_times)[{bench}] = (long(*)[{bench}])malloc({dense_block_count} * {bench} * sizeof(long));\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{dense_block_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")

def _append_dense_array_allocations(code: list[str], y_size: int, x_size: int, val_size: int) -> None:
    code.append(f"\tdouble *y = (double*)malloc({y_size} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({x_size} * sizeof(double));\n")
    code.append(f"\tdouble *val = (double*)malloc({val_size if val_size > 0 else 1} * sizeof(double));\n")

def _append_result_output(code: list[str], output_size: int) -> None:
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{output_size}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")

def _append_free_allocated_memory(code: list[str], include_csr: bool) -> None:
    code.append(f"\tfree(dense_block_times);\n")
    code.append(f"\tfree(y);\n")
    code.append(f"\tfree(x);\n")
    code.append(f"\tfree(val);\n")
    if include_csr:
        code.append(f"\tfree(csr_val);\n")
        code.append(f"\tfree(indptr);\n")
        code.append(f"\tfree(indices);\n")

def _append_csr_array_allocations(
    code: list[str],
    csr_val_size: int,
    indptr_size: int,
    indices_size: int,
) -> None:
    if csr_val_size > 0:
        code.append(f"\tdouble *csr_val = (double*)malloc({csr_val_size} * sizeof(double));\n")
        code.append(f"\tint *indptr = (int*)malloc({indptr_size} * sizeof(int));\n")
        code.append(f"\tint *indices = (int*)malloc({indices_size} * sizeof(int));\n")
    else:
        code.append("\tdouble *csr_val = (double*)malloc(1 * sizeof(double));\n")
        code.append(f"\tint *indptr = (int*)malloc(1 * sizeof(int));\n")
        code.append(f"\tint *indices = (int*)malloc(1 * sizeof(int));\n")
    code.append(f"\tif (!csr_val || !indptr || !indices) {{\n")
    code.append(f"\t\tprintf(\"Memory allocation failed for csr_val/indptr/indices\\n\");\n")
    code.append(f"\t\treturn 1;\n")
    code.append(f"\t}}\n")

def _append_benchmark_memsets(
    code: list[str],
    y_size: int,
    val_size: int,
    csr_val_size: int,
    indptr_size: int,
    indices_size: int,
    indent: str = "\t\t",
) -> None:
    code.append(f"{indent}memset(y, 0, sizeof(double)*{y_size});\n")
    code.append(f"{indent}memset(val, 0, {val_size if val_size > 0 else 1} * sizeof(double));\n")
    _append_csr_memsets(code, csr_val_size, indptr_size, indices_size, indent=indent)

def _append_csr_memsets(
    code: list[str],
    csr_val_size: int,
    indptr_size: int,
    indices_size: int,
    indent: str = "\t\t",
) -> None:
    if csr_val_size > 0:
        code.append(f"{indent}memset(csr_val, 0, {csr_val_size} * sizeof(double));\n")
        code.append(f"{indent}memset(indptr, 0, {indptr_size} * sizeof(int));\n")
        code.append(f"{indent}memset(indices, 0, {indices_size} * sizeof(int));\n")

def _append_timing_output(code: list[str], bench: int, dense_block_count: int) -> None:
    code.append('\tprintf("Sparse: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tprintf(\"%lu,\", sparse_times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")
    code.append('\tprintf("Dense: ");\n')
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tlong total_dense = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{dense_block_count}; j++) {{\n")
    code.append("\t\t\ttotal_dense += dense_block_times[j][i];\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"%lu,\", total_dense);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int j=0; j<{dense_block_count}; j++) {{\n")
    code.append('\t\tprintf("Dense Block %d: ", j+1);\n')
    code.append(f"\t\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\t\tprintf(\"%lu,\", dense_block_times[j][i]);\n")
    code.append("\t\t}\n")
    code.append("\t\tprintf(\"\\n\");\n")
    code.append("\t}\n")

def _append_benchmark_file_open(code: list[str], vbr_path: str, tensor_path: str) -> None:
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(tensor_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")

def _append_read_dense_inputs_from_vbrc(
    code: list[str],
    vbr_path: str,
    tensor_path: str,
    x_bound: int,
    val_size: int,
) -> None:
    _append_benchmark_file_open(code, vbr_path, tensor_path)
    code.append(f"\tmemset(x, 0, (size_t){x_bound} * sizeof(double));\n")
    code.append(f"\tmemset(val, 0, (size_t)({val_size if val_size > 0 else 1}) * sizeof(double));\n")
    code.append("\tchar c;\n")
    code.append("\tint x_size = 0, val_size = 0;\n")
    _append_read_double_array(code, "val", indent="\t")
    code.append("\tfclose(file1);\n")
    code.append(f'''\twhile (x_size < {x_bound} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
\t\tx_size++;
\t}}
\tfclose(file2);
''')

def _append_read_double_array(
    code: list[str],
    array_name: str,
    label: str | None = None,
    indent: str = "\t\t",
) -> None:
    label = label or array_name
    code.append(f"{indent}val_size=0;\n")
    code.append(f"{indent}assert(fscanf(file1, \"{label}=[%c\", &c) == 1);\n")
    code.append(f"{indent}if (c != ']') {{\n")
    code.append(f"{indent}\tungetc(c, file1);\n")
    code.append(f"{indent}\tassert(fscanf(file1, \"%lf\", &{array_name}[val_size]) == 1.0);\n")
    code.append(f"{indent}\tval_size++;\n")
    code.append(f"{indent}\twhile (1) {{\n")
    code.append(f"{indent}\t\tassert(fscanf(file1, \"%c\", &c) == 1);\n")
    code.append(f"{indent}\t\tif (c == ',') {{\n")
    code.append(f"{indent}\t\t\tassert(fscanf(file1, \"%lf\", &{array_name}[val_size]) == 1.0);\n")
    code.append(f"{indent}\t\t\tval_size++;\n")
    code.append(f"{indent}\t\t}} else if (c == ']') {{\n")
    code.append(f"{indent}\t\t\tbreak;\n")
    code.append(f"{indent}\t\t}} else {{\n")
    code.append(f"{indent}\t\t\tassert(0);\n")
    code.append(f"{indent}\t\t}}\n")
    code.append(f"{indent}\t}}\n")
    code.append(f"{indent}}}\n")
    code.append(f"{indent}if(fscanf(file1, \"%c\", &c));\n")
    code.append(f"{indent}assert(c=='\\n');\n")

def _append_read_int_array(
    code: list[str],
    array_name: str,
    label: str | None = None,
    indent: str = "\t\t",
) -> None:
    label = label or array_name
    code.append(f"{indent}val_size=0;\n")
    code.append(f"{indent}assert(fscanf(file1, \"{label}=[%d\", &{array_name}[val_size]) == 1.0);\n")
    code.append(f"{indent}val_size++;\n")
    code.append(f"{indent}while (1) {{\n")
    code.append(f"{indent}\tassert(fscanf(file1, \"%c\", &c) == 1);\n")
    code.append(f"{indent}\tif (c == ',') {{\n")
    code.append(f"{indent}\t\tassert(fscanf(file1, \"%d\", &{array_name}[val_size]) == 1.0);\n")
    code.append(f"{indent}\t\tval_size++;\n")
    code.append(f"{indent}\t}} else if (c == ']') {{\n")
    code.append(f"{indent}\t\tbreak;\n")
    code.append(f"{indent}\t}} else {{\n")
    code.append(f"{indent}\t\tassert(0);\n")
    code.append(f"{indent}\t}}\n")
    code.append(f"{indent}}}\n")
    code.append(f"{indent}if(fscanf(file1, \"%c\", &c));\n")
    code.append(f"{indent}assert(c=='\\n');\n")

def _append_spreg_initialization(
    code: list[str],
    vbr_path: str,
    csr_val_size: int,
    indptr_size: int,
    indices_size: int,
    nrows: int,
    ncols: int,
    threads: int,
) -> None:
    code.append(f"\n\t// Initial data load for spreg executor initialization\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    _append_csr_memsets(code, csr_val_size, indptr_size, indices_size, indent="\t")
    code.append("\tchar c;\n")
    code.append("\tint val_size=0;\n")
    _append_read_double_array(code, "val", indent="\t")
    _append_read_double_array(code, "csr_val", indent="\t")
    _append_read_int_array(code, "indptr", indent="\t")
    _append_read_int_array(code, "indices", indent="\t")
    code.append("\tfclose(file1);\n")
    code.append(f"\n\t// Initialize sparse-register-tiling executor (not timed)\n")
    code.append(f"\tvoid *spreg_handle = spmm_spreg_init(csr_val, indices, indptr, {nrows}, {ncols}, 512, {threads});\n")
    code.append("\tif (spreg_handle == NULL) {\n")
    code.append("\t\tprintf(\"Failed to initialize sparse-register-tiling executor\\n\");\n")
    code.append("\t\treturn 1;\n")
    code.append("\t}\n")

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

def spmv_kernel_mkl():
    """SpMV kernel using MKL's cblas_dgemv for dense blocks.

    Dense block values are stored column-major: val[(j-j_start)*block_rows + (i-i_start)]
    With CblasColMajor + CblasNoTrans, this matches our storage layout directly.

    y[block_rows] += A[block_rows x block_cols] * x[block_cols]
    """
    code = """
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

def _emit_spmv_naive_dense_blocks(code, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks):
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
                    count += 1
                nnz_block += 1
    return count

def _emit_spmv_naive_sparse_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices, kernel_name):
    if len(ublocks) > 0:
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append(f"\t\t{kernel_name}(y, csr_val, indices, indptr, x, {rpntr[-1]});\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

def _emit_spmv_mkl_sparse_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices, beta):
    if len(ublocks) > 0:
        code.append(f"""\t\tsparse_matrix_t A;
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;\n""")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append(f"\t\tmkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, {beta}, y);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

def _emit_spmv_spv8_sparse_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices):
    if len(ublocks) > 0:
        code.append(f"\t\tstruct csr_matrix mat = input_matrix({len(csr_val)}, {rpntr[-1]}, {cpntr[-1]}, csr_val, indices, indptr);\n")
        code.append("\t\tstruct tr_matrix tr = process(&mat);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append("\t\tspmv_tr_spvv8_kernel(&tr, x, y);\n")
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tsparse_times[i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")

def _gen_single_threaded_spmv_naive_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_kernel_include, sparse_kernel_function, sparse_kernel_call, dense_first=False, threads=1, use_mkl=False):
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
    if use_mkl and "#include <mkl.h>" not in (sparse_kernel_include or ""):
        code.append("#include <mkl.h>\n")
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
    _append_dense_array_allocations(code, rpntr[-1], cpntr[-1], len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1], len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
    if (len(indptr) > 0):
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1]))

    # Dispatch kernels in the appropriate order
    if dense_first:
        # Dense kernel first, then sparse kernel
        count = _emit_spmv_naive_dense_blocks(code, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks)
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
    else:
        # Sparse kernel first, then dense kernel (original behavior)
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
        count = _emit_spmv_naive_dense_blocks(code, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks)

    code.append("\t}\n")

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1])
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    return code

def _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_kernel_include, sparse_kernel_function, sparse_kernel_call, dense_dispatch, dense_first=False, threads=1):
    """Common code generation for single-threaded SpMV with configurable dense dispatch.

    ``dense_dispatch`` controls the per-block dense dispatch:

    * ``"mixed"`` -- threshold-based dispatch: ``cblas_dgemv`` for large,
      well-shaped blocks; handwritten kernels (``spmv_kernel``,
      ``spmv_kernel_2``, ``spmv_kernel_3``) for small/skinny/single-row/
      single-column blocks.
    * ``"mkl"`` -- always emit ``spmv_kernel_mkl`` (``cblas_dgemv``) for
      every dense block, regardless of shape.

    Args:
        sparse_kernel_include: String with additional includes (e.g., '#include "utility.h"\\n' or '#include <mkl.h>\\n')
        sparse_kernel_function: String with additional kernel function definitions (e.g., spmv_sparse_naive function)
        sparse_kernel_call: Function that generates the sparse kernel call code, takes (code, ublocks, csr_val, rpntr, cpntr, indptr, indices) as args
        dense_dispatch: ``"mixed"`` (threshold-based) or ``"mkl"`` (always cblas_dgemv).
        dense_first: If True, execute dense kernel before sparse kernel. If False, execute sparse kernel first.
        threads: Number of threads for MKL parallelization (default: 1)
    """
    assert dense_dispatch in ("mixed", "mkl"), f"Invalid dense_dispatch: {dense_dispatch}"
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

    # Mixed dispatch needs handwritten kernels for skinny/small blocks; always-MKL only needs cblas_dgemv.
    if dense_dispatch == "mixed":
        code.append(spmv_kernel())      # General naive kernel
        code.append(spmv_kernel_2())    # 1xM (single row) kernel
        code.append(spmv_kernel_3())    # Nx1 (single column) kernel
        code.append("\n")
    code.append(spmv_kernel_mkl())
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
    _append_dense_array_allocations(code, rpntr[-1], cpntr[-1], len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1], len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
    if (len(indptr) > 0):
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1]))

    # Build the dense-block dispatch as a separate list so we can place it
    # before or after the sparse kernel call without duplicating logic.
    # Threshold-based ("mixed") dispatch uses MKL only for large, reasonably-shaped
    # blocks. Handwritten kernels are faster for small-to-medium square blocks
    # (up to ~100x100), while MKL wins for larger blocks. Skinny blocks (e.g.
    # 2xN / Nx2) should NEVER use MKL -- MKL has too much call overhead for
    # these, handwritten kernels are 2-3x faster.
    MKL_AREA_THRESHOLD = 10000  # Use MKL for blocks with area >= 10000 (~100x100)
    MIN_DIM_FOR_MKL = 16  # Minimum dimension to consider MKL (avoids skinny blocks)
    dense_code = []
    count = 0
    nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    rows = rpntr[a+1] - rpntr[a]
                    cols = cpntr[b+1] - cpntr[b]
                    dense_code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    if dense_dispatch == "mkl":
                        # Always-MKL dispatch: every dense block goes through cblas_dgemv.
                        dense_code.append(f"\t\tspmv_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    elif cols == 1:
                        # Nx1 (single column) - use kernel_3
                        dense_code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                    elif rows == 1:
                        # 1xM (single row) - use kernel_2 (dot product)
                        dense_code.append(f"\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    elif rows * cols < MKL_AREA_THRESHOLD or min(rows, cols) < MIN_DIM_FOR_MKL:
                        # Small/medium block OR skinny block - use general naive kernel (faster due to lower overhead)
                        dense_code.append(f"\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    else:
                        # Large block with reasonable aspect ratio - use MKL dgemv
                        dense_code.append(f"\t\tspmv_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    dense_code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    dense_code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count += 1
                nnz_block += 1

    if dense_first:
        code.extend(dense_code)
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
    else:
        sparse_kernel_call(code, ublocks, csr_val, rpntr, cpntr, indptr, indices)
        code.extend(dense_code)

    code.append("\t}\n")

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1])
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    return code

def _gen_single_threaded_spmv_uzp_sparse_common(
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
    dir_name, filename, vbr_dir, dense_dispatch,
    bench: int = 5, sparse_mtx_path: str = "", threads: int = 1,
) -> int:
    """Shared body for UZP-sparse SpMV codegen with configurable dense dispatch.

    ``dense_dispatch`` controls the per-block dense dispatch:
    * ``"mixed"`` -- ``cblas_dgemv`` for normal blocks, ``spmv_kernel_3`` for Nx1.
    * ``"mkl"``  -- always ``cblas_dgemv``, regardless of shape.

    UZP preparation (z_polyhedrator + spf_aggregator) runs ONCE via a shell
    script outside the timing loop.  Only the UZP kernel execution is timed.
    """
    assert dense_dispatch in ("mixed", "mkl"), f"Invalid dense_dispatch: {dense_dispatch}"
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

    # Mixed dispatch needs the Nx1 fallback kernel; always-MKL doesn't.
    if dense_dispatch == "mixed":
        code.append(spmv_kernel_3())
        code.append("\n")
    code.append(spmv_kernel_mkl())
    code.append("\n")

    code.append("int main() {\n")
    code.append(f"\tconst int nrows = {rpntr[-1]};\n")
    code.append(f"\tconst int ncols = {cpntr[-1]};\n")
    code.append(f"\tconst int bench = {bench};\n\n")

    code.append("\t// Allocate vectors\n")
    _append_dense_array_allocations(code, rpntr[-1], cpntr[-1], len(val))
    code.append("\tassert(y && x && val);\n\n")

    code.append("\t// Read dense block values from VBRC (outside timing loop)\n")
    _append_read_dense_inputs_from_vbrc(code, vbr_path, vector_path, cpntr[-1], len(val))
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
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    code.append("\n")

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
                    if dense_dispatch == "mixed" and (cpntr[b + 1] - cpntr[b]) == 1:
                        # Nx1 (single column) - keep handwritten kernel_3 to avoid MKL call overhead.
                        code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                    else:
                        code.append(f"\t\tspmv_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);\n")
                    count += 1
                nnz_block += 1
    code.append("\t}\n\n")

    _append_timing_output(code, bench, count)
    _append_result_output(code, rpntr[-1])
    _append_free_allocated_memory(code, False)
    code.append("\treturn 0;\n")
    code.append("}\n")
    _write_generated_code(dir_name, filename, code)

    time2 = time.time_ns() // 1_000_000
    return time2 - time1

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

    # When sparse dispatch is naive, sparse dispatch comes before dense dispatch
    sparse_kernel_call = partial(_emit_spmv_naive_sparse_call, kernel_name=kernel_name)
    code = _gen_single_threaded_spmv_naive_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, '', kernel_func, sparse_kernel_call, dense_first=False, threads=threads)
    _write_generated_code(dir_name, filename, code)
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

    sparse_include = ''
    if len(ublocks) > 0:
        sparse_include = "#include <mkl.h>\n#include <mkl_spblas.h>\n"
    sparse_kernel_call = partial(_emit_spmv_mkl_sparse_call, beta=beta)
    code = _gen_single_threaded_spmv_naive_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, dense_first=dense_first, threads=threads, use_mkl=True)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_naive_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using naive dense kernels + SpV8 sparse kernel.
    
    When sparse dispatch is spv8, dense dispatch comes before sparse dispatch.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # When sparse dispatch is spv8, dense dispatch comes before sparse dispatch
    dense_first = True

    sparse_include = '#include "utility.h"\n' if len(ublocks) > 0 else ''
    # SpV8 kernel has internal OpenMP parallelization, just pass threads for omp_set_num_threads
    code = _gen_single_threaded_spmv_naive_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', _emit_spmv_spv8_sparse_call, dense_first=dense_first, threads=threads)
    _write_generated_code(dir_name, filename, code)
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
    _append_dense_array_allocations(code, rpntr[-1], cpntr[-1], len(val))
    code.append("\tassert(y && x && val);\n\n")

    code.append("\t// Read dense block values from VBRC (outside timing loop)\n")
    _append_read_dense_inputs_from_vbrc(code, vbr_path, vector_path, cpntr[-1], len(val))
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
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    code.append("\n")

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

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1])
    _append_free_allocated_memory(code, False)
    code.append("\treturn 0;\n")
    code.append("}\n")
    _write_generated_code(dir_name, filename, code)

    time2 = time.time_ns() // 1_000_000
    return time2 - time1

def gen_single_threaded_spmv_mkl_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using always-MKL dense dispatch (cblas_dgemv for every block) + naive 3-loop CSR sparse kernel.

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

    # When sparse dispatch is naive, sparse dispatch comes before dense dispatch
    sparse_kernel_call = partial(_emit_spmv_naive_sparse_call, kernel_name=kernel_name)
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, '', kernel_func, sparse_kernel_call, "mkl", dense_first=False, threads=threads)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_mkl_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using always-MKL dense dispatch (cblas_dgemv for every block) + MKL sparse kernel.

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

    # MKL sparse functions need mkl_spblas.h (mkl.h is already included by _gen_single_threaded_spmv_common)
    sparse_include = ''
    if len(ublocks) > 0:
        sparse_include = "#include <mkl_spblas.h>\n"
    sparse_kernel_call = partial(_emit_spmv_mkl_sparse_call, beta=beta)
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, "mkl", dense_first=dense_first, threads=threads)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_mkl_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using always-MKL dense dispatch (cblas_dgemv for every block) + SpV8 sparse kernel.

    When sparse dispatch is spv8, dense dispatch comes before sparse dispatch.

    Args:
        threads: Number of threads for parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # When sparse dispatch is spv8, dense dispatch comes before sparse dispatch
    dense_first = True

    sparse_include = '#include "utility.h"\n' if len(ublocks) > 0 else ''
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', _emit_spmv_spv8_sparse_call, "mkl", dense_first=dense_first, threads=threads)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_mkl_uzp(
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
    dir_name, filename, vbr_dir,
    bench: int = 5, sparse_mtx_path: str = "", threads: int = 1,
) -> int:
    """UZP sparse dispatch + always-MKL dense dispatch (cblas_dgemv for every block)."""
    return _gen_single_threaded_spmv_uzp_sparse_common(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
        dir_name, filename, vbr_dir, "mkl",
        bench=bench, sparse_mtx_path=sparse_mtx_path, threads=threads,
    )

def gen_single_threaded_spmv_mixed_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using mixed dense dispatch (cblas_dgemv + handwritten kernels) + naive 3-loop CSR sparse kernel.

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

    # When sparse dispatch is naive, sparse dispatch comes before dense dispatch
    sparse_kernel_call = partial(_emit_spmv_naive_sparse_call, kernel_name=kernel_name)
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, '', kernel_func, sparse_kernel_call, "mixed", dense_first=False, threads=threads)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_mixed_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using mixed dense dispatch (cblas_dgemv + handwritten kernels) + MKL sparse kernel.

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

    # MKL sparse functions need mkl_spblas.h (mkl.h is already included by _gen_single_threaded_spmv_common)
    sparse_include = ''
    if len(ublocks) > 0:
        sparse_include = "#include <mkl_spblas.h>\n"
    sparse_kernel_call = partial(_emit_spmv_mkl_sparse_call, beta=beta)
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', sparse_kernel_call, "mixed", dense_first=dense_first, threads=threads)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_mixed_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate code using mixed dense dispatch (cblas_dgemv + handwritten kernels) + SpV8 sparse kernel.

    When sparse dispatch is spv8, dense dispatch comes before sparse dispatch.

    Args:
        threads: Number of threads for parallelization (default: 1)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000

    # When sparse dispatch is spv8, dense dispatch comes before sparse dispatch
    dense_first = True

    sparse_include = '#include "utility.h"\n' if len(ublocks) > 0 else ''
    code = _gen_single_threaded_spmv_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, sparse_include, '', _emit_spmv_spv8_sparse_call, "mixed", dense_first=dense_first, threads=threads)
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv_mixed_uzp(
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
    dir_name, filename, vbr_dir,
    bench: int = 5, sparse_mtx_path: str = "", threads: int = 1,
) -> int:
    """UZP sparse dispatch + mixed dense dispatch (cblas_dgemv with Nx1 fallback to ``spmv_kernel_3``)."""
    return _gen_single_threaded_spmv_uzp_sparse_common(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
        dir_name, filename, vbr_dir, "mixed",
        bench=bench, sparse_mtx_path=sparse_mtx_path, threads=threads,
    )

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

def _gen_multi_threaded_stub(*args, **kwargs):
    raise NotImplementedError("Multi-threaded code generation stubs are placeholders for a future implementation.")

def gen_multi_threaded_spmv_naive_naive(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_naive_mkl(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_naive_spv8(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_naive_uzp(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mkl_naive(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mkl_mkl(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mkl_spv8(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mkl_uzp(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mixed_naive(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mixed_mkl(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mixed_spv8(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmv_mixed_uzp(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

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

def _gen_single_threaded_spmm_naive_sparse_common(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, dense_dispatch, bench:int=5, threads:int=1)->int:
    """Shared body for SpMM codegen with handwritten naive sparse dispatch.

    ``dense_dispatch`` controls the per-block dense dispatch:
    * ``"mixed"`` -- threshold-based dispatch (``should_use_mkl_for_block``):
      MKL ``cblas_dgemm`` for chunky blocks, handwritten kernels otherwise.
    * ``"mkl"``  -- always emit ``spmm_kernel_mkl`` (``cblas_dgemm``) for every dense block.
    """
    assert dense_dispatch in ("mixed", "mkl"), f"Invalid dense_dispatch: {dense_dispatch}"
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
    # Mixed dispatch needs handwritten kernel variants for thin/vector-like blocks.
    if dense_dispatch == "mixed":
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
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
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
                    # Always-MKL dispatch routes every block through cblas_dgemm; mixed
                    # dispatch picks MKL only for "chunky" blocks via should_use_mkl_for_block.
                    if dense_dispatch == "mkl" or should_use_mkl_for_block(block_rows, block_cols):
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

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1] * 512)
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    # Log dispatch statistics
    print(f"  [{dense_dispatch}_naive] Block dispatch: {mkl_blocks} MKL, {naive_blocks} naive (of {mkl_blocks + naive_blocks} total dense blocks)")
    return time2-time1

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
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
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

    _append_timing_output(code, bench, count)
    
    _append_result_output(code, rpntr[-1] * 512)
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
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
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
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

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1] * 512)
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
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
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    
    # For spreg, load CSR data once to initialize the executor. The benchmark
    # loop reloads data for consistency with the other sparse backends.
    if (len(ublocks) > 0):
        _append_spreg_initialization(code, vbr_path, len(csr_val), len(indptr), len(indices), rpntr[-1], cpntr[-1], threads)

    # === BENCHMARK LOOP - load data each iteration (outside timing), like SpMV ===
    code.append(f"\n\t// Benchmark loop - load data each iteration (outside timing)\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
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

    _append_timing_output(code, bench, count)
    
    _append_result_output(code, rpntr[-1] * 512)

    # Cleanup sparse-register-tiling executor
    if len(ublocks) > 0:
        code.append("\n\t// Cleanup sparse-register-tiling executor\n")
        code.append("\tspmm_spreg_cleanup(spreg_handle);\n")
    
    _append_free_allocated_memory(code, True)
    
    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmm_mkl_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with always-MKL dense dispatch (cblas_dgemm for every block) + handwritten naive sparse kernel."""
    return _gen_single_threaded_spmm_naive_sparse_common(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
        dir_name, filename, vbr_dir, "mkl", bench=bench, threads=threads,
    )

def gen_single_threaded_spmm_mkl_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with always-MKL dense dispatch (cblas_dgemm for every block) + MKL sparse kernel (mkl_sparse_d_mm).

    Every dense block is dispatched through MKL ``cblas_dgemm``
    (``spmm_kernel_mkl``), regardless of shape; MKL ``mkl_sparse_d_mm`` for
    the sparse CSR part.
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
    # Always-MKL dispatch only needs the MKL dense kernel.
    code.append(spmm_kernel_mkl())
    code.append("\n")
    code.append("int main() {\n")
    # Set thread counts
    code.append(f"\tmkl_set_num_threads({threads});\n\n")
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))

    if (len(ublocks) > 0):
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
                    # Always-MKL dispatch: every dense block goes through cblas_dgemm.
                    code.append(f"\t\tspmm_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1] * 512)
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    print(f"  [mkl_mkl] Block dispatch: {count} MKL (of {count} total dense blocks)")
    return time2-time1

def gen_single_threaded_spmm_mkl_spreg(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with always-MKL dense dispatch (cblas_dgemm for every block) + sparse-register-tiling sparse kernel.

    Every dense block is dispatched through MKL ``cblas_dgemm``
    (``spmm_kernel_mkl``), regardless of shape; sparse-register-tiling for
    the sparse remainder.

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
    # Always-MKL dispatch only needs the MKL dense kernel.
    code.append(spmm_kernel_mkl())
    code.append("\n")
    code.append("int main() {\n")
    # Set thread counts
    code.append(f"\tmkl_set_num_threads({threads});\n\n")
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)

    # === INITIALIZE SPARSE-REGISTER-TILING EXECUTOR (once, needs initial data load) ===
    if (len(ublocks) > 0):
        _append_spreg_initialization(code, vbr_path, len(csr_val), len(indptr), len(indices), rpntr[-1], cpntr[-1], threads)

    code.append(f"\n\t// Benchmark loop - load data each iteration (outside timing)\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
    code.append("\t\tfclose(file1);\n")
    code.append('''\t\twhile (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
            x_size++;
        }}
        fclose(file2);\n'''.format(cpntr[-1] * 512))
    if (len(ublocks) > 0):
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
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
                    # Always-MKL dispatch: every dense block goes through cblas_dgemm.
                    code.append(f"\t\tspmm_kernel_mkl(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append(f"\t\tdense_block_times[{count}][i] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    count+=1
                nnz_block += 1
    code.append("\t}\n")

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1] * 512)

    if len(ublocks) > 0:
        code.append("\n\t// Cleanup sparse-register-tiling executor\n")
        code.append("\tspmm_spreg_cleanup(spreg_handle);\n")

    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    print(f"  [mkl_spreg] Block dispatch: {count} MKL (of {count} total dense blocks)")
    return time2-time1

def gen_single_threaded_spmm_mixed_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with mixed dense dispatch (cblas_dgemm + handwritten kernels) + handwritten naive sparse kernel."""
    return _gen_single_threaded_spmm_naive_sparse_common(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val,
        dir_name, filename, vbr_dir, "mixed", bench=bench, threads=threads,
    )

def gen_single_threaded_spmm_mixed_mkl(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with mixed dense dispatch (MKL cblas_dgemm + handwritten kernels) + MKL sparse kernel (mkl_sparse_d_mm).

    Uses MKL MKL for dense blocks and MKL's mkl_sparse_d_mm for the sparse CSR part.
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
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)
    # Benchmark loop - load data each iteration (outside timing)
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
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
                    # Mixed dispatch picks MKL only for "chunky" blocks via should_use_mkl_for_block.
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

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1] * 512)
    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    # Log dispatch statistics
    print(f"  [mixed_mkl] Block dispatch: {mkl_blocks} MKL, {naive_blocks} naive (of {mkl_blocks + naive_blocks} total dense blocks)")
    return time2-time1

def gen_single_threaded_spmm_mixed_spreg(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5, threads:int=1)->int:
    """Generate SpMM code with mixed dense dispatch (MKL cblas_dgemm + handwritten kernels) + sparse-register-tiling sparse kernel.

    Threshold-based dispatch for dense blocks (MKL or handwritten); sparse-register-tiling for the sparse remainder.
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
    _append_dense_array_allocations(code, rpntr[-1] * 512, cpntr[-1] * 512, len(val))
    _append_csr_array_allocations(code, len(csr_val), len(indptr), len(indices))
    code.append("\tstruct timespec t1, t2;\n")
    prev_count = _count_dense_blocks(rpntr, cpntr, bpntrb, bpntre, bindx, ublocks)
    _append_timing_workspace(code, bench, prev_count)

    # === INITIALIZE SPARSE-REGISTER-TILING EXECUTOR (once, needs initial data load) ===
    if (len(ublocks) > 0):
        _append_spreg_initialization(code, vbr_path, len(csr_val), len(indptr), len(indices), rpntr[-1], cpntr[-1], threads)

    code.append(f"\n\t// Benchmark loop - load data each iteration (outside timing)\n")
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    _append_benchmark_file_open(code, vbr_path, vector_path)
    _append_benchmark_memsets(code, rpntr[-1] * 512, len(val), len(csr_val), len(indptr), len(indices))
    code.append("\t\tchar c;\n")
    code.append(f"\t\tint x_size=0, val_size=0;\n")
    _append_read_double_array(code, "val")
    if (len(ublocks) > 0):
        _append_read_double_array(code, "csr_val")
        _append_read_int_array(code, "indptr")
        _append_read_int_array(code, "indices")
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
                    # Mixed dispatch picks MKL only for "chunky" blocks via should_use_mkl_for_block.
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

    _append_timing_output(code, bench, count)

    _append_result_output(code, rpntr[-1] * 512)

    # Cleanup sparse-register-tiling executor
    if len(ublocks) > 0:
        code.append("\n\t// Cleanup sparse-register-tiling executor\n")
        code.append("\tspmm_spreg_cleanup(spreg_handle);\n")

    _append_free_allocated_memory(code, True)

    code.append("}\n")
    _write_generated_code(dir_name, filename, code)
    time2 = time.time_ns() // 1_000_000
    # Log dispatch statistics
    print(f"  [mixed_spreg] Block dispatch: {mkl_blocks} MKL, {naive_blocks} naive (of {mkl_blocks + naive_blocks} total dense blocks)")
    return time2-time1

def vbr_spmm_codegen(filename: str, dir_name: str, vbr_dir: str, threads: int, bench: int = 5, mkl: bool = False)->int:
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbr_path)
    time1 = time.time_ns() // 1_000_000
    # Use single-threaded codegen with threads parameter - it has internal OpenMP support
    gen_single_threaded_spmm_naive_naive(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench, threads)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_multi_threaded_spmm_naive_naive(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_naive_mkl(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_naive_spreg(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_mkl_naive(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_mkl_mkl(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_mkl_spreg(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_mixed_naive(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_mixed_mkl(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)

def gen_multi_threaded_spmm_mixed_spreg(*args, **kwargs):
    return _gen_multi_threaded_stub(*args, **kwargs)
