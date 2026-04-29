import os
import pathlib
import time


FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")


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
