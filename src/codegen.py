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

def num_past_unrolled(ublocks: list[int], indx: list[int], start_pos: int) -> int:
    num_unrolled = 0
    for ublock in ublocks:
        if ublock < start_pos:
            num_unrolled += indx[ublock+1] - indx[ublock]
        else:
            break
    return num_unrolled


def vbr_spmv_codegen_for_all(density: int = 0):
    if density == 0:
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_SpMV"
    else:
        raise Exception("Not implemented")
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    runtimes = {}
    for filename in os.listdir(input_dir_name):
        assert(filename.endswith(".vbr"))
        core_name = filename[:-len(".vbr")]
        run_time = vbr_spmv_codegen(core_name, density, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes

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

def gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench:int=5)->int:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    if (len(ublocks) > 0):
        code.append("#include <mkl.h>\n")
        code.append("#include <mkl_spblas.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append(spmv_kernel())
    code.append("\n")
    code.append(spmv_kernel_2())
    code.append("\n")
    code.append(spmv_kernel_3())
    code.append("\n")
    code.append("int main() {\n")
    code.append(f"\tlong times[{bench}];\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tdouble y[{rpntr[-1]}] = {{0}};\n")
    code.append(f"\tdouble x[{cpntr[-1]}] = {{0}};\n")
    if len(val) > 0:
        code.append(f"\tdouble val[{len(val)}] = {{0}};\n")
    else:
        code.append(f"\tdouble val[1] = {{0}};\n")
    if len(csr_val) > 0:
        code.append(f"\tdouble csr_val[{len(csr_val)}] = {{0}};\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%c", &c) == 1);
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
        code.append('''\tval_size=0;
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
        code.append(f"""\tint indptr[{len(indptr)}] = {{0}};
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
    code.append("\tfclose(file1);\n")
    code.append('''
    while (x_size < {0} && fscanf(file2, "%lf,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    if len(ublocks) > 0:
        code.append(f"""\tsparse_matrix_t A;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, {rpntr[-1]}, {cpntr[-1]}, indptr, indptr+1, indices, csr_val);
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_set_num_threads(1);\n""")
    code.append("\tstruct timespec t1;\n")
    code.append("\tstruct timespec t2;\n")
    code.append(f"\tfor (int i=0; i<{bench+1}; i++) {{\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1]))
    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
    count = 0
    nnz_block = 0
    if (len(ublocks) > 0):
        code.append("\t\tmkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);\n")
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    if (rpntr[a+1] - rpntr[a]) == 1:
                        code.append(f"\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    elif (cpntr[b+1] - cpntr[b]) == 1:
                        code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                    else:
                        code.append(f"\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    count+=1
                nnz_block += 1
    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
    code.append("\t\tif (i!=0)\n")
    code.append("\t\t\ttimes[i-1] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
    code.append("\t}\n")
    code.append('\tprintf("{0} = ");\n'.format(filename))
    code.append("\tfor (int i=0; i<{0}; i++) {{\n".format(bench))
    code.append("\t\tprintf(\"%lu,\", times[i]);\n")
    code.append("\t}\n")
    code.append("\tprintf(\"\\n\");\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")
    code.append("}\n")
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
        f.write(f"double y[{rpntr[-1]}] = {{0}};\n")
        f.write(f"double x[{cpntr[-1]}] = {{0}};\n")
        if len(val) > 0:
            f.write(f"double val[{len(val)}] = {{0}};\n")
        else:
            f.write(f"double val[1] = {{0}};\n")
        if len(ublocks) > 0:
            f.write(f"double csr_val[{len(csr_val)}] = {{0}};\n\n")
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
        f.write("}\n")

def vbr_spmv_codegen(filename: str, dir_name: str, vbr_dir: str, threads: int, bench: int = 5)->int:
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbr_path)
    time1 = time.time_ns() // 1_000_000
    if threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
    time2 = time.time_ns() // 1_000_000
    return time2-time1