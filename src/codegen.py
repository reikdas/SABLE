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
        run_time = vbr_spmv_codegen(core_name, output_dir_name, vbr_dir=input_dir_name, threads=1)
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

def spmv_sparse():
    # Write the C code for sparse SpMV CSR
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

def gen_loop(i_start: int, i_end: int,
             j_start: int, j_end: int,
             indx_offset: int) -> str:
    i_extent = i_end - i_start
    j_extent = j_end - j_start
    return (
        f"\t\tfor j in range({j_extent}):\n"
        f"\t\t    for i in range({i_extent}):\n"
        f"\t\t        out[i + {i_start}] "
        f"+= val[{indx_offset} + j*{i_extent} + i] "
        f"* vec[j + {j_start}]\n"
    )

def gen_single_threaded_spmv_python(val: list[float], indx: list[int], bindx: list[int], rpntr: list[int], cpntr: list[int], bpntrb: list[int], bpntre: list[int], ublocks: list[int], indptr: list[int], indices: list[int], csr_val: list[float], dir_name: str, filename: str, vbr_dir: str, bench: int = 5) -> int:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time1 = time.time_ns() // 1_000_000
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    code = []
    code.append(f"""import time
from pathlib import Path

import numpy as np
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T\n\n""")
    
    count = 0
    
    if len(val) > 0:
        code.append(f"""@I.ir_module
class DenseModule:
    @T.prim_func
    def spmv_dense(VEC: T.handle,
                    VAL: T.handle,
    """)
        if len(csr_val) > 0:
            code.append("\t\t\t\tOUT_SPARSE: T.handle,\n")
        code.append(f"""\t\t\t\tOUT: T.handle):
        vec = T.match_buffer(VEC, ({cpntr[-1]},), "float64")
        val = T.match_buffer(VAL, ({len(val)},), "float64")
        out = T.match_buffer(OUT, ({rpntr[-1]},), "float64")\n""")

        if len(csr_val) > 0:
            code.append(f"\t\tout_sparse = T.match_buffer(OUT_SPARSE, ({rpntr[-1]},), 'float64')\n")

        code.append(f"""
        for i in T.serial({rpntr[-1]}):""")
        if len(csr_val) > 0:
            code.append("""
            out[i] = out_sparse[i]            
""")
        else:
            code.append("""
            out[i] = 0.0
""")
        nnz_block = 0
        for a in range(len(rpntr)-1):
            if bpntrb[a] == -1: 
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if nnz_block not in ublocks:
                        i_extent = rpntr[a+1] - rpntr[a]
                        j_extent = cpntr[b+1] - cpntr[b]
                        code.append(f"""
        T.evaluate(tvm.tir.call_packed("my_timer.start", {count+1}))
        for j in T.serial({j_extent}):
            for i in T.serial({i_extent}):
                with T.block("db{nnz_block}"):
                    T.init()
                    out[i + {rpntr[a]}] += val[{indx[count]} + j * {i_extent} + i] * vec[j + {cpntr[b]}]
        T.evaluate(tvm.tir.call_packed("my_timer.stop", {count+1}))\n""")
                        count += 1
                    nnz_block += 1

        param_str = 'vec: R.Tensor(("k",), dtype="float64"), val: R.Tensor(("v",), dtype="float64")'
        arg_str = "vec, val"
        if len(csr_val) > 0:
            param_str += ', sparse_out: R.Tensor(("n",), dtype="float64")'
            arg_str += ', sparse_out'
        code.append(f"""
    @R.function
    def main({param_str}):
        cls = DenseModule
        out = R.call_tir(cls.spmv_dense, ({arg_str}), out_sinfo=R.Tensor(({rpntr[-1]},), dtype="float64"))
        return out\n""")

    if len(csr_val) > 0:
        code.append(f"""
@I.ir_module
class SparseModule:
    @T.prim_func
    def spmv_sparse(VEC: T.handle,
                    CSR_VAL: T.handle,
                    INDICES: T.handle,
                    IND_PTR: T.handle,
                    OUT: T.handle):
        vec = T.match_buffer(VEC, ({cpntr[-1]},), "float64")
        csr_val = T.match_buffer(CSR_VAL, ({len(csr_val)},), "float64")
        indices = T.match_buffer(INDICES, ({len(indices)},), "int32")
        indptr = T.match_buffer(IND_PTR, ({len(indptr)},), "int32")
        out = T.match_buffer(OUT, ({rpntr[-1]},), "float64")

        for i in T.serial({rpntr[-1]}):
            out[i] = 0.0
        T.evaluate(tvm.tir.call_packed("my_timer.start", 0))
        for i in T.serial({rpntr[-1]}):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            for j in T.serial(row_end - row_start):
                out[i] += csr_val[row_start + j] * vec[indices[row_start + j]]
        T.evaluate(tvm.tir.call_packed("my_timer.stop", 0))

    @R.function
    def main(vec: R.Tensor(("k",), dtype="float64"),
             csr_val: R.Tensor(("n",), dtype="float64"),
             indices: R.Tensor(("m",), dtype="int32"),
             indptr: R.Tensor(("l",), dtype="int32")):
        cls = SparseModule
        out = R.call_tir(cls.spmv_sparse, (vec, csr_val, indices, indptr), out_sinfo=R.Tensor(({rpntr[-1]},), dtype="float64"))
        return out\n""")

    code.append(f"""
if __name__ == "__main__":
    # Initialize timer_log with correct number of timers
    # Timer 0: Sparse, Timers 1 to {count}: Dense blocks
    timer_log = {{i: [] for i in range({count + 1})}}

    @tvm.register_func("my_timer.start")
    def timer_start(id: int):
        if "_timers" not in globals():
            global _timers
            _timers = {{}}
        _timers[id] = time.perf_counter_ns()

    @tvm.register_func("my_timer.stop")
    def timer_stop(id: int):
        t2 = time.perf_counter_ns()
        t1 = _timers.get(id, None)
        if t1 is None:
            print(f"Timer {{id}} was not started!")
        else:
            timer_log[id].append(t2 - t1)

    vbrc_path = Path("{os.path.abspath(vbr_path)}")
    vector_path = Path("{vector_path}")
    expected = {{
        "val": (np.float64, {len(val)}),
        "csr_val": (np.float64, {len(csr_val)}),
        "indptr": (np.int32, {len(indptr)}),
        "indices": (np.int32, {len(indices)}),
    }}
    arrays = {{}}
    with vbrc_path.open() as f:
        for line in f:
            for key, (dtype, _) in expected.items():
                if line.startswith(f"{{key}}="):
                    payload = line.split("=", 1)[1].strip().lstrip("[").rstrip("]\\n")
                    arrays[key] = np.fromstring(payload, sep=",", dtype=dtype)
                    break
    missing = [k for k in expected if k not in arrays]
    if missing:
        raise ValueError(f"Keys not found in {{vbrc_path}}: {{missing}}")
    for key, (_, want) in expected.items():
        got = arrays[key].size
        assert got == want, f"{{key}} has length {{got}}, expected {{want}}"

    val, csr_val = arrays["val"], arrays["csr_val"]
    indptr, indices = arrays["indptr"], arrays["indices"]

    x = np.fromstring(vector_path.read_text(), sep=",", dtype=np.float64)
    assert x.size == {cpntr[-1]}, f"x length {{x.size}}, expected {cpntr[-1]}"\n""")

    if len(val) > 0:
        code.append(f"\tval_arg = tvm.nd.array(val, device=tvm.cpu())\n")
    if len(csr_val) > 0:
        code.append(f"""\tdata_arg = tvm.nd.array(csr_val, device=tvm.cpu())
    indices_arg = tvm.nd.array(indices, device=tvm.cpu())
    indptr_arg = tvm.nd.array(indptr, device=tvm.cpu())\n""")
    code.append(f"\tvec_arg = tvm.nd.array(x, device=tvm.cpu())\n")

    if (len(csr_val) > 0):
        code.append(f"""
    target_sparse = tvm.target.Target("llvm -num-cores 1 -mtriple=x86_64-pc-linux-gnu")
    mod_sparse = SparseModule
    ex_sparse = relax.build(mod_sparse, target=target_sparse)
    vm_sparse = relax.VirtualMachine(ex_sparse, tvm.cpu())\n""")

    if len(val) > 0:
        code.append(f"""
    target_dense = tvm.target.Target("llvm -num-cores 1 -mtriple=x86_64-pc-linux-gnu -mattr=+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl,+avx512vbmi,+avx512vnni,+avx512bitalg,+avx512fp16")
    mod_dense = DenseModule
    ex_dense = relax.build(mod_dense, target=target_dense)
    vm_dense = relax.VirtualMachine(ex_dense, tvm.cpu())\n""")

    code.append(f"""
    N = {bench}
    B = 8
    buf = [None] * B
    final_buf = [None] * B
    
    for base in range(0, N, B):
        n = min(B, N - base)\n""")
    
    if (len(csr_val) > 0):
        code.append('''
        for k in range(n):
            buf[k] = vm_sparse["main"](vec_arg, data_arg, indices_arg, indptr_arg)\n''')
        
    if len(val) > 0:
        if len(csr_val) > 0:
            param_str = "vec_arg, val_arg, buf[k]"
        else:
            param_str = "vec_arg, val_arg"
        code.append(f'''
        for k in range(n):
            final_buf[k] = vm_dense["main"]({param_str})\n''')

    code.append(f'''
    print("Sparse: ", end="")
    for t in timer_log[0]:
        print(f"{{t}},", end="")
    print()

    # Print all total dense times (sum of all dense blocks)
    print("Dense: ", end="")
    # Determine the number of iterations based on available data
    num_iterations = max(len(timer_log[0]), max([len(timer_log[i]) for i in range(1, {count + 1})] if {count} > 0 else [0]))
    for i in range(num_iterations):
        total_dense = 0
        for dense_id in range(1, {count + 1}):
            if i < len(timer_log[dense_id]):
                total_dense += timer_log[dense_id][i]
        print(f"{{total_dense}},", end="")
    print()

    # Print all individual dense block times
    for dense_id in range(1, {count + 1}):
        print(f"Dense Block {{dense_id}}: ", end="")
        for t in timer_log[dense_id]:
            print(f"{{t}},", end="")
        print()
    print()\n''')

    if len(val) == 0 and len(csr_val) > 0:
        code.append('''\tfor elem in buf[0].numpy():\n''')
    elif len(val) > 0:
        code.append('''\tfor elem in final_buf[0].numpy():\n''')
    else:
        raise Exception("Something unexpected happened")
    
    code.append('''\t\tprint(elem)\n''')

    full_source = "".join(code).expandtabs(4)
    with open(os.path.join(dir_name, filename+".py"), "w") as f:
        f.writelines(full_source)
    time2 = time.time_ns() // 1_000_000
    return time2-time1


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
    code.append("#include <assert.h>\n\n")
    code.append(spmv_kernel())
    code.append("\n")
    code.append(spmv_kernel_2())
    code.append("\n")
    code.append(spmv_kernel_3())
    code.append("\n")
    code.append(spmv_sparse())
    code.append("\n")
    code.append("int main() {\n")
    code.append(f"\tlong times[{bench}];\n")
    code.append(f"\tdouble *y = (double*)malloc({rpntr[-1]} * sizeof(double));\n")
    code.append(f"\tdouble *x = (double*)malloc({cpntr[-1]} * sizeof(double));\n")
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
    code.append(f"\tlong dense_block_times[{prev_count}][{bench}];\n")
        # Initialize arrays to zero
    code.append(f"\tfor (int i=0; i<{bench}; i++) {{\n")
    code.append("\t\tsparse_times[i] = 0;\n")
    code.append(f"\t\tfor (int j=0; j<{prev_count}; j++) {{\n")
    code.append("\t\t\tdense_block_times[j][i] = 0;\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append(f"\tfor (int i=0; i<{bench+1}; i++) {{\n")
    code.append(f"\t\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\t\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\t\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\t\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\t\tmemset(y, 0, sizeof(double)*{0});\n".format(rpntr[-1]))
    code.append(f"\t\tmemset(x, 0, {cpntr[-1]} * sizeof(double));\n")
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
        fclose(file2);\n'''.format(cpntr[-1]))
    if (len(ublocks) > 0):
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
        code.append("\t\tspmv_sparse(y, csr_val, indices, indptr, x, {0});\n".format(rpntr[-1]))
        code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
        code.append("\t\tif (i!=0) {\n")
        code.append("\t\t\tsparse_times[i-1] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
        code.append("\t\t}\n")
    else:
        code.append("\t\tif (i!=0) {\n")
        code.append("\t\t\tsparse_times[i-1] = 0;\n")
        code.append("\t\t}\n")

    count = 0
    nnz_block = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    # Start timing for this dense block
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
                    if (rpntr[a+1] - rpntr[a]) == 1:
                        code.append(f"\t\tspmv_kernel_2(y, x, val, {rpntr[a]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    elif (cpntr[b+1] - cpntr[b]) == 1:
                        code.append(f"\t\tspmv_kernel_3(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {indx[count]});\n")
                    else:
                        code.append(f"\t\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    # End timing for this dense block
                    code.append("\t\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
                    code.append("\t\tif (i!=0) {\n")
                    code.append(f"\t\t\tdense_block_times[{count}][i-1] = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);\n")
                    code.append("\t\t}\n")
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
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%lf\\n\", y[i]);\n")
    code.append("\t}\n")
    
    # Free allocated memory
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

def vbr_spmv_codegen(filename: str, dir_name: str, vbr_dir: str, threads: int, bench: int = 5, mkl: bool = False)->int:
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbr_path)
    time1 = time.time_ns() // 1_000_000
    if mkl:
        gen_single_threaded_spmv_dgemv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
    elif threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def vbr_spmv_codegen_python(filename: str, dir_name: str, vbr_dir: str, bench: int = 5)->int:
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbr_path)
    time1 = time.time_ns() // 1_000_000
    gen_single_threaded_spmv_python(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dir_name, filename, vbr_dir, bench)
    time2 = time.time_ns() // 1_000_000
    return time2-time1