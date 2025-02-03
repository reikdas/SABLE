import inspect
import itertools
import os
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Union

from utils.fileio import read_vbr

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

curr_block_instructions = []

def _codegen(f):
    global curr_block_instructions
    prev = curr_block_instructions
    try:
        curr_block_instructions = []
        f()
        if len(curr_block_instructions) > 0:
            res = ";".join(str(e) for e in curr_block_instructions if e!="")
            res += ";"
        else:
            res = ""
    finally:
        curr_block_instructions = prev
    return f"{res}"

def codegen(fun):
    def res(*args, **kwargs):
        return _codegen(lambda: fun(*args, **kwargs))
    return res

ids = defaultdict(lambda: itertools.count())

def fresh_name(prefix='i'):
    id = next(ids[prefix])
    if id == 0:
        return f"{prefix}"
    else:
        return f"{prefix}${id}"

class NumVal:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name
    
    def __add__(self, v2: "NumVal"):
        return NumVal(f"({self} + {v2})")

    def __sub__(self, v2: "NumVal"):
        return NumVal(f"({self} - {v2})")

    def __mul__(self, v2: "NumVal"):
        return NumVal(f"({self} * {v2})")
    
    def __radd__(self, v2: int):
        return NumVal(f"({v2} + {self})")

class ConcreteNumVal(NumVal):
    def __init__(self, name: str, value):
        self.name = name
        self.value = value

class ArrayVal:
    def __init__(self, name: str):
        self.name = name
    
    def slice(self, start: NumVal):
        return ArrayVal(f"(&{self.name}[{start}])")

    def __getitem__(self, idx: NumVal):
        return NumVal(f"{self.name}[{idx}]")
    
    def __setitem__(self, idx: NumVal, val: NumVal):
        curr_block_instructions.append(f"{self.name}[{idx}] = {val}")

class ConcreteArrayVal:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value
    
    def slice(self, start: int):
        name = f"{self.name}_slice"
        n = f"{fresh_name(name)}"
        curr_block_instructions.append(f"\nfloat* {n} = &({self.name}[{start}])")
        return ConcreteArrayVal(n, self.value[start:])

    def __getitem__(self, idx: int):
        return ConcreteNumVal(f"{self.name}[{idx}]", self.value[idx])


@dataclass
class RepRange:
    start: int
    stop: int

    def __len__(self):
        return self.stop - self.start

def it(r: Union[RepRange, range], fun: Callable[[NumVal], Any]):
    iname = list(inspect.signature(fun).parameters.keys())[0]

    if isinstance(r, RepRange):
        # i = NumVal(fresh_name(iname))
        i = NumVal(iname)
        z = codegen(fun)(i)
        if z != "":
            curr_block_instructions.append(f"for (int {i} = {r.start}; {i} < {r.stop}; {i}++) {{")
            curr_block_instructions.append(z+"\n")
            curr_block_instructions.append("}\n")
    else:
        for i in r:
            curr_block_instructions.append(codegen(fun)(i))

def it2(r1: range, r2: range, f: Callable[[NumVal, NumVal], Any]):
    return it(r1, lambda i: it(r2, lambda j: f(i, j)))

def it3(r1: range, r2: range, r3: range, f: Callable[[NumVal, NumVal, NumVal], Any]):
    return it(r1, lambda i: it(r2, lambda j: it(r3, lambda k: f(i, j, k))))

def isDense(val):
    if isinstance(val, ConcreteNumVal):
        return val.value!=0
    if isinstance(val, NumVal):
        return True
    else:
        raise Exception("Invalid type")
    
def spmv(
    row_idxs: Union[range, RepRange], # (row_start, row_end)
    col_idxs: range, # (col_start, col_end)
    col_maj_val, # dense block from vbr
    x: ArrayVal, # dense vector / matrix to multiply
    y: ArrayVal, # output
):
    def op(j, i):
        val = col_maj_val[(j - col_idxs.start)*len(row_idxs) + (i - row_idxs.start)]
        if isDense(val):
            y[i] += val * x[j]

    return it2(col_idxs, row_idxs, op)

def spmm(
    row_idxs: Union[range, RepRange], # (row_start, row_end)
    col_idxs: Union[range, RepRange], # (col_start, col_end)
    dense_idxs: RepRange,
    col_maj_val, # dense block from vbr
    x: ArrayVal, # dense vector / matrix to multiply
    y: ArrayVal, # output
):
    def op(i, k, j):
        val = col_maj_val[(k-col_idxs.start)*len(row_idxs) + (i - row_idxs.start)]
        if isDense(val):
            y[i*512 + j] += val * x[k*512+j]

    return it3(row_idxs, col_idxs, dense_idxs, op)

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
    
def vbr_spmm_codegen_for_all(density: int = 0):
    if density == 0:
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_SpMM"
    else:
        raise Exception("Not implemented")
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    runtimes = {}
    for filename in os.listdir(input_dir_name):
        assert(filename.endswith(".vbr"))
        core_name = filename[:-len(".vbr")]
        run_time = vbr_spmm_codegen(core_name, density, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes

def vbr_spmm_cuda_codegen_for_all(density: int = 0):
    if density == 0:
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_SpMM_cuda"
    else:
        input_dir_name = "Generated_VBR_Sparse"
        output_dir_name = "Generated_SpMM_cuda_Sparse"
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    runtimes = {}
    for filename in os.listdir(input_dir_name):
        assert(filename.endswith(".vbr"))
        core_name = filename[:-len(".vbr")]
        run_time = vbr_spmm_cuda_codegen(core_name, dir_name=output_dir_name, vbr_dir=input_dir_name, density=density)
        runtimes[core_name] = run_time
    return runtimes

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

def vbr_spmv_cuda_codegen_for_all(density: int = 0):
    if density == 0:
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_SpMV_cuda"
    else:
        input_dir_name = "Generated_VBR_Sparse"
        output_dir_name = "Generated_SpMV_cuda_Sparse"
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    runtimes = {}
    for filename in os.listdir(input_dir_name):
        assert(filename.endswith(".vbr"))
        core_name = filename[:-len(".vbr")]
        run_time = vbr_spmv_cuda_codegen(core_name, dir_name=output_dir_name, vbr_dir=input_dir_name, density=density)
        runtimes[core_name] = run_time
    return runtimes

def gen_single_threaded_spmv_compressed(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, coo_i, coo_j, dir_name, filename, vbr_dir):
    time1 = time.time_ns() // 1_000_000
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbrc")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append("int spmv_kernel(float *y, const float* x, const float* val, int i_start, int i_end, int j_start, int j_end, int val_offset) {\n")
    code.append("\t\tfor (int j = j_start; j < j_end; j++) {\n")
    code.append("\tfor (int i = i_start; i < i_end; i++) {\n")
    code.append("\t\t\ty[i] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n\n")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat* y = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
    code.append(f"\tfloat* x = (float*)calloc({cpntr[-1] + 1}, sizeof(float));\n")
    code.append(f"\tfloat* val = (float*)calloc({len(val) + 1}, sizeof(float));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''
    while (x_size < {0} && fscanf(file2, "%f,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    # code.append("\tint count = 0;\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    count2 = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if count not in ublocks:
                    code.append(f"\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                else:
                    num_elems = indx[count+1] - indx[count]
                    code.append("\n")
                    for i, j, v in zip(coo_i[count2:count2+num_elems], coo_j[count2:count2+num_elems], [elem for elem in range(indx[count],indx[count+1])]):
                        code.append(f"\ty[{i}] += val[{v}] * x[{j}];")
                    code.append("\n")
                    count2 += num_elems
                count+=1
    code.append("\n\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%f\\n\", y[i]);\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append("int spmv_kernel(float *y, const float* x, const float* val, int i_start, int i_end, int j_start, int j_end, int val_offset) {\n")
    code.append("\t\tfor (int j = j_start; j < j_end; j++) {\n")
    code.append("\tfor (int i = i_start; i < i_end; i++) {\n")
    code.append("\t\t\ty[i] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n\n")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat* y = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
    code.append(f"\tfloat* x = (float*)calloc({cpntr[-1] + 1}, sizeof(float));\n")
    code.append(f"\tfloat* val = (float*)calloc({len(val) + 1}, sizeof(float));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''
    while (x_size < {0} && fscanf(file2, "%f,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    # code.append("\tint count = 0;\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if density > 0:
                    sparse_count = 0
                    dense_count = 0
                    count2 = 0
                    # Check if the block is more than 25% dense
                    for _ in range(rpntr[a], rpntr[a+1]):
                        for _ in range(cpntr[b], cpntr[b+1]):
                            if val[indx[count]+count2] == 0.0:
                                sparse_count+=1
                            else:
                                dense_count+=1
                            count2+=1
                    if (dense_count/(dense_count + sparse_count))*100 > density:
                        # code.append(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                        code.append(f"\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    else:
                        code.append(codegen(lambda: spmv(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    # code.append(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    code.append(f"\tspmv_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                count+=1
    code.append("\n\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%f\\n\", y[i]);\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density: int, dir_name: str, filename: str, vbr_dir: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <sys/time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n")
        f.write("#include <pthread.h>\n\n")
        f.write("float *x, *val, *y;\n\n")
        work_per_br = [0]*len(rpntr)
        count = 0
        for a in range(len(rpntr) - 1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if density > 0:
                        sparse_count = 0
                        dense_count = 0
                        count2 = 0
                        for _ in range(rpntr[a], rpntr[a+1]):
                            for _ in range(cpntr[b], cpntr[b+1]):
                                if val[indx[count]+count2] == 0.0:
                                    sparse_count+=1
                                else:
                                    dense_count+=1
                                count2+=1
                        if (dense_count/(dense_count + sparse_count))*100 > density:
                            work_per_br[a] += (rpntr[a+1] - rpntr[a])*(cpntr[b+1] - cpntr[b])
                    else:
                        work_per_br[a] += (rpntr[a+1] - rpntr[a])*(cpntr[b+1] - cpntr[b])
                    count+=1
        del count
        thread_br_map = split_chunks(work_per_br, threads)
        funcount = 0
        for br_list in thread_br_map:
            f.write(f"void *func{funcount}(){{\n")
            for a in br_list:
                if bpntrb[a] == -1:
                    continue
                indx_count = 0
                valid_cols = bindx[bpntrb[a]:bpntre[a]]
                for b in range(len(cpntr)-1):
                    if b in valid_cols:
                        if density>0:
                            sparse_count = 0
                            dense_count = 0
                            count2 = 0
                            for _ in range(rpntr[a], rpntr[a+1]):
                                for _ in range(cpntr[b], cpntr[b+1]):
                                    if val[indx[bpntrb[a]+indx_count]+count2] == 0.0:
                                        sparse_count+=1
                                    else:
                                        dense_count+=1
                                    count2+=1
                            if (dense_count/(dense_count + sparse_count))*100 > density:
                                f.write(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[bpntrb[a]+indx_count]), ArrayVal("x"), ArrayVal("y")))
                            else:
                                f.write(codegen(lambda: spmv(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), ConcreteArrayVal("val", val).slice(indx[bpntrb[a]+indx_count]), ArrayVal("x"), ArrayVal("y")))())
                        else:
                            f.write(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[bpntrb[a]+indx_count]), ArrayVal("x"), ArrayVal("y")))
                        indx_count += 1
            f.write("}\n")
            funcount += 1
        num_working_threads = len(thread_br_map)
        f.write("\n")
        f.write("int main() {\n")
        f.write(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
        f.write("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
        f.write(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
        f.write("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
        f.write(f"\ty = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
        f.write(f"\tx = (float*)calloc({cpntr[-1] + 1}, sizeof(float));\n")
        f.write(f"\tval = (float*)calloc({len(val) + 1}, sizeof(float));\n")
        f.write("\tchar c;\n")
        f.write(f"\tint x_size=0, val_size=0;\n")
        f.write('''
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
    assert(c=='\\n');
    fclose(file1);''')
        f.write('''
    while (x_size < {0} && fscanf(file2, "%f,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
        f.write("\tstruct timeval t1;\n")
        f.write("\tgettimeofday(&t1, NULL);\n")
        f.write("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
        f.write(f"\tpthread_t tid[{num_working_threads}];\n")
        for a in range(num_working_threads):
            f.write(f"\tpthread_create(&tid[{a}], NULL, &func{a}, NULL);\n")
        for a in range(num_working_threads):
            f.write(f"\tpthread_join(tid[{a}], NULL);\n")
        f.write("\tstruct timeval t2;\n")
        f.write("\tgettimeofday(&t2, NULL);\n")
        f.write("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
        f.write("\tfree(x);\n")
        f.write("\tfree(val);\n")
        f.write("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
        f.write(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
        f.write("\t\tprintf(\"%.2f\\n\", y[i]);\n")
        f.write("\t}\n")
        f.write("}\n")

def gen_spmm_libxsmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name: str, filename: str, vbr_dir: str):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rpntr[-1]}x512.matrix")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    code.append("#include <libxsmm_source.h>\n\n")
    code.append("""
int lowestMultiple(int x, int y) {
    if (x % y == 0) {
        return x;
    }
    else if (y % x == 0) {
        return y;
    }
    else {
        return ((x / y) + 1) * y;
    }
}\n""")
    code.append("int main() {\n")
    code.append("\tlibxsmm_init();\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat *y = (float*)aligned_alloc(64, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *x = (float*)aligned_alloc(64, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *val = (float*)aligned_alloc(64, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append('''
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    code.append("\tconst char trans='T';\n")
    code.append("\tconst char notrans='N';\n")
    code.append("\tconst float alpha = 1.0;\n")
    code.append("\tlibxsmm_blasint i_range;\n")
    code.append("\tlibxsmm_blasint j_range;\n")
    code.append("\tconst libxsmm_blasint k_range = 512;\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                code.append(f"\ti_range = {rpntr[a+1] - rpntr[a]};\n")
                code.append(f"\tj_range = {cpntr[b+1] - cpntr[b]};\n")
                code.append(f"\tlibxsmm_sgemm(&notrans, &trans, &k_range, &i_range, &j_range, &alpha, &x[{cpntr[b]*512}], &k_range, &val[{indx[count]}], &i_range, &alpha, &y[{rpntr[a] * 512}], &k_range);\n")
                count+=1
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[i*512 + j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("\tlibxsmm_finalize();\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def gen_spmm_cblas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name: str, filename: str, vbr_dir: str):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rpntr[-1]}x512.matrix")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n")
    code.append("#include <cblas.h>\n\n")
    code.append("""
int lowestMultiple(int x, int y) {
    if (x % y == 0) {
        return x;
    }
    else if (y % x == 0) {
        return y;
    }
    else {
        return ((x / y) + 1) * y;
    }
}\n""")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat *y = (float*)aligned_alloc(64, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *x = (float*)aligned_alloc(64, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *val = (float*)aligned_alloc(64, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append('''
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    code.append("\tconst float alpha = 1.0;\n")
    code.append("\tint i_range;\n")
    code.append("\tint j_range;\n")
    code.append("\tconst int k_range = 512;\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                code.append(f"\ti_range = {rpntr[a+1] - rpntr[a]};\n")
                code.append(f"\tj_range = {cpntr[b+1] - cpntr[b]};\n")
                code.append(f"\tcblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, k_range, i_range, j_range, alpha, &x[{cpntr[b]*512}], k_range, &val[{indx[count]}], i_range, alpha, &y[{rpntr[a] * 512}], k_range);\n")
                count+=1
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[i*512 + j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density: int, dir_name: str, filename: str, vbr_dir: str):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append("""
int lowestMultiple(int x, int y) {
    if (x % y == 0) {
        return x;
    }
    else if (y % x == 0) {
        return y;
    }
    else {
        return ((x / y) + 1) * y;
    }
}\n""")
    code.append("""
int spmm_kernel(float *y, const float* x, const float* val, int i_start, int i_end, int j_start, int j_end, int val_offset) {
	for (int i = i_start; i < i_end; i++) {
		for (int j = j_start; j < j_end; j++) {
			for (int k = 0; k < 512; k++) {
				y[i*512 + k] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[((j*512) + k)]);
			}
		}
	}
}\n\n""")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat *y = (float*)aligned_alloc(64, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *x = (float*)aligned_alloc(64, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *val = (float*)aligned_alloc(64, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append('''
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    # code.append("\tint count = 0;\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if density > 0:
                    sparse_count = 0
                    dense_count = 0
                    count2 = 0
                    # Check if the block is more than 25% dense
                    for _ in range(rpntr[a], rpntr[a+1]):
                        for _ in range(cpntr[b], cpntr[b+1]):
                            if val[indx[count]+count2] == 0.0:
                                sparse_count+=1
                            else:
                                dense_count+=1
                            count2+=1
                    if (dense_count/(dense_count + sparse_count))*100 > density:
                        # code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                        code.append(f"\tspmm_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    else:
                        code.append(codegen(lambda: spmm(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), RepRange(0, 512), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    # code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    code.append(f"\tspmm_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                count+=1
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[i*512 + j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density: int, dir_name: str, filename: str, vbr_dir: str):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rpntr[-1]}x512.matrix")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <string.h>\n")
    code.append("#include <pthread.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append("""
int lowestMultiple(int x, int y) {
    if (x % y == 0) {
        return x;
    }
    else if (y % x == 0) {
        return y;
    }
    else {
        return ((x / y) + 1) * y;
    }
}\n""")
    code.append("float *x, *val, *y;\n\n")
    work_per_br = [0]*len(rpntr)
    count = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if density > 0:
                    sparse_count = 0
                    dense_count = 0
                    count2 = 0
                    for _ in range(rpntr[a], rpntr[a+1]):
                        for _ in range(cpntr[b], cpntr[b+1]):
                            if val[indx[count]+count2] == 0.0:
                                sparse_count+=1
                            else:
                                dense_count+=1
                            count2+=1
                    if (dense_count/(dense_count + sparse_count))*100 > density:
                        work_per_br[a] += (rpntr[a+1] - rpntr[a])*(cpntr[b+1] - cpntr[b])
                else:
                    work_per_br[a] += (rpntr[a+1] - rpntr[a])*(cpntr[b+1] - cpntr[b])
                count+=1
    del count
    thread_br_map = split_chunks(work_per_br, threads)
    funcount = 0
    for br_list in thread_br_map:
        code.append(f"void *func{funcount}(){{\n")
        for a in br_list:
            if bpntrb[a] == -1:
                continue
            indx_count = 0
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if density>0:
                        sparse_count = 0
                        dense_count = 0
                        count2 = 0
                        for _ in range(rpntr[a], rpntr[a+1]):
                            for _ in range(cpntr[b], cpntr[b+1]):
                                if val[indx[bpntrb[a]+indx_count]+count2] == 0.0:
                                    sparse_count+=1
                                else:
                                    dense_count+=1
                                count2+=1
                        if (dense_count/(dense_count + sparse_count))*100 > density:
                            code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[bpntrb[a]+indx_count]), ArrayVal("x"), ArrayVal("y")))
                        else:
                            code.append(codegen(lambda: spmm(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), RepRange(0, 512), ConcreteArrayVal("val", val).slice(indx[bpntrb[a]+indx_count]), ArrayVal("x"), ArrayVal("y")))())
                    else:
                        code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[bpntrb[a]+indx_count]), ArrayVal("x"), ArrayVal("y")))
                    indx_count += 1
        code.append("}\n")
        funcount += 1
    num_working_threads = len(thread_br_map)
    code.append("\n")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\ty = (float*)aligned_alloc(64, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tx = (float*)aligned_alloc(64, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tval = (float*)aligned_alloc(64, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append('''
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(rpntr[-1]))
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    code.append(f"\tpthread_t tid[{num_working_threads}];\n")
    for a in range(num_working_threads):
        code.append(f"\tpthread_create(&tid[{a}], NULL, &func{a}, NULL);\n")
    for a in range(num_working_threads):
        code.append(f"\tpthread_join(tid[{a}], NULL);\n")
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[i*512 + j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def vbr_spmv_cuda_codegen(filename: str, dir_name: str, vbr_dir: str, density: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{rpntr[-1]}.vector")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append('''#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}\n\n''')
    code.append("__global__ void spmv(float *y, float* x, float* val, int i_start, int i_end, int j_start, int j_end, int offset) {\n")
    code.append("\tint i_len = i_end - i_start;\n")
    code.append("\tfor (int i=blockIdx.x * blockDim.x + threadIdx.x; i<i_len; i+=blockDim.x * gridDim.x) {\n")
    code.append("\t\t for (int j=blockIdx.y * blockDim.y + threadIdx.y; j<(j_end - j_start); j+=blockDim.y * gridDim.y) {\n")
    code.append("\t\t\t\ty[i+i_start] += (&val[offset])[(j*i_len) + i] * x[j+j_start];\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n\n")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\tfloat *y, *x, *val;\n")
    code.append(f"\tgpuErrchk(cudaMallocManaged(&y, {rpntr[-1]}*sizeof(float)));\n")
    code.append(f"\tgpuErrchk(cudaMallocManaged(&x, {rpntr[-1] + 1}*sizeof(float)));\n")
    code.append(f"\tgpuErrchk(cudaMallocManaged(&val, {len(val) + 1}*sizeof(float)));\n")
    code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(x, {rpntr[-1] + 1}*sizeof(float), cudaCpuDeviceId));\n")
    code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(val, {len(val) + 1}*sizeof(float), cudaCpuDeviceId));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\twhile (x_size < {0} && fscanf(file2, "%f,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(rpntr[-1]))
    code.append(f"\tint id = cudaGetDevice(&id);\n")
    code.append(f"\tgpuErrchk(cudaMemAdvise(x, {rpntr[-1] + 1}*sizeof(float), cudaMemAdviseSetReadMostly, id));\n")
    code.append(f"\tgpuErrchk(cudaMemAdvise(val, {len(val) + 1}, cudaMemAdviseSetReadMostly, id));\n")
    code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(y, {rpntr[-1]}*sizeof(float), id));\n")
    code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(x, {rpntr[-1] + 1}*sizeof(float), id));\n")
    code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(val, {len(val) + 1}*sizeof(float), id));\n")
    code.append("\tint blockSize, minGridSize, gridSize;\n")
    code.append("\tcudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmv, 0, 0);\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        code.append(f"\tgridSize = ({rpntr[a+1] - rpntr[a]} + blockSize - 1)/blockSize;\n")
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if density > 0:
                    sparse_count = 0
                    dense_count = 0
                    count2 = 0
                    # Check if the block is more than 25% dense
                    for _ in range(rpntr[a], rpntr[a+1]):
                        for _ in range(cpntr[b], cpntr[b+1]):
                            if val[indx[count]+count2] == 0.0:
                                sparse_count+=1
                            else:
                                dense_count+=1
                            count2+=1
                    if dense_count > (density * (sparse_count+dense_count))//100:
                        code.append(f"\tspmv<<<gridSize, blockSize>>>(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        code.append("\tgpuErrchk(cudaPeekAtLastError());\n")
                    # code.append("\tgpuErrchk(cudaDeviceSynchronize());\n")
                    else:
                        code.append(codegen(lambda: spmv(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    code.append(f"\tspmv<<<gridSize, blockSize>>>(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\tgpuErrchk(cudaPeekAtLastError());\n")
                    # code.append("\tgpuErrchk(cudaDeviceSynchronize());\n")
                count+=1
        code.append("\tgpuErrchk(cudaDeviceSynchronize());\n")
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%f\\n\", y[i]);\n")
    code.append("\t}\n")
    code.append("\tgpuErrchk(cudaFree(y));\n")
    code.append("\tgpuErrchk(cudaFree(x));\n")
    code.append("\tgpuErrchk(cudaFree(val));\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".cu"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmm_cuda_codegen_cublas(filename: str, dir_name: str, vbr_dir: str, density: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rpntr[-1]}x512.matrix")
    code = []
    code.append("""#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <assert.h>
#include <sys/time.h>

#define CUDA_CHECK(func)                                                       \\
{                                                                              \\
    cudaError_t status = (func);                                               \\
    if (status != cudaSuccess) {                                               \\
        printf("CUDA API failed at line %d with error: %s (%d)",             \\
               __LINE__, cudaGetErrorString(status), status);                  \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

#define CUBLAS_CHECK(func)                                                       \\
{                                                                              \\
    cublasStatus_t status = (func);                                               \\
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \\
        printf("CUDA API failed at line %d with error: (%d)",             \\
               __LINE__, status);                  \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

int main(void) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    const float alpha = 1.0f;
    const float beta = 1.0f;
""")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\tfloat *y, *x, *val;\n")
    code.append(f"\ty=(float*)malloc({rpntr[-1]*512}*sizeof(float));\n")
    code.append(f"\tx = (float*)malloc({cpntr[-1]*512}*sizeof(float));\n")
    code.append(f"\tval = (float*)malloc({len(val)}*sizeof(float));\n")
    code.append(f"\tfloat *x_d, *val_d, *y_d;\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(rpntr[-1]))
    code.append(f"\tCUDA_CHECK(cudaMalloc((void**)&y_d, {rpntr[-1]*512}*sizeof(float)));\n")
    code.append(f"\tCUDA_CHECK(cudaMalloc((void**)&x_d, {cpntr[-1]*512}*sizeof(float)));\n")
    code.append(f"\tCUDA_CHECK(cudaMalloc((void**)&val_d, {len(val)}*sizeof(float)));\n")
    code.append(f"\tCUDA_CHECK(cudaMemcpyAsync(x_d, x, {cpntr[-1]*512}*sizeof(float), cudaMemcpyHostToDevice, stream));\n")
    code.append(f"\tCUDA_CHECK(cudaMemcpyAsync(val_d, val, {len(val)}*sizeof(float), cudaMemcpyHostToDevice, stream));\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                code.append(f"cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, {rpntr[a+1] - rpntr[a]}, 512, {cpntr[b+1] - cpntr[b]}, &alpha, &val_d[{indx[count]}], {rpntr[a+1] - rpntr[a]}, &x_d[{cpntr[b]*512}], 512, &beta, &y_d[{rpntr[a] * 512}], {rpntr[a+1] - rpntr[a]});\n")
                count+=1
        code.append("\tcudaDeviceSynchronize();\n")
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tCUDA_CHECK(cudaMemcpyAsync(y, y_d, {rpntr[-1]*512}*sizeof(float), cudaMemcpyDeviceToHost, stream));\n")
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tfor (int j=0; j<512; j++) {\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[j*{rpntr[-1]} + i]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("\tCUDA_CHECK(cudaFree(y_d));\n")
    code.append("\tCUDA_CHECK(cudaFree(x_d));\n")
    code.append("\tCUDA_CHECK(cudaFree(val_d));\n")
    code.append("\tfree(y);\n")
    code.append("\tfree(x);\n")
    code.append("\tfree(val);\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".cu"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmm_cuda_codegen(filename: str, dir_name: str, vbr_dir: str, density: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rpntr[-1]}x512.matrix")
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append('''#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}\n\n''')
    code.append("__global__ void spmm(float *y, float* x, float* val, int i_start, int i_end, int j_start, int j_end, int offset) {\n")
    code.append("\tint i_len = i_end - i_start;\n")
    code.append("\tfor (int i=blockIdx.x * blockDim.x + threadIdx.x; i<i_len; i+=blockDim.x * gridDim.x) {\n")
    code.append("\t\t for (int j=blockIdx.y * blockDim.y + threadIdx.y; j<(j_end - j_start); j+=blockDim.y * gridDim.y) {\n")
    code.append("\t\t\t for (int k=0; k<512; k++) {\n")
    code.append("\t\t\t\ty[((i+i_start)*512) + k] += (&val[offset])[(j*i_len) + i] * x[((j+j_start)*512)+k];\n")
    code.append("\t\t\t}\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n\n")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append("\tfloat *y, *x, *val;\n")
    code.append(f"\tgpuErrchk(cudaMallocManaged(&y, {rpntr[-1]*512}*sizeof(float)));\n")
    code.append(f"\tgpuErrchk(cudaMallocManaged(&x, {cpntr[-1]*512}*sizeof(float)));\n")
    code.append(f"\tgpuErrchk(cudaMallocManaged(&val, {len(val)}*sizeof(float)));\n")
    # code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(x, {rpntr[-1] + 1}*sizeof(float), cudaCpuDeviceId));\n")
    # code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(val, {len(val) + 1}*sizeof(float), cudaCpuDeviceId));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append('''\tassert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
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
    assert(c=='\\n');
    fclose(file1);\n''')
    code.append('''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(rpntr[-1]))
    # code.append(f"\tint id = cudaGetDevice(&id);\n")
    # code.append(f"\tgpuErrchk(cudaMemAdvise(x, {rpntr[-1] + 1}*sizeof(float), cudaMemAdviseSetReadMostly, id));\n")
    # code.append(f"\tgpuErrchk(cudaMemAdvise(val, {len(val) + 1}, cudaMemAdviseSetReadMostly, id));\n")
    # code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(y, {rpntr[-1]}*sizeof(float), id));\n")
    # code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(x, {rpntr[-1] + 1}*sizeof(float), id));\n")
    # code.append(f"\tgpuErrchk(cudaMemPrefetchAsync(val, {len(val) + 1}*sizeof(float), id));\n")
    code.append("\tint blockSize, minGridSize, gridSize;\n")
    code.append("\tcudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmm, 0, 0);\n")
    code.append("\tstruct timeval t1;\n")
    code.append("\tgettimeofday(&t1, NULL);\n")
    code.append("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        code.append(f"\tgridSize = ({rpntr[a+1] - rpntr[a]} + blockSize - 1)/blockSize;\n")
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if density > 0:
                    sparse_count = 0
                    dense_count = 0
                    count2 = 0
                    # Check if the block is more than 25% dense
                    for _ in range(rpntr[a], rpntr[a+1]):
                        for _ in range(cpntr[b], cpntr[b+1]):
                            if val[indx[count]+count2] == 0.0:
                                sparse_count+=1
                            else:
                                dense_count+=1
                            count2+=1
                    if dense_count > (density * (sparse_count+dense_count))//100:
                        code.append(f"\tspmm<<<gridSize, blockSize>>>(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                        code.append("\tgpuErrchk(cudaPeekAtLastError());\n")
                    # code.append("\tgpuErrchk(cudaDeviceSynchronize());\n")
                    else:
                        code.append(codegen(lambda: spmm(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), RepRange(0, 512), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    code.append(f"\tspmm<<<gridSize, blockSize>>>(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    code.append("\tgpuErrchk(cudaPeekAtLastError());\n")
                    # code.append("\tgpuErrchk(cudaDeviceSynchronize());\n")
                count+=1
        code.append("\tgpuErrchk(cudaDeviceSynchronize());\n")
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tfor (int j=0; j<512; j++) {\n")
    code.append("\t\t\tprintf(\"%f\\n\", y[i*512+j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("\tgpuErrchk(cudaFree(y));\n")
    code.append("\tgpuErrchk(cudaFree(x));\n")
    code.append("\tgpuErrchk(cudaFree(val));\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".cu"), "w") as f:
        f.writelines(code)
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmv_codegen(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000_000
    if threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000_000
    return time2-time1

def vbr_spmm_codegen(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmm_codegen_libxsmm(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_spmm_libxsmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename, vbr_dir)
    else:
        raise NotImplementedError
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmm_codegen_cblas(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_spmm_cblas(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename, vbr_dir)
    else:
        raise NotImplementedError
    time2 = time.time_ns() // 1_000
    return time2-time1