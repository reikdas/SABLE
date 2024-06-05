import inspect
import itertools
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Union

from src.fileio import read_vbr

curr_block_instructions = []

def _codegen(f):
    global curr_block_instructions
    prev = curr_block_instructions
    try:
        curr_block_instructions = []
        f()
        if len(curr_block_instructions) > 0:
            res = ";\n".join(str(e) for e in curr_block_instructions if e!="")
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
        curr_block_instructions.append(f"float* {n} = &({self.name}[{start}])")
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
    
def vbr_spmm_codegen_for_all(dense_blocks_only: bool = True):
    if dense_blocks_only:
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
        run_time = vbr_spmm_codegen(core_name, dense_blocks_only, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes

def vbr_spmm_cuda_codegen_for_all(dense_blocks_only: bool):
    if dense_blocks_only:
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
        run_time = vbr_spmm_cuda_codegen(core_name, dir_name=output_dir_name, vbr_dir=input_dir_name, dense_blocks_only=dense_blocks_only)
        runtimes[core_name] = run_time
    return runtimes

def vbr_spmv_codegen_for_all(dense_blocks_only: bool = True):
    if dense_blocks_only:
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
        run_time = vbr_spmv_codegen(core_name, dense_blocks_only, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes

def vbr_spmv_cuda_codegen_for_all(dense_blocks_only: bool):
    if dense_blocks_only:
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
        run_time = vbr_spmv_cuda_codegen(core_name, dir_name=output_dir_name, vbr_dir=input_dir_name, dense_blocks_only=dense_blocks_only)
        runtimes[core_name] = run_time
    return runtimes

def gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = f"generated_vector_{rpntr[-1]}.vector"
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <assert.h>\n\n")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat* y = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
    code.append(f"\tfloat* x = (float*)calloc({rpntr[-1] + 1}, sizeof(float));\n")
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
    fclose(file2);\n'''.format(rpntr[-1]))
    code.append("\tint count = 0;\n")
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
                if (not dense_blocks_only):
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
                    if dense_count > 100:
                        code.append(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    else:
                        code.append(codegen(lambda: spmv(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    code.append(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
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

def gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = f"generated_vector_{rpntr[-1]}.vector"
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <sys/time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n")
        f.write("#include <pthread.h>\n\n")
        f.write("float *x, *val, *y;\n\n")
        preemptive_count = 0
        for a in range(len(rpntr) - 1):
            if bpntrb[a] == -1:
                continue
            preemptive_count += 1
        per_func = [0]*threads
        for i in range(preemptive_count):
            per_func[i % threads] += 1
        num_working_threads = sum(1 for element in per_func if element != 0)
        funcount = 0
        func_idx = 0
        count = 0
        for a in range(len(rpntr) - 1):
            if bpntrb[a] == -1:
                if a == len(rpntr) - 2 and func_idx!=0:
                    f.write("}\n")
                continue
            if func_idx == 0:
                f.write(f"void *func{funcount}(){{\n")
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    if (not dense_blocks_only):
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
                        if dense_count > 100:
                            f.write(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                        else:
                            f.write(codegen(lambda: spmv(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                    else:
                        f.write(codegen(spmv)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    count+=1
            func_idx += 1
            if func_idx == per_func[funcount] or a == len(rpntr) - 2:
                funcount += 1
                func_idx = 0
                f.write("}\n")
        f.write("\n")
        f.write("int main() {\n")
        f.write(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
        f.write("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
        f.write(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
        f.write("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
        f.write(f"\ty = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
        f.write(f"\tx = (float*)calloc({rpntr[-1] + 1}, sizeof(float));\n")
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
    fclose(file2);\n'''.format(rpntr[-1]))
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

def gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = f"generated_matrix_{rpntr[-1]}x512.matrix"
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
    fclose(file2);\n'''.format(rpntr[-1]))
    code.append("\tint count = 0;\n")
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
                if (not dense_blocks_only):
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
                    if dense_count > (15 * (sparse_count+dense_count))//100:
                        code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    else:
                        code.append(codegen(lambda: spmm(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), RepRange(0, 512), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
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

def gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = f"generated_matrix_{rpntr[-1]}x512.matrix"
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
    preemptive_count = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            continue
        preemptive_count += 1
    per_func = [0]*threads
    for i in range(preemptive_count):
        per_func[i % threads] += 1
    num_working_threads = sum(1 for element in per_func if element != 0)
    funcount = 0
    func_idx = 0
    count = 0
    for a in range(len(rpntr) - 1):
        if bpntrb[a] == -1:
            if a == len(rpntr) - 2 and func_idx!=0:
                code.append("}\n")
            continue
        if func_idx == 0:
            code.append(f"void *func{funcount}(){{\n")
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if (not dense_blocks_only):
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
                    if dense_count > (15 * (sparse_count+dense_count))//100:
                        code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    else:
                        code.append(codegen(lambda: spmm(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), RepRange(0, 512), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                count+=1
        func_idx += 1
        if func_idx == per_func[funcount] or a == len(rpntr) - 2:
            funcount += 1
            func_idx = 0
            code.append("}\n")
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

def vbr_spmv_cuda_codegen(filename: str, dir_name: str, vbr_dir: str, dense_blocks_only: bool):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = f"generated_vector_{rpntr[-1]}.vector"
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
                if (not dense_blocks_only):
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
                    if dense_count > (15 * (sparse_count+dense_count))//100:
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

def vbr_spmm_cuda_codegen(filename: str, dir_name: str, vbr_dir: str, dense_blocks_only: bool):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = f"generated_matrix_{rpntr[-1]}x512.matrix"
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
                if (not dense_blocks_only):
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
                    if dense_count > (15 * (sparse_count+dense_count))//100:
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

def vbr_spmv_codegen(filename: str, dense_blocks_only: bool, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmm_codegen(filename: str, dense_blocks_only: bool, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dense_blocks_only, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2-time1
