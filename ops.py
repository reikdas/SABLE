import os
import time
from typing import Any, Callable, Union

from src.codegen import (ArrayVal, ConcreteArrayVal, NumVal, RepRange,
                         gen_multi_threaded_codegen,
                         gen_single_threaded_codegen, get_val, isDense, it)
from src.fileio import read_vbr


def it2(r1: Union[range, RepRange], r2: Union[range, RepRange], f: Callable[[NumVal, NumVal], Any]):
    return it(r1, lambda i: it(r2, lambda j: f(i, j)))


def it3(
    r1: Union[range, RepRange], r2: Union[range, RepRange], r3: Union[range, RepRange], f: Callable[[NumVal, NumVal, NumVal], Any]
):
    return it(r1, lambda i: it(r2, lambda j: it(r3, lambda k: f(i, j, k))))


def spmv(
    row_idxs: Union[range, RepRange],
    col_idxs: Union[range, RepRange],
    col_maj_val: Union[ArrayVal, ConcreteArrayVal],
):
    x = ArrayVal("x")
    y = ArrayVal("y")

    def op(j: NumVal, i: NumVal):
        val = get_val(row_idxs, col_idxs, col_maj_val, i, j)
        if isDense(val):
            y[i] += val * x[j]

    return it2(col_idxs, row_idxs, op)


def spmm(
    row_idxs: Union[range, RepRange],
    col_idxs: Union[range, RepRange],
    col_maj_val: Union[ArrayVal, ConcreteArrayVal],
):
    x = ArrayVal("x")
    y = ArrayVal("y")

    def op(i: NumVal, k: NumVal, j: NumVal):
        val = get_val(row_idxs, col_idxs, col_maj_val, i, k)
        if isDense(val):
            y[i * 512 + j] += val * x[k * 512 + j]

    return it3(row_idxs, col_idxs, RepRange(0, 512), op)


def gen_single_threaded_spmv(
    val: list[Union[int, float]],
    indx: list[int],
    bindx: list[int],
    rpntr: list[int],
    cpntr: list[int],
    bpntrb: list[int],
    bpntre: list[int],
    density: int,
    dir_name: str,
    filename: str,
    vbr_dir: str,
):
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
    code.append(f'\tFILE *file1 = fopen("{os.path.abspath(vbr_path)}", "r");\n')
    code.append('\tif (file1 == NULL) { printf("Error opening file1"); return 1; }\n')
    code.append(f'\tFILE *file2 = fopen("{os.path.abspath(vector_path)}", "r");\n')
    code.append('\tif (file2 == NULL) { printf("Error opening file2"); return 1; }\n')
    code.append(f"\tfloat* y = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
    code.append(f"\tfloat* x = (float*)calloc({rpntr[-1] + 1}, sizeof(float));\n")
    code.append(f"\tfloat* val = (float*)calloc({len(val) + 1}, sizeof(float));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
    code.append(
        """\tassert(fscanf(file1, "val=[%f", &val[val_size]) == 1.0);
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
    fclose(file1);\n"""
    )
    code.append(
        """
    while (x_size < {0} && fscanf(file2, "%f,", &x[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n""".format(
            rpntr[-1]
        )
    )
    code.extend(gen_single_threaded_codegen(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, filename, spmv))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append('\t\tprintf("%f\\n", y[i]);\n')
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)


def gen_multi_threaded_spmv(
    threads: int,
    val: list[Union[int, float]],
    indx: list[int],
    bindx: list[int],
    rpntr: list[int],
    cpntr: list[int],
    bpntrb: list[int],
    bpntre: list[int],
    density: int,
    dir_name: str,
    filename: str,
    vbr_dir: str,
):
    assert threads > 0
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    vector_path = f"generated_vector_{rpntr[-1]}.vector"
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <sys/time.h>\n")
    code.append("#include <stdlib.h>\n")
    code.append("#include <assert.h>\n")
    code.append("#include <pthread.h>\n\n")
    code.append("float *x, *val, *y;\n\n")
    code.append("int init() {\n")
    code.append(f'\tFILE *file1 = fopen("{os.path.abspath(vbr_path)}", "r");\n')
    code.append('\tif (file1 == NULL) { printf("Error opening file1"); return 1; }\n')
    code.append(f'\tFILE *file2 = fopen("{os.path.abspath(vector_path)}", "r");\n')
    code.append('\tif (file2 == NULL) { printf("Error opening file2"); return 1; }\n')
    code.append(f"\ty = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
    code.append(f"\tx = (float*)calloc({rpntr[-1] + 1}, sizeof(float));\n")
    code.append(f"\tval = (float*)calloc({len(val) + 1}, sizeof(float));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
    code.append(
        """
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
fclose(file1);"""
    )
    code.append(
        """
while (x_size < {0} && fscanf(file2, "%f,", &x[x_size]) == 1) {{
    x_size++;
}}
fclose(file2);\n""".format(
            rpntr[-1]
        )
    )
    code.append("}\n")
    code.extend(gen_multi_threaded_codegen(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, filename, spmv))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append('\t\tprintf("%.2f\\n", y[i]);\n')
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)


def gen_single_threaded_spmm(
    val: list[Union[int, float]],
    indx: list[int],
    bindx: list[int],
    rpntr: list[int],
    cpntr: list[int],
    bpntrb: list[int],
    bpntre: list[int],
    density: int,
    dir_name: str,
    filename: str,
    vbr_dir: str,
):
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
    code.append(
        """
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
}\n"""
    )
    code.append("int main() {\n")
    code.append(f'\tFILE *file1 = fopen("{os.path.abspath(vbr_path)}", "r");\n')
    code.append('\tif (file1 == NULL) { printf("Error opening file1"); return 1; }\n')
    code.append(f'\tFILE *file2 = fopen("{os.path.abspath(matrix_path)}", "r");\n')
    code.append('\tif (file2 == NULL) { printf("Error opening file2"); return 1; }\n')
    code.append(f"\tfloat *y = (float*)aligned_alloc(64, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *x = (float*)aligned_alloc(64, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *val = (float*)aligned_alloc(64, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append(
        """
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
    fclose(file1);\n"""
    )
    code.append(
        """\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n""".format(
            rpntr[-1]
        )
    )
    code.extend(gen_single_threaded_codegen(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, filename, spmm))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f'\t\t\tprintf("%f\\n", y[i*512 + j]);\n')
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)


def gen_multi_threaded_spmm(
    threads: int,
    val: list[Union[int, float]],
    indx: list[int],
    bindx: list[int],
    rpntr: list[int],
    cpntr: list[int],
    bpntrb: list[int],
    bpntre: list[int],
    density: int,
    dir_name: str,
    filename: str,
    vbr_dir: str,
):
    assert threads > 0
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
    code.append(
        """
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
}\n"""
    )
    code.append("float *x, *val, *y;\n\n")
    code.append("int init() {\n")
    code.append(f'\tFILE *file1 = fopen("{os.path.abspath(vbr_path)}", "r");\n')
    code.append('\tif (file1 == NULL) { printf("Error opening file1"); return 1; }\n')
    code.append(f'\tFILE *file2 = fopen("{os.path.abspath(matrix_path)}", "r");\n')
    code.append('\tif (file2 == NULL) { printf("Error opening file2"); return 1; }\n')
    code.append(f"\ty = (float*)aligned_alloc(64, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lowestMultiple({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tx = (float*)aligned_alloc(64, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lowestMultiple({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tval = (float*)aligned_alloc(64, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lowestMultiple({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint val_size=0;\n")
    code.append(
        """
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
    fclose(file1);\n"""
    )
    code.append(
        """\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &x[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n""".format(
            rpntr[-1]
        )
    )
    code.append("}\n")
    code.extend(gen_multi_threaded_codegen(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, filename, spmm))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f'\t\t\tprintf("%f\\n", y[i*512 + j]);\n')
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename + ".c"), "w") as f:
        f.writelines(code)


def vbr_spmv_codegen(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2 - time1


def vbr_spmm_codegen(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2 - time1


def vbr_spmv_codegen_for_all(density: int):
    if density == 0:
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_SpMV"
    else:
        raise Exception("Not implemented")
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    runtimes = {}
    for filename in os.listdir(input_dir_name):
        assert filename.endswith(".vbr")
        core_name = filename[: -len(".vbr")]
        run_time = vbr_spmv_codegen(core_name, density, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes


def vbr_spmm_codegen_for_all(density):
    if density == 0:
        input_dir_name = "Generated_VBR"
        output_dir_name = "Generated_SpMM"
    else:
        raise Exception("Not implemented")
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    runtimes = {}
    for filename in os.listdir(input_dir_name):
        assert filename.endswith(".vbr")
        core_name = filename[: -len(".vbr")]
        run_time = vbr_spmm_codegen(core_name, density, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes
