import os
import pathlib

from utils.fileio import read_vbr

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")


def only_nonzeros_spmv(filename: str, dir_name: str, vbr_dir: str):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cpntr[-1]}.vector")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
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
        assert(fscanf(file2, "%f,", &x[i]) == 1);
    }}
    fclose(file2);\n'''.format(cpntr[-1]))
    code.append("\tstruct timespec t1;\n")
    code.append("\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                for i in range(rpntr[a], rpntr[a+1]):
                    for j in range(cpntr[b], cpntr[b+1]):
                        v_idx = indx[count] + ((j-cpntr[b])*(rpntr[a+1]-rpntr[a])) + (i-rpntr[a])
                        v = val[v_idx]
                        if v != 0:
                            code.append(f"\ty[{i}] += val[{v_idx}] * x[{j}];\n")
                count += 1
    code.append("\tstruct timespec t2;\n")
    code.append("\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
    code.append("\tprintf(\"{0} = %lu\\n\", (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec));\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[i]);\n")
    code.append("\t\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def only_nonzeros_spmm(filename: str, dir_name: str, vbr_dir: str):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{rpntr[-1]}x512.matrix")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    code = []
    code.append("#include <stdio.h>\n")
    code.append("#include <time.h>\n")
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
    fclose(file2);\n'''.format(cpntr[-1]))
    code.append("\tstruct timespec t1;\n")
    code.append("\tclock_gettime(CLOCK_MONOTONIC, &t1);\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                for i in range(rpntr[a], rpntr[a+1]):
                    for j in range(cpntr[b], cpntr[b+1]):
                        v_idx = indx[count] + ((j-cpntr[b])*(rpntr[a+1]-rpntr[a])) + (i-rpntr[a])
                        v = val[v_idx]
                        if v != 0:
                            code.append("\tfor (int k = 0; k < 512; k++) {\n")
                            code.append(f"\t\ty[{i*512}+k] += val[{v_idx}] * x[{j*512}+k];\n")
                            code.append("\t}\n")
                count += 1
    code.append("\tstruct timespec t2;\n")
    code.append("\tclock_gettime(CLOCK_MONOTONIC, &t2);\n")
    code.append("\tprintf(\"{0} = %lu\\n\", (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec));\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append(f"\t\tfor (int j=0; j<512; j++) {{\n")
    code.append(f"\t\t\tprintf(\"%f\\n\", y[i*512 + j]);\n")
    code.append("\t\t}\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)
