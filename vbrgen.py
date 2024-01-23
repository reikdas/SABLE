import random
import numpy
import sys
import os
import time

from src.fileio import read_data
from src.gen_vbr_matrices import generate_vbr_matrices

def spmv_codegen(bench=False):
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    d = {}
    for filename in os.listdir("Generated_Data"):
        assert(filename.endswith(".data"))
        rel_path = os.path.join("Generated_Data", filename)
        x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(rel_path)
        filename = filename[:-5]
        if bench:
            time1 = time.time_ns() // 1_000_000
        with open(os.path.join(dir_name, filename+".c"), "w") as f:
            f.write("#include <stdio.h>\n")
            f.write("#include <sys/time.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write("#include <assert.h>\n\n")
            f.write("int main() {\n")
            f.write(f"\tFILE *file = fopen(\"{os.path.abspath(rel_path)}\", \"r\");\n")
            f.write("\tif (file == NULL) { printf(\"Error opening file\"); return 1; }\n")
            f.write(f"\tint* y = (int*)calloc({len(x)}, sizeof(int));\n")
            f.write(f"\tint* x = (int*)calloc({len(x) + 1}, sizeof(int));\n")
            f.write(f"\tint* val = (int*)calloc({len(val) + 1}, sizeof(int));\n")
            f.write(f"\tint* indx = (int*)calloc({len(indx) + 1}, sizeof(int));\n")
            f.write(f"\tint* bindx = (int*)calloc({len(bindx) + 1}, sizeof(int));\n")
            f.write(f"\tint* rpntr = (int*)calloc({len(rpntr) + 1}, sizeof(int));\n")
            f.write(f"\tint* cpntr = (int*)calloc({len(cpntr) + 1}, sizeof(int));\n")
            f.write(f"\tint* bpntrb = (int*)calloc({len(bpntrb) + 1}, sizeof(int));\n")
            f.write(f"\tint* bpntre = (int*)calloc({len(bpntre) + 1}, sizeof(int));\n")
            f.write("\tchar c;\n")
            f.write(f"\tint x_size=0, val_size=0, indx_size=0, bindx_size=0, rpntr_size=0, cpntr_size=0, bpntrb_size=0, bpntre_size=0;\n")
            for variable in ["x", "val", "indx", "bindx", "rpntr", "cpntr", "bpntrb", "bpntre"]:
                f.write('''
        assert(fscanf(file, "{0}=[%d", &{0}[{0}_size]) == 1);
        {0}_size++;
        while (1) {{
            assert(fscanf(file, \"%c\", &c) == 1);
            if (c == ',') {{
                assert(fscanf(file, \"%d\", &{0}[{0}_size]) == 1);
                {0}_size++;
            }} else if (c == ']') {{
                break;
            }} else {{
                assert(0);
            }}
        }}
        fscanf(file, \"%c\", &c);
        assert(c=='\\n');\n'''.format(variable))
            f.write("\tfclose(file);\n")
            f.write("\tint count = 0;\n")
            f.write("\tstruct timeval t1;\n")
            f.write("\tgettimeofday(&t1, NULL);\n")
            f.write("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
            count = 0
            for a in range(len(rpntr)-1):
                if bpntrb[a] == -1:
                    continue
                valid_cols = bindx[bpntrb[a]:bpntre[a]]
                for b in range(len(cpntr)-1):
                    if b in valid_cols:
                        f.write("\tcount = 0;\n")
                        f.write(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                        f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                        f.write(f"\t\t\ty[i] += val[{indx[count]}+count] * x[j];\n")
                        f.write("\t\t\tcount++;\n")
                        f.write("\t\t}\n")
                        f.write("\t}\n")
                        count+=1
            f.write("\tstruct timeval t2;\n")
            f.write("\tgettimeofday(&t2, NULL);\n")
            f.write("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
            f.write("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
            f.write(f"\tfor (int i=0; i<{len(x)}; i++) {{\n")
            f.write("\t\tprintf(\"%d\\n\", y[i]);\n")
            f.write("\t}\n")
            f.write("}\n")
        if bench:
            time2 = time.time_ns() // 1_000_000
            d[filename] = time2-time1
    if bench:
        return d

def write_mm_file(filename, M):
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{M.shape[0]} {M.shape[1]} {numpy.count_nonzero(M)}\n")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i][j] != 0:
                    f.write(f"{i+1} {j+1} {M[i][j]}\n")

def find_nonneg(l):
    for _, ele in enumerate(l):
        if ele != -1:
            return ele
    assert(False)
    return -1

def mtx_gen():
    dir_name = "Generated_Matrix"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for filename in os.listdir("Generated_Data"):
        assert(filename.endswith(".data"))
        x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(os.path.join("Generated_Data", filename))
        filename = filename[:-5]
        M = numpy.zeros((rpntr[-1], cpntr[-1]))
        count = 0
        bpntrb_start = find_nonneg(bpntrb)
        for a in range(len(rpntr) - 1):
            valid_cols = bindx[bpntrb[a]-bpntrb_start:bpntre[a]-bpntrb_start]
            for b in range(len(cpntr) - 1):
                if b in valid_cols:
                    count2 = 0
                    for j in range(cpntr[b], cpntr[b+1]):
                        for i in range(rpntr[a], rpntr[a+1]):
                            M[i][j] = val[indx[count]+count2]
                            count2 += 1
                    count += 1
        write_mm_file(os.path.join(dir_name, filename + ".mtx"), M)

def random_splits(n, a):
    if n <= 0 or a <= 0:
        raise ValueError("Both n and a must be positive integers.")

    # Generate a list of 'a-1' random numbers between 1 and n-1
    split_numbers = sorted(random.sample(range(1, n), a - 1))

    # Calculate the differences between consecutive split numbers
    differences = [split_numbers[0]] + [split_numbers[i] - split_numbers[i - 1] for i in range(1, a - 1)] + [n - split_numbers[-1]]

    return differences


if __name__ == "__main__":
    generate_vbr_matrices()
    mtx_gen()
