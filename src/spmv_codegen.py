import os
import time

from src.fileio import read_data

'''
This file contains functionality to generate C code for SpMV for the VBR matrices.
'''

def vbr_spmv_codegen_for_all(bench=False):
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    runtimes = {}
    for filename in os.listdir("Generated_Data"):
        assert(filename.endswith(".data"))
        run_time = vbr_spmv_codegen(filename, dir_name=dir_name)
        runtimes[filename[:-5]] = run_time
        
    return runtimes
        
def vbr_spmv_codegen(filename: str, dir_name = "Generated_SpMV"):
    rel_path = os.path.join("Generated_Data", filename)
    x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(rel_path)
    filename = filename[:-5]

    time1 = time.time_ns() // 1_000
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
    if(fscanf(file, \"%c\", &c));
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
                    f.write(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                    f.write(f"\t\t#pragma omp parallel for reduction(+:y[:{rpntr[a+1]}-{rpntr[a]}])\n")
                    f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\t\ty[i] += val[{indx[count]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];\n")
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

        time2 = time.time_ns() // 1_000
    return time2-time1
