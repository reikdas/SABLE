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
    # Find size of auxillary array
    largest_diff = rpntr[1] - rpntr[0]
    for a in range(len(rpntr)-1):
        current_diff = rpntr[a+1] - rpntr[a]
        if current_diff > largest_diff:
            largest_diff = current_diff
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <sys/time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n\n")
        f.write("int main() {\n")
        f.write(f"\tFILE *file = fopen(\"{os.path.abspath(rel_path)}\", \"r\");\n")
        f.write("\tif (file == NULL) { printf(\"Error opening file\"); return 1; }\n")
        f.write(f"\tfloat* y = (float*)calloc({len(x)}, sizeof(float));\n")
        f.write(f"\tfloat* x = (float*)calloc({len(x) + 1}, sizeof(float));\n")
        f.write(f"\tfloat* val = (float*)calloc({len(val) + 1}, sizeof(float));\n")
        f.write("\tchar c;\n")
        f.write(f"\tint x_size=0, val_size=0;\n")
        for variable in ["x", "val"]:
            f.write('''
    assert(fscanf(file, "{0}=[%f", &{0}[{0}_size]) == 1.0);
    {0}_size++;
    while (1) {{
        assert(fscanf(file, \"%c\", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file, \"%f\", &{0}[{0}_size]) == 1.0);
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
        f.write("\tstruct timeval t1;\n")
        f.write("\tgettimeofday(&t1, NULL);\n")
        f.write("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
        f.write("\tint n = omp_get_max_threads();\n")
        f.write("\tint thread_id;\n")
        f.write(f"\tint aux[{largest_diff}*n];\n")
        count = 0
        for a in range(len(rpntr)-1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    f.write(f"\tmemset(aux, 0, sizeof(aux));\n")
                    f.write(f"\t#pragma omp parallel for\n")
                    f.write(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                    f.write(f"\t\tthread_id = omp_get_thread_num();\n")
                    f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\t\taux[(i-{rpntr[a]}) + ({rpntr[a+1]} - {rpntr[a]})*thread_id] += val[{indx[count]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n")
                    f.write(f"\tfor (int i={rpntr[a]}; i<{rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\tfor (int j=0; j<n; j++) {{\n")
                    f.write(f"\t\t\ty[i] += aux[(i-{rpntr[a]})+({rpntr[a+1]} - {rpntr[a]})*j];\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n")
                    count+=1
        f.write("\tstruct timeval t2;\n")
        f.write("\tgettimeofday(&t2, NULL);\n")
        f.write("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
        f.write("\tfree(x);\n")
        f.write("\tfree(val);\n")
        f.write("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
        f.write(f"\tfor (int i=0; i<{len(x)}; i++) {{\n")
        f.write("\t\tprintf(\"%.2f\\n\", y[i]);\n")
        f.write("\t}\n")
        f.write("}\n")
        time2 = time.time_ns() // 1_000
    return time2-time1
