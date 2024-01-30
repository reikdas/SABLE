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
        f.write(c_header(x, val, rel_path))
        for v in ["x", "val"]:
            f.write(c_scan(v))
        f.write(c_end_scan())
        count = 0
        for a in range(len(rpntr)-1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    f.write(c_compute(count, a, b, indx, cpntr, rpntr))
                    count+=1
        f.write(c_time(x, filename))

        time2 = time.time_ns() // 1_000
    return time2-time1

# f-strings (below) require {{ / }} instead of { / } because unary braces hold Python code
# Also, \ has to be escaped: \\
# Otherwise it's literal C
#
def c_header(x, val, rel_path):
       return f"""\
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>

int main() {{
    FILE *file = fopen("{os.path.abspath(rel_path)}", "r");
    if (file == NULL) {{ printf("Error opening file"); return 1; }}
    int* y = (int*)calloc({len(x)}, sizeof(int));
    int* x = (int*)calloc({len(x) + 1}, sizeof(int));
    int* val = (int*)calloc({len(val) + 1}, sizeof(int));
    char c;
    int x_size=0, val_size=0;
"""

def c_scan(v):
    return f"""
    assert(fscanf(file, "{v}=[%d", &{v}[{v}_size]) == 1);
    {v}_size++;
    while (1) {{
        assert(fscanf(file, "%c", &c) == 1);
        if (c == ',') {{
            assert(fscanf(file, "%d", &{v}[{v}_size]) == 1);
            {v}_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    if(fscanf(file, "%c", &c));
    assert(c=='\\n');
"""

def c_end_scan():
    return """\
    fclose(file);
    int count = 0;
    struct timeval t1;
    gettimeofday(&t1, NULL);
    long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
"""

def c_compute(count, a, b, indx, cpntr, rpntr):
    return f"""\
    for (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{
        #pragma omp parallel for reduction(+:y[:{rpntr[a+1]}-{rpntr[a]}])
        for (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{
            y[i] += val[{indx[count]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];
        }}
    }}
"""

def c_time(x, filename):
    return f"""\
    struct timeval t2;
    gettimeofday(&t2, NULL);
    long t2s = t2.tv_sec * 1000000L + t2.tv_usec;
    printf("{filename} = %lu\\n", t2s-t1s);
    for (int i=0; i<{len(x)}; i++) {{
        printf("%d\\n", y[i]);
    }}
}}
"""
