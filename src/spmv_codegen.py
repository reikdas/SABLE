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
        
def vbr_spmv_codegen(filename: str, dir_name = "Generated_SpMV", threads=16):
    rel_path = os.path.join("Generated_Data", filename)
    x, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_data(rel_path)
    filename = filename[:-5]
    time1 = time.time_ns() // 1_000
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <sys/time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n")
        f.write("#include <pthread.h>\n\n")
        f.write("float *x, *val, *y;\n\n")
        funcount = 0
        for a in range(len(rpntr) - 1):
            if bpntrb[a] == -1:
                continue
            valid_cols = bindx[bpntrb[a]:bpntre[a]]
            for b in range(len(cpntr)-1):
                if b in valid_cols:
                    f.write(f"void *func{funcount}(){{\n")
                    f.write(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                    f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\t\ty[i] += val[{indx[funcount]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n")
                    f.write("}\n")
                    funcount += 1
        f.write("\n")
        f.write("int main() {\n")
        f.write(f"\tFILE *file = fopen(\"{os.path.abspath(rel_path)}\", \"r\");\n")
        f.write("\tif (file == NULL) { printf(\"Error opening file\"); return 1; }\n")
        f.write(f"\ty = (float*)calloc({len(x)}, sizeof(float));\n")
        f.write(f"\tx = (float*)calloc({len(x) + 1}, sizeof(float));\n")
        f.write(f"\tval = (float*)calloc({len(val) + 1}, sizeof(float));\n")
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
        f.write(f"\tpthread_t tid[{threads}];\n")
        new_thread_id = threads - 1
        assgn = [-1] * (len(rpntr)-1)
        count = 0
        thread_list = []
        for b in range(len(cpntr) - 1):
            for a in range(len(rpntr)-1):
                if bpntrb[a] == -1:
                    continue
                valid_cols = bindx[bpntrb[a]:bpntre[a]]
                if b in valid_cols:
                    if assgn[a] == -1:
                        if new_thread_id < 0:
                            assgn_thread = thread_list[0]
                            f.write(f"\tpthread_join(tid[{assgn_thread}], NULL);\n")
                            f.write(f"\tpthread_create(&tid[{assgn_thread}], NULL, &func{count}, NULL);\n")
                            assgn[assgn.index(assgn_thread)] = -1
                            assgn[a] = assgn_thread
                            thread_list.pop(0)
                            thread_list.append(assgn_thread)
                        else:
                            f.write(f"\tpthread_create(&tid[{new_thread_id}], NULL, &func{count}, NULL);\n")
                            assgn[a] = new_thread_id
                            thread_list.append(new_thread_id)
                            new_thread_id -= 1
                    else:
                        f.write(f"\tpthread_join(tid[{assgn[a]}], NULL);\n")
                        f.write(f"\tpthread_create(&tid[{assgn[a]}], NULL, &func{count}, NULL);\n")
                        thread_list.remove(assgn[a])
                        thread_list.append(assgn[a])
                    count+=1
        for a in range(len(assgn)):
            if assgn[a] != -1:
                f.write(f"\tpthread_join(tid[{assgn[a]}], NULL);\n")
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
