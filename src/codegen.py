import os
import time

from src.fileio import read_vbr

def codegen_for_all(op: str):
    if op.lower() == "spmv":
        return vbr_spmv_codegen_for_all()
    elif op.lower() == "spmm":
        return vbr_spmm_codegen_for_all()
    else:
        raise ValueError("Invalid operation")
    
def vbr_spmm_codegen_for_all():
    pass

def vbr_spmv_codegen_for_all(threads=16):
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    runtimes = {}
    for filename in os.listdir("Generated_VBR"):
        assert(filename.endswith(".vbr"))
        core_name = filename[:-len(".vbr")]
        run_time = vbr_spmv_codegen(core_name, dir_name=dir_name)
        runtimes[core_name] = run_time
    return runtimes

def gen_single_threaded(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename):
    vbr_path = os.path.join("Generated_VBR", filename + ".vbr")
    vector_path = os.path.join("Generated_Vector", filename + ".vector")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.write("#include <stdio.h>\n")
        f.write("#include <sys/time.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <assert.h>\n\n")
        f.write("int main() {\n")
        f.write(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
        f.write("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
        f.write(f"\tFILE *file2 = fopen(\"{os.path.abspath(vector_path)}\", \"r\");\n")
        f.write("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
        f.write(f"\tfloat* y = (float*)calloc({rpntr[-1]}, sizeof(float));\n")
        f.write(f"\tfloat* x = (float*)calloc({rpntr[-1] + 1}, sizeof(float));\n")
        f.write(f"\tfloat* val = (float*)calloc({len(val) + 1}, sizeof(float));\n")
        f.write("\tchar c;\n")
        f.write(f"\tint x_size=0, val_size=0;\n")
        varfile = {"val": "file1", "x": "file2"}
        for var, file in varfile.items():
            f.write('''
    assert(fscanf({1}, "{0}=[%f", &{0}[{0}_size]) == 1.0);
    {0}_size++;
    while (1) {{
        assert(fscanf({1}, \"%c\", &c) == 1);
        if (c == ',') {{
            assert(fscanf({1}, \"%f\", &{0}[{0}_size]) == 1.0);
            {0}_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    if(fscanf({1}, \"%c\", &c));
    assert(c=='\\n');\n
    fclose({1});\n'''.format(var, file))
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
                    f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\t\ty[i] += val[{indx[count]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n")
                    count+=1
        f.write("\tstruct timeval t2;\n")
        f.write("\tgettimeofday(&t2, NULL);\n")
        f.write("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
        f.write("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
        f.write(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
        f.write("\t\tprintf(\"%f\\n\", y[i]);\n")
        f.write("\t}\n")
        f.write("}\n")

def gen_multi_threaded(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename):
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join("Generated_VBR", filename + ".vbr")
    vector_path = os.path.join("Generated_Vector", filename + ".vector")
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
                    f.write(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                    f.write(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                    f.write(f"\t\t\ty[i] += val[{indx[count]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n")
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
        varfile = {"val": "file1", "x": "file2"}
        for var, file in varfile.items():
            f.write('''
    assert(fscanf({1}, "{0}=[%f", &{0}[{0}_size]) == 1.0);
    {0}_size++;
    while (1) {{
        assert(fscanf({1}, \"%c\", &c) == 1);
        if (c == ',') {{
            assert(fscanf({1}, \"%f\", &{0}[{0}_size]) == 1.0);
            {0}_size++;
        }} else if (c == ']') {{
            break;
        }} else {{
            assert(0);
        }}
    }}
    if(fscanf({1}, \"%c\", &c));
    assert(c=='\\n');\n
    fclose({1});\n'''.format(var, file))
        f.write("\tstruct timeval t1;\n")
        f.write("\tgettimeofday(&t1, NULL);\n")
        f.write("\tlong t1s = t1.tv_sec * 1000000L + t1.tv_usec;\n")
        f.write(f"\tpthread_t tid[{threads}];\n")
        for a in range(threads):
            f.write(f"\tpthread_create(&tid[{a}], NULL, &func{a}, NULL);\n")
        for a in range(threads):
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

def vbr_spmv_codegen(filename: str, dir_name = "Generated_SpMV", threads=16):
    vbr_path = os.path.join("Generated_VBR", filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    filename = filename[:-5]
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename)
    else:
        gen_multi_threaded(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename)
    time2 = time.time_ns() // 1_000
    return time2-time1