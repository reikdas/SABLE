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
    dir_name = "Generated_SpMM"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    runtimes = {}
    for filename in os.listdir("Generated_VBR"):
        assert(filename.endswith(".vbr"))
        core_name = filename[:-len(".vbr")]
        run_time = vbr_spmm_codegen(core_name, dir_name=dir_name)
        runtimes[core_name] = run_time
    return runtimes

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

def gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join("Generated_VBR", filename + ".vbr")
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
                code.append(f"\tfor (int j = {cpntr[b]}; j < {cpntr[b+1]}; j++) {{\n")
                code.append(f"\t\tfor (int i = {rpntr[a]}; i < {rpntr[a+1]}; i++) {{\n")
                code.append(f"\t\t\ty[i] += val[{indx[count]}+(j-{cpntr[b]})*({rpntr[a+1]} - {rpntr[a]}) + (i-{rpntr[a]})] * x[j];\n")
                code.append("\t\t}\n")
                code.append("\t}\n")
                count+=1
    code.append("\tstruct timeval t2;\n")
    code.append("\tgettimeofday(&t2, NULL);\n")
    code.append("\tlong t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n")
    code.append("\tprintf(\"{0} = %lu\\n\", t2s-t1s);\n".format(filename))
    code.append(f"\tfor (int i=0; i<{rpntr[-1]}; i++) {{\n")
    code.append("\t\tprintf(\"%f\\n\", y[i]);\n")
    code.append("\t}\n")
    code.append("}\n")
    with open(os.path.join(dir_name, filename+".c"), "w") as f:
        f.writelines(code)

def gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename):
    dir_name = "Generated_SpMV"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vbr_path = os.path.join("Generated_VBR", filename + ".vbr")
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

def gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename, vbr_dir="Generated_VBR"):
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
int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}
int lcm(int a, int b) {
    return (abs(a) / gcd(a, b)) * abs(b);
}\n\n""")
    code.append("int main() {\n")
    code.append(f"\tFILE *file1 = fopen(\"{os.path.abspath(vbr_path)}\", \"r\");\n")
    code.append("\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n")
    code.append(f"\tFILE *file2 = fopen(\"{os.path.abspath(matrix_path)}\", \"r\");\n")
    code.append("\tif (file2 == NULL) { printf(\"Error opening file2\"); return 1; }\n")
    code.append(f"\tfloat *y = (float*)aligned_alloc(64, lcm({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(y, 0, lcm({rpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *x = (float*)aligned_alloc(64, lcm({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(x, 0, lcm({cpntr[-1] * 512}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tfloat *val = (float*)aligned_alloc(64, lcm({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append(f"\tmemset(val, 0, lcm({len(val)}*sizeof(float), 64*sizeof(float)));\n")
    code.append("\tchar c;\n")
    code.append(f"\tint x_size=0, val_size=0;\n")
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
    code.append("\tfloat tmp;\n")
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                code.append(f"\tfor (int i={rpntr[a]}; i<{rpntr[a+1]}; i++) {{\n")
                code.append(f"\t\tfor (int k={cpntr[b]}; k<{cpntr[b+1]}; k++) {{\n")
                code.append(f"\t\t\ttmp=val[{indx[count]}+ (k-{cpntr[b]})*{rpntr[a+1]-rpntr[a]} + (i-{rpntr[a]})];\n")
                code.append(f"\t\t\t#pragma GCC unroll 4\n")
                code.append(f"\t\t\tfor (int j=0; j<512; j++) {{\n")
                code.append(f"\t\t\t\ty[i*512 + j] += tmp* x[k*512 + j];\n")
                code.append("\t\t\t}\n")
                code.append("\t\t}\n")
                code.append("\t}\n")
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

def vbr_spmv_codegen(filename: str, dir_name = "Generated_SpMV", threads=16):
    vbr_path = os.path.join("Generated_VBR", filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename)
    time2 = time.time_ns() // 1_000
    return time2-time1

def vbr_spmm_codegen(filename: str, dir_name: str = "Generated_SpMM", vbr_dir="Generated_VBR", threads=1):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2-time1
