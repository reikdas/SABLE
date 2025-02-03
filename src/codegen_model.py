import os
import pathlib
import time
import joblib

from utils.fileio import read_vbr
from src.codegen import codegen, spmm, ArrayVal, ConcreteArrayVal, RepRange, split_chunks

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")
    
def vbr_spmm_codegen_for_all(density: int = 0):
    if density == 0:
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
        run_time = vbr_spmm_codegen(core_name, density, dir_name=output_dir_name, vbr_dir=input_dir_name, threads=1)
        runtimes[core_name] = run_time
    return runtimes


def gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density: int, dir_name: str, filename: str, vbr_dir: str):
    model = joblib.load(os.path.join(BASE_PATH, "models", "density_threshold_spmm.pkl"))
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    matrix_path = os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_matrix_{cpntr[-1]}x512.matrix")
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
    code.append("""
int spmm_kernel(float *y, const float* x, const float* val, int i_start, int i_end, int j_start, int j_end, int val_offset) {
	for (int i = i_start; i < i_end; i++) {
		for (int j = j_start; j < j_end; j++) {
			for (int k = 0; k < 512; k++) {
				y[i*512 + k] += ((&val[val_offset])[(((j-j_start)*(i_end-i_start)) + (i-i_start))] * x[((j*512) + k)]);
			}
		}
	}
}\n\n""")
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
    # code.append("\tint count = 0;\n")
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
                if density > 0:
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
                    calc_density = (dense_count/(dense_count + sparse_count))*100
                    block_size = (rpntr[a+1] - rpntr[a])*(cpntr[b+1] - cpntr[b])
                    if model.predict([[block_size, calc_density]])[0] == 1:
                        # code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                        code.append(f"\tspmm_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
                    else:
                        code.append(codegen(lambda: spmm(range(rpntr[a], rpntr[a+1]), range(cpntr[b], cpntr[b+1]), RepRange(0, 512), ConcreteArrayVal("val", val).slice(indx[count]), ArrayVal("x"), ArrayVal("y")))())
                else:
                    # code.append(codegen(spmm)(RepRange(rpntr[a], rpntr[a+1]), RepRange(cpntr[b], cpntr[b+1]), RepRange(0, 512), ArrayVal("val").slice(indx[count]), ArrayVal("x"), ArrayVal("y")))
                    code.append(f"\tspmm_kernel(y, x, val, {rpntr[a]}, {rpntr[a+1]}, {cpntr[b]}, {cpntr[b+1]}, {indx[count]});\n")
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

def gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density: int, dir_name: str, filename: str, vbr_dir: str):
    raise NotImplementedError

def vbr_spmm_codegen(filename: str, density: int, dir_name: str, vbr_dir: str, threads: int):
    vbr_path = os.path.join(vbr_dir, filename + ".vbr")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
    time1 = time.time_ns() // 1_000
    if threads == 1:
        gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    else:
        gen_multi_threaded_spmm(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, density, dir_name, filename, vbr_dir)
    time2 = time.time_ns() // 1_000
    return time2-time1