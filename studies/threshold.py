import pathlib
import statistics
import subprocess
import sys

import psutil
from tqdm import tqdm

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.baseline import *
from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from synthesize_matrices.vbr_matrices_gen import vbr_matrix_gen
from utils.convert_real_to_vbr import convert_vbr_to_compressed
from utils.fileio import read_vbr, write_dense_vector
from utils.utils import extract_mul_nums

FILEPATH=pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 60 * 3

DENSE_CODEGEN_DIR = "Generated_SpMV_threshold_Dense"
SPARSE_CODEGEN_DIR = "Generated_SpMV_threshold_Sparse"
DENSE_VBR_DIR = "Generated_VBR_threshold_Dense"
SPARSE_VBR_DIR = "Generated_VBR_threshold_Sparse"

def calculate_threshold():
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    perc_zeros_list = [0, 20, 40, 50, 75, 80, 85, 90, 95, 99]
    dims = [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400]
    mat_side = 10000
    assert (mat_side % dim == 0 for dim in dims)
    write_dense_vector(1.0, mat_side)
    with open(os.path.join(FILEPATH,"threshold_results.csv"), "w") as f:
        f.write(f"dim1,dim2,perc_zeros,nnz,CSR_time,sable_time\n")
        for dim1 in tqdm(dims, desc="Processing dimensions"):
            for dim2 in dims:
                for perc_zeros in tqdm(perc_zeros_list, desc=f"Dim {dim1}: Processing % zeros", leave=False):
                    nnz = (dim1*dim2*(100-perc_zeros))//100
                    fname: str = vbr_matrix_gen(mat_side, mat_side, "uniform", mat_side//dim1, mat_side//dim2, 1, perc_zeros, 0, True, DENSE_VBR_DIR)
                    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(DENSE_VBR_DIR, fname+".vbr"))
                    # print(fname)
                    convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, DENSE_VBR_DIR, 0)
                    vbr_spmv_codegen(fname, dir_name=DENSE_CODEGEN_DIR, vbr_dir=DENSE_VBR_DIR, threads=1, bench=50)
                    try:
                        subprocess.run(["taskset", "-a", "-c", str(core), "gcc", f"{fname}.c", "-o", fname] + CFLAGS, cwd=DENSE_CODEGEN_DIR, check=True, timeout=COMPILE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print("SABLE Dense: Compilation failed for ", fname)
                        continue
                    try:
                        output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"./{fname}"], cwd=DENSE_CODEGEN_DIR).decode("utf-8").split("\n")[0]
                    except subprocess.CalledProcessError:
                        print("SABLE Dense: Execution failed for ", fname)
                        continue
                    output = extract_mul_nums(output)
                    median_sable_time_dense = statistics.median([float(x) for x in output])
                    write_dense_vector(1.0, dim2)
                    fname2: str = vbr_matrix_gen(dim1, dim2, "uniform", 1, 1, 1, perc_zeros, 0, True, SPARSE_VBR_DIR)
                    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(SPARSE_VBR_DIR, fname2+".vbr"))
                    convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname2, SPARSE_VBR_DIR, 100)
                    vbr_spmv_codegen(fname2, dir_name=SPARSE_CODEGEN_DIR, vbr_dir=SPARSE_VBR_DIR, threads=1, bench=50)
                    try:
                        subprocess.run(["taskset", "-a", "-c", str(core), "gcc", f"{fname2}.c", "-o", fname2] + CFLAGS, cwd=SPARSE_CODEGEN_DIR, check=True, timeout=COMPILE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print("SABLE Sparse: Compilation failed for ", fname2)
                        continue
                    try:
                        output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"./{fname2}"], cwd=SPARSE_CODEGEN_DIR).decode("utf-8").split("\n")[0]
                    except subprocess.CalledProcessError as e:
                        print("SABLE Sparse: Execution failed for ", fname2, " with ", e)
                        continue
                    output = extract_mul_nums(output)
                    median_sable_time_sparse = statistics.median([float(x) for x in output])
                    f.write(",".join([str(dim1), str(dim2), str(perc_zeros), str(nnz), str(median_sable_time_sparse), str(median_sable_time_dense)]))
                    f.write("\n")
                    f.flush()

if __name__ == "__main__":
    calculate_threshold()
