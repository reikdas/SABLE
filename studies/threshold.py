import pathlib
import statistics
import subprocess
import sys

import psutil
from tqdm import tqdm

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from utils.convert_real_to_vbr import vbrc_matrix_gen, _generate_vbrc_filename
from utils.fileio import write_dense_vector
from utils.utils import extract_mul_nums

FILEPATH=pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 1000
COMPILE_TIMEOUT = 60 * 60 * 3

DENSE_CODEGEN_DIR = os.path.join(FILEPATH, "..", "Generated_SpMV_threshold_Dense")
SPARSE_CODEGEN_DIR = os.path.join(FILEPATH, "..", "Generated_SpMV_threshold_Sparse")
DENSE_VBR_DIR = os.path.join(FILEPATH, "..", "Generated_VBR_threshold_Dense")
SPARSE_VBR_DIR = os.path.join(FILEPATH, "..", "Generated_VBR_threshold_Sparse")

def calculate_threshold():
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    perc_zeros_list = [0, 20, 40, 50, 75, 80, 85, 90, 95, 99]
    dims = [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 625, 1000, 1250, 2000, 2500, 5000]
    # mat_side = 10000
    # assert (mat_side % dim == 0 for dim in dims)
    # write_dense_vector(1.0, mat_side)
    with open(os.path.join(FILEPATH,"threshold_results.csv"), "w") as f:
        f.write(f"dim1,dim2,perc_zeros,nnz,CSR_time,sable_time\n")
        for dim1 in tqdm(reversed(dims), desc="Processing dimensions"):
            for dim2 in tqdm(reversed(dims), desc=f"Dim {dim1}: Processing dimensions", leave=False):
                write_dense_vector(1.0, dim2)
                for perc_zeros in tqdm(perc_zeros_list, desc=f"Dim {dim2}: Processing % zeros", leave=False):
                    nnz = (dim1*dim2*(100-perc_zeros))//100
                    fname: str = vbrc_matrix_gen(dim1, dim2, "uniform", 1, 1, 1, perc_zeros, 0, True, DENSE_VBR_DIR, 0)
                    vbr_spmv_codegen_python(fname, dir_name=DENSE_CODEGEN_DIR, vbr_dir=DENSE_VBR_DIR, bench=BENCHMARK_FREQ)
                    try:
                        output = subprocess.check_output(["taskset", "-a", "-c", str(core), "python3", f"{fname}.py"], cwd=DENSE_CODEGEN_DIR).decode("utf-8").split("\n")[0]
                    except subprocess.CalledProcessError:
                        print("SABLE Dense: Execution failed for ", fname)
                        continue
                    output = extract_mul_nums(output)
                    median_sable_time_dense = statistics.median([float(x) for x in output])
                    fname2: str = vbrc_matrix_gen(dim1, dim2, "uniform", 1, 1, 1, perc_zeros, 0, True, SPARSE_VBR_DIR, 100)
                    vbr_spmv_codegen_python(fname2, dir_name=SPARSE_CODEGEN_DIR, vbr_dir=SPARSE_VBR_DIR, bench=BENCHMARK_FREQ)
                    try:
                        output = subprocess.check_output(["taskset", "-a", "-c", str(core), "python3", f"{fname2}.py"], cwd=os.path.join(BASE_PATH, SPARSE_CODEGEN_DIR)).decode("utf-8").split("\n")[0]
                    except subprocess.CalledProcessError as e:
                        print("SABLE Sparse: Execution failed for ", fname2, " with ", e)
                        continue
                    output = extract_mul_nums(output)
                    median_sable_time_sparse = statistics.median([float(x) for x in output])
                    f.write(",".join([str(dim1), str(dim2), str(perc_zeros), str(nnz), str(median_sable_time_sparse), str(median_sable_time_dense)]))
                    f.write("\n")
                    f.flush()
                    os.remove(os.path.join(DENSE_VBR_DIR, fname+".vbrc"))
                    os.remove(os.path.join(SPARSE_VBR_DIR, fname2+".vbrc"))

if __name__ == "__main__":
    calculate_threshold()
