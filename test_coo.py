import pathlib
import statistics
import subprocess
import sys

import psutil
import scipy

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.baseline import *
from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from src.autopartition import cut_indices2, similarity2
from utils.convert_real_to_vbr import convert_vbr_to_compressed, convert_sparse_to_vbr
from utils.fileio import write_dense_vector
from utils.utils import extract_mul_nums, check_file_matches_parent_dir

FILEPATH=pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 60 * 3

CODEGEN_DIR = "Generated_SpMV_COO"
mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
VBR_DIR = "Generated_VBR_COO"

eval = [
    # "karted",
    # "nemsemm1",
    "bcsstk36",
    # "jan99jac120",
    # "lowThrust_12",
    # "juba40k",
    "heart1",
]

skip = [
    "xenon1",
    "c-73",
    "boyd1",
    "SiO",
    "crashbasis",
    "2cubes_sphere",
]

if __name__ == "__main__":
    if not os.path.exists(CODEGEN_DIR):
        os.makedirs(CODEGEN_DIR)
    if not os.path.exists(VBR_DIR):
        os.makedirs(VBR_DIR)
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    with open(os.path.join(FILEPATH, "test_coo.csv"), "w") as f:
        f.write("Matrix,Time(ns)\n")
        for file_path in mtx_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                if fname not in eval:
                    continue
                if fname in skip:
                    continue
                print(fname)
                mtx = scipy.io.mmread(file_path)
                mtx_size = mtx.shape[0] * mtx.shape[1]
                A = scipy.sparse.csc_matrix(mtx, copy=False)
                cpntr, rpntr = cut_indices2(A, 0.2, similarity2)
                write_dense_vector(1.0, cpntr[-1])
                val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(A, rpntr, cpntr, fname, VBR_DIR)
                val, indx, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j, coo_val = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, VBR_DIR, 100)
                vbr_spmv_codegen(fname, dir_name=CODEGEN_DIR, vbr_dir=VBR_DIR, threads=1)
                try:
                    subprocess.run(["taskset", "-a", "-c", str(core), "gcc", "-o", fname, f"{fname}.c"] + CFLAGS, cwd=CODEGEN_DIR, check=True, timeout=COMPILE_TIMEOUT)
                # try:
                #     subprocess.run(["taskset", "-a", "-c", str(core), "./split_compile.sh", CODEGEN_DIR + "/" + fname + ".c", "2000"], cwd=BASE_PATH, check=True, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("SABLE Sparse: Compilation failed for ", fname)
                    continue
                try:
                    output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"./{fname}"], cwd=CODEGEN_DIR).decode("utf-8").split("\n")[0]
                # try:
                #     output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"./{fname}"], cwd=os.path.join(BASE_PATH,"split-and-binaries",fname)).decode("utf-8").split("\n")[0]
                except subprocess.CalledProcessError:
                    print("SABLE Sparse: Execution failed for ", fname)
                    continue
                output = extract_mul_nums(output)
                median_sable_time = statistics.median([float(x) for x in output])
                f.write(f"{fname},{median_sable_time}\n")
                f.flush()
                print(f"Done {fname}")
