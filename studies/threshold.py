import pathlib
import re
import statistics
import subprocess
import sys
import psutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.baseline import *
from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from synthesize_matrices.vbr_matrices_gen import vbr_matrix_gen
from utils.fileio import write_dense_vector, read_vbr
from utils.utils import extract_mul_nums
from utils.convert_real_to_vbr import convert_vbr_to_compressed

FILEPATH=pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 60 * 3

CODEGEN_DIR = "Generated_SpMV_threshold"
MTX_DIR = "Generated_MMarket_threshold"
VBR_DIR = "Generated_VBR_threshold"

def draw_heatmap():
    data = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))
    data = data[data['sable_time'] != 0]
    # Calculate the ratio of nonzeros_time to sable_time
    data["time_ratio"] = data[f"CSR_time"] / data["sable_time"]
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap_data = data.pivot(index="perc_zeros", columns="dim", values="time_ratio")
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": f"Speedup of SABLE dense over SABLE sparse"}
    )
    plt.xlabel("Side of nnz_block")
    plt.ylabel("Percentage of Zeros per nnz_block")
    plt.savefig(os.path.join(FILEPATH,"heatmap.png"))

def calculate_threshold():
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    perc_zeros_list = [0, 20, 40, 50, 75, 80, 85, 90, 95, 99]
    dims = [2, 4, 8, 10, 20, 50, 100, 200, 400]
    mat_side = 100000
    assert (mat_side%dim == 0 for dim in dims)
    write_dense_vector(1.0, mat_side)
    with open(os.path.join(FILEPATH,"threshold_results.csv"), "w") as f:
        f.write(f"dim,perc_zeros,nnz,CSR_time,sable_time\n")
        for dim in tqdm(dims, desc="Processing dimensions"):
            for perc_zeros in tqdm(perc_zeros_list, desc=f"Dim {dim}: Processing % zeros", leave=False):
                split = mat_side//dim
                nnz = (dim*dim*(100-perc_zeros))//100
                fname: str = vbr_matrix_gen(mat_side, mat_side, "uniform", split, split, 1, perc_zeros, 0, True, VBR_DIR)
                # vbr_to_mtx(fname+".vbr", dir_name=MTX_DIR, vbr_dir=VBR_DIR)
                val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(VBR_DIR, fname+".vbr"))
                print(fname)
                convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, VBR_DIR, 0)
                vbr_spmv_codegen(fname, dir_name=CODEGEN_DIR, vbr_dir=VBR_DIR, threads=1)
                try:
                    subprocess.run(["taskset", "-a", "-c", str(core), "./split_compile.sh", CODEGEN_DIR + "/" + fname + ".c", "2000"], cwd=BASE_PATH, check=True, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("SABLE Dense: Compilation failed for ", fname)
                    continue
                try:
                    output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"./{fname}"], cwd=os.path.join(BASE_PATH,"split-and-binaries",fname)).decode("utf-8").split("\n")[0]
                except subprocess.CalledProcessError:
                    print("SABLE Dense: Execution failed for ", fname)
                    continue
                output = extract_mul_nums(output)
                median_sable_time_dense = statistics.median([float(x) for x in output])
                convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, VBR_DIR, 100)
                vbr_spmv_codegen(fname, dir_name=CODEGEN_DIR, vbr_dir=VBR_DIR, threads=1)
                try:
                    subprocess.run(["taskset", "-a", "-c", str(core), "./split_compile.sh", CODEGEN_DIR + "/" + fname + ".c", "2000"], cwd=BASE_PATH, check=True, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("SABLE Sparse: Compilation failed for ", fname)
                    continue
                try:
                    output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"./{fname}"], cwd=os.path.join(BASE_PATH,"split-and-binaries",fname)).decode("utf-8").split("\n")[0]
                except subprocess.CalledProcessError:
                    print("SABLE Sparse: Execution failed for ", fname)
                    continue
                output = extract_mul_nums(output)
                median_sable_time_sparse = statistics.median([float(x) for x in output])
                print("SABLE Dense: ", median_sable_time_dense)
                print("SABLE Sparse: ", median_sable_time_sparse)
                f.write(",".join([str(dim), str(perc_zeros), str(nnz), str(median_sable_time_sparse), str(median_sable_time_dense)]))
                f.write("\n")
                f.flush()

if __name__ == "__main__":
    calculate_threshold()
    draw_heatmap()
