import pathlib
import re
import statistics
import subprocess
import sys

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
from utils.mtx_matrices_gen import vbr_to_mtx
from utils.utils import extract_mul_nums
from utils.convert_real_to_vbr import convert_vbr_to_compressed

FILEPATH=pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 10

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
        cbar_kws={"label": f"Speedup of SABLE over CSR SpMV"}
    )
    plt.xlabel("Side of nnz_block")
    plt.ylabel("Percentage of Zeros per nnz_block")
    plt.savefig(os.path.join(FILEPATH,"heatmap.png"))

def calculate_threshold():
    perc_zeros_list = [0, 10, 20, 30, 40, 50, 75, 99]
    dims = [10, 20, 50, 100, 400]
    mat_side = 100000
    assert (mat_side%dim == 0 for dim in dims)
    write_dense_vector(1.0, mat_side)
    subprocess.check_output(["g++", "-o", "csr-spmv", "csr-spmv.cpp"] + CFLAGS, cwd=os.path.join(BASE_PATH, "src"))
    with open(os.path.join(FILEPATH,"threshold_results.csv"), "w") as f:
        f.write(f"dim,perc_zeros,nnz,CSR_time,sable_time\n")
        for dim in tqdm(dims, desc="Processing dimensions"):
            for perc_zeros in tqdm(perc_zeros_list, desc=f"Dim {dim}: Processing % zeros", leave=False):
                split = mat_side//dim
                nnz = (dim*dim*(100-perc_zeros))//100
                fname = vbr_matrix_gen(mat_side, mat_side, "uniform", split, split, 100, perc_zeros, 0, True, VBR_DIR)
                vbr_to_mtx(fname+".vbr", dir_name=MTX_DIR, vbr_dir=VBR_DIR)
                val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(VBR_DIR, fname+".vbr"))
                convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, 0, fname, VBR_DIR)
                print(fname)
                baseline_output = subprocess.run(["./csr-spmv", os.path.join(BASE_PATH, MTX_DIR, fname+".mtx"), str(1), str(BENCHMARK_FREQ), os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{mat_side}.vector")], capture_output=True, cwd=os.path.join(BASE_PATH, "src"))
                execution_time = baseline_output.stdout.decode("utf-8").split("\n")[0]
                match = re.search(r"\d+", execution_time)
                if not match:
                    raise Exception("Unable to parse baseline output")
                baseline_times = float(match.group())
                vbr_spmv_codegen(fname, dir_name=CODEGEN_DIR, vbr_dir=VBR_DIR, threads=1)
                try:
                    subprocess.run(["gcc", "-o", fname, fname+".c"] + CFLAGS, cwd=CODEGEN_DIR, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("SABLE: Compilation failed for ", fname)
                    continue
                output = subprocess.check_output(["./"+fname], cwd=CODEGEN_DIR).decode("utf-8").split("\n")[0]
                output = extract_mul_nums(output)
                median_sable_time = statistics.median([float(x) for x in output])
                print("SABLE: ", median_sable_time)
                print("CSR SpMV: ", baseline_times)
                f.write(",".join([str(dim), str(perc_zeros), str(nnz), str(baseline_times), str(median_sable_time)]))
                f.write("\n")
                f.flush()

if __name__ == "__main__":
    calculate_threshold()
    draw_heatmap()
