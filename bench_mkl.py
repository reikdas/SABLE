import subprocess
import os
import pathlib

from scipy.io import mmread

from utils.utils import check_file_matches_parent_dir
from utils.fileio import write_dense_vector
from src.consts import CFLAGS as CFLAGS

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5

MKL_PATH = os.path.join(BASE_PATH, "..", "intel", "oneapi", "mkl", "latest")
MKL_FLAGS = [f"-I{MKL_PATH}/include", f"-L{MKL_PATH}/lib/intel64", "-lmkl_rt"]

if __name__ == "__main__":
    mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)
    subprocess.check_output(["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), "g++", "-o", "mkl-spmv", "mkl-spmv.cpp"] + CFLAGS + MKL_FLAGS, cwd=os.path.join(BASE_PATH, "src"))
    threads = [1]
    for thread in threads:
        with open(os.path.join(BASE_PATH, "results", f"mkl-spmv-suitesparse_{thread}thrd.csv"), "w") as f:
            f.write("Matrix,Time(ns)\n")
            for file_path in mtx_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                    fname = pathlib.Path(file_path).resolve().stem
                    print(f"Processing {fname}")
                    mtx = mmread(file_path)
                    cols = mtx.get_shape()[1]
                    write_dense_vector(1.0, cols)
                    output = subprocess.run(["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), f"{BASE_PATH}/src/mkl-spmv", str(file_path), str(thread), str(BENCHMARK_FREQ), os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cols}.vector")], capture_output=True, check=True, text=True)
                    csr_spmv_exec_time = float(output.stdout.split(" ")[1])
                    f.write(f"{fname},{csr_spmv_exec_time}\n")
                    f.flush()
