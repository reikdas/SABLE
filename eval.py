import os
import pathlib
import statistics
import subprocess

import psutil
from scipy.io import mmread

from src.consts import CFLAGS as CFLAGS
from utils.utils import extract_mul_nums

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5

def eval_single_proc(eval):
    mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    thread = 1
    # Compile CSR-SpMV
    subprocess.check_output(["g++", "-o", "csr-spmv", "csr-spmv.cpp"] + CFLAGS, cwd=os.path.join(BASE_PATH, "src"))
    with open(os.path.join(BASE_PATH, "results", "res.csv"), "w") as f:
        f.write("Filename,SABLE(ns),PSC(ns),CSRSpmv,SpeedupOverPSC,SpeedupOverCSRSpmv\n")
        for fname in eval:
            output = subprocess.check_output(["taskset", "-a", "-c", str(core), f"{BASE_PATH}/split-and-binaries/{fname}/{fname}"]).decode("utf-8").split("\n")[0]
            output = extract_mul_nums(output)
            median_exec_time_unroll = statistics.median([float(x) for x in output])
            
            file_path = pathlib.Path(os.path.join(mtx_dir, fname, f"{fname}.mtx"))
            
            output = subprocess.run(["taskset", "-a", "-c", str(core), f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(thread)], capture_output=True, check=True).stdout.decode("utf-8")
            psc_times_str: list[str] = output.split("\n")[:-1]
            psc_times: list[float] = [float(time) for time in psc_times_str]
            median_psc_time = float(statistics.median(psc_times))
            
            mtx = mmread(file_path)
            cols = mtx.get_shape()[1]
            output = subprocess.run(["taskset", "-a", "-c", str(core), f"{BASE_PATH}/src/csr-spmv", str(file_path), str(thread), str(BENCHMARK_FREQ), os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cols}.vector")], capture_output=True, check=True, text=True)
            csr_spmv_exec_time = float(output.stdout.split(" ")[1])
            
            if median_exec_time_unroll == 0:
                f.write(f"{fname},{median_exec_time_unroll},{median_psc_time},{csr_spmv_exec_time},Div by Zero,Div by Zero\n")
            else:
                speedup_over_psc = round(median_psc_time / median_exec_time_unroll, 2)
                speedup_over_csr_spmv = round(csr_spmv_exec_time / median_exec_time_unroll, 2)
                f.write(f"{fname},{median_exec_time_unroll},{median_psc_time},{csr_spmv_exec_time},{speedup_over_psc},{speedup_over_csr_spmv}\n")
            f.flush()
            print(f"Done {fname}")

if __name__ == "__main__":
    eval = [
    # "std1_Jac3",
    # "ts-palko",
    # "dbir2",
    # "Zd_Jac2",
    # "dbir1",
    # "msc23052",
    # "heart1",
    # "nemsemm1",
    # "bcsstk36",
    "karted"
    ]
    eval_single_proc(eval)
