import os
import pathlib
import statistics
import subprocess

import psutil

from utils.utils import check_file_matches_parent_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5

def eval_single_proc(eval):
    mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    thread = 1
    with open(os.path.join(BASE_PATH, "results", "res.csv"), "w") as f:
        f.write("Filename,SABLE,PSC,Speedup\n")
        for fname in eval:
            execution_time_unroll: list[float] = []
            for _ in range(BENCHMARK_FREQ):
                output = subprocess.run(["taskset", "-a", "-c", str(core), f"{BASE_PATH}/split-and-binaries/{fname}/{fname}"], capture_output=True, check=True)
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_time_unroll.append(float(execution_time))
            file_path = pathlib.Path(os.path.join(mtx_dir, fname, f"{fname}.mtx"))
            output = subprocess.run(["taskset", "-a", "-c", str(core), f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(thread)], capture_output=True, check=True).stdout.decode("utf-8")
            psc_times_str: list[str] = output.split("\n")[:-1]
            psc_times: list[float] = [float(time) for time in psc_times_str]
            if float(statistics.median(execution_time_unroll)) == 0:
                f.write(f"{fname},{statistics.median(execution_time_unroll)}us,{statistics.median(psc_times)}us,Div by Zero\n")
            else:
                f.write(f"{fname},{statistics.median(execution_time_unroll)}us,{statistics.median(psc_times)}us,{round(float(statistics.median(psc_times))/float(statistics.median(execution_time_unroll)), 2)}\n")
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
