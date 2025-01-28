#
# Warning, this is a copy-pasto of and an adaption from:
#
#    https://github.com/reikdas/SABLE/blob/large_suite-adhitha/bench_launched.py
#
# to experiment with C code splitting
#


import os
import pathlib
import subprocess
import time
import statistics
import psutil

import pandas as pd

from src.codegen2 import vbr_spmv_codegen
from src.autopartition import cut_indices2, similarity2, my_convert_dense_to_vbr
from utils.fileio import read_vbr, write_dense_vector
from compile import compile

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 60 * 4

codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse_thresh0.2_cut2_sim2_fastc")
vbr_dir = os.path.join(BASE_PATH, "vbr_dir")

matrices = ["/home/dynamo/a/apelenit/SABLE/Suitesparse/p0548.mtx"]

def time_ms(): return time.time_ns() // 1_000_000
def time_s(): return time_ms() // 1_000

def mtx_to_vbr(log, mtx_path, vbr_path):
    time1 = time_s()
    success = my_convert_dense_to_vbr((str(mtx_path), str(vbr_path)), 0.2, cut_indices2, similarity2)
    time2 = time_s()
    print("\t... done in " + str(time2-time1) + " s")
    if not success:
        log.write(f"{fname},ERROR2,ERROR2,ERROR2,ERROR2,ERROR2\n")
        log.flush()
        return False
    return True

if __name__ == "__main__":
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    # mat_dist = pd.read_csv("mat_dist.csv", header=None)
    # matrices = mat_dist.iloc[rank:rank+1].values[~pd.isnull(mat_dist.iloc[rank:rank+1].values)]
    # with open("mat_dist_" + str(rank) + ".csv", "w") as f:
    log = "log.csv"
    with open(log, "w") as f:
        f.write("Filename,Codegen(ms),Compile(ms),SABLE(us),PSC(us),Speedup\n")
        for file_path in matrices:
            file_path = pathlib.Path(file_path)
            fname = file_path.resolve().stem
            print(f"Process {fname} on core {core}")
            try:

                print("Convert .mtx -> .vbr")
                dest_path = pathlib.Path(vbr_dir, fname, fname + ".vbr")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if not os.path.exists(dest_path):
                    if not mtx_to_vbr(f, file_path, dest_path):
                        print(f"Done {fname} with a failure on the mtx->vbr step")
                        continue
                else:
                    print("\t... skipped since .vbr exists")

                print("Load .vbr")
                _, _, _, _, cpntr, _, _ = read_vbr(dest_path)
                write_dense_vector(1.0, cpntr[-1])

                # Sable bench
                print("Sable: stage 1 (generate C)")
                time1 = time_ms()
                codegen_time = vbr_spmv_codegen(fname, density=8, dir_name=codegen_dir, vbr_dir=dest_path.parent, threads=1)
                time2 = time_ms()
                print("\t... done in " + str(time2-time1) + " ms")
                print("Sable: stage 0 (generate binary)")
                bin = ""
                try:
                    time1 = time_s()
                    bin = compile(BASE_PATH, fname, core, codegen_dir, COMPILE_TIMEOUT)
                    time2 = time_s()
                    compile_time = time2-time1
                    print("\t... done in " + str(compile_time) + " s")
                except subprocess.TimeoutExpired:
                    f.write(f"{fname},{codegen_time}ms,TO,TO,TO,TO\n")
                    f.flush()

                print("Sable: run")
                subprocess.run(["taskset", "-a", "-c", str(core), bin], capture_output=True, check=True)
                execution_time_unroll = []
                for _ in range(BENCHMARK_FREQ):
                    output = subprocess.run(["taskset", "-a", "-c", str(core), bin], capture_output=True, check=True)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_time_unroll.append(execution_time)

                # PSC bench
                # TODO: comment out for now
                # output = subprocess.run(["taskset", "-a", "-c", str(core), f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                # psc_times = output.split("\n")[:-1]
                psc_times = "1"

                print("Right down stats in " + log)
                if float(statistics.median(execution_time_unroll)) == 0:
                    f.write(f"{fname},{codegen_time}ms,{compile_time}s,{statistics.median(execution_time_unroll)}us,{statistics.median(psc_times)}us,Div by Zero\n")
                else:
                    # step_bar.update(1)
                    f.write(f"{fname},{codegen_time}ms,{compile_time}s,{statistics.median(execution_time_unroll)}us,{statistics.median(psc_times)}us,{round(float(statistics.median(psc_times))/float(statistics.median(execution_time_unroll)), 2)}\n")
                f.flush()

                print(f"Done {fname}")
            except Exception as e:
                f.write(f"{fname},ERROR3,ERROR3,ERROR3,ERROR3,ERROR3\n")
                f.flush()
                print(f"Errored {fname} with exception: {e}")
