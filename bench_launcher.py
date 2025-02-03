import os
import pathlib
import subprocess
import time
import psutil

import pandas as pd
from mpi4py import MPI

from src.codegen import gen_single_threaded_spmv_compressed
from src.autopartition import cut_indices2, similarity2, my_convert_dense_to_vbr
from utils.fileio import write_dense_vector

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 60 * 4

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_large"))
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr_thresh0.2_cut2_sim2"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse_thresh0.2_cut2_sim2_fastc")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    mat_dist = pd.read_csv("mat_dist.csv", header=None)
    matrices = mat_dist.iloc[rank:rank+1].values[~pd.isnull(mat_dist.iloc[rank:rank+1].values)]
    with open("mat_dist_" + str(rank) + ".csv", "w") as f:
        f.write("Filename,Codegen(ms),Compile(ms)\n")
        for file_path in matrices:
            file_path = pathlib.Path(file_path)
            fname = file_path.resolve().stem
            print(f"Rank {rank}: Process {fname} on core {core}")
            try:
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, ublocks, coo_i, coo_j = my_convert_dense_to_vbr((str(file_path), str(dest_path)), 0.2, cut_indices2, similarity2)
                if val is None:
                    f.write(f"{fname},ERROR2,ERROR2\n")
                    f.flush()
                    print(f"Done {fname}")
                    continue
                write_dense_vector(1.0, cpntr[-1])
                codegen_time = gen_single_threaded_spmv_compressed(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, coo_i, coo_j, codegen_dir, fname, dest_path.parent)
                try:
                    time1 = time.time_ns() // 1_000_000
                    subprocess.run(["taskset", "-a", "-c", str(core), "./split.sh", "64", codegen_dir + "/" + fname + ".c"], cwd=BASE_PATH, check=True, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
                    time2 = time.time_ns() // 1_000_000
                    compile_time = time2-time1
                except subprocess.TimeoutExpired:
                    f.write(f"{fname},{codegen_time}ms,ERROR,ERROR\n")
                    f.flush()
                    print(f"Done {fname}")
                    continue
                f.write(f"{fname},{codegen_time}ms,{compile_time}ms\n")
                f.flush()
                print(f"Done {fname}")
            except Exception as e:
                f.write(f"{fname},ERROR3,ERROR3\n")
                f.flush()
                print(f"Errored {fname} with exception: {e}")
