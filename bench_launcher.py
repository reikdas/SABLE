import os
import pathlib
import subprocess
import time

import pandas as pd
import psutil
import scipy
from mpi4py import MPI

from src.codegen import gen_single_threaded_spmv, gen_multi_threaded_spmv
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from src.autopartition import cut_indices2, similarity2, my_convert_dense_to_vbrc
from utils.fileio import write_dense_vector, read_vbr, read_vbrc
from utils.convert_real_to_vbr import convert_vbr_to_compressed, convert_sparse_to_vbr
from utils.utils import timeout
from bench_parallel_launcher import THREADS as THREADS
from bench_parallel_launcher import mtx_dir as mtx_dir
from bench_parallel_launcher import codegen_dir as codegen_dir
from bench_parallel_launcher import vbr_dir as vbr_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 60 * 4
PARTITION_TIMEOUT = 60 * 60 * 3

mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV")

@timeout(PARTITION_TIMEOUT)
def vbrc_wrapper(file_path, dest_path):
    return my_convert_dense_to_vbrc((str(file_path), str(dest_path)), 0.2, cut_indices2, similarity2)

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
                # val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(vbr_dir, fname, fname+".vbrc"))
                # val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(vbr_dir, fname, fname+".vbr"))
                # A = scipy.io.mmread(file_path)
                # A = scipy.sparse.csc_matrix(A, copy=False)
                time1 = time.time_ns() // 1_000_000
                # cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
                time2 = time.time_ns() // 1_000_000
                partition_time = time2-time1
                time1 = time.time_ns() // 1_000_000
                # val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(A, rpntr, cpntr, fname, os.path.join(vbr_dir,fname))
                time2 = time.time_ns() // 1_000_000
                VBR_convert_time = time2-time1
                time1 = time.time_ns() // 1_000_000
                val,indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(vbr_dir, fname, fname+".vbr"))
                val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, os.path.join(vbr_dir,fname))
                time2 = time.time_ns() // 1_000_000
                compress_time = time2-time1
                if len(val)==0:
                    f.write(f"{fname},ERROR5,ERROR5\n")
                    f.flush()
                    print(f"{fname} has no dense blocks")
                    continue
                if val is None:
                    f.write(f"{fname},ERROR2,ERROR2\n")
                    f.flush()
                    print(f"{fname} could not partition")
                    continue
                write_dense_vector(1.0, cpntr[-1])
                for thread in THREADS:
                    codegen_time = 0
                    if thread == 1:
                        codegen_time = gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(thread), fname, os.path.join(vbr_dir, fname), bench=100)
                    else:
                        codegen_time = gen_multi_threaded_spmv(thread, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(thread), fname, os.path.join(vbr_dir, fname))
                    try:
                        time1 = time.time_ns() // 1_000_000
                        subprocess.run(["taskset", "-a", "-c", str(core), "gcc", f"{fname}.c", "-o", fname] + CFLAGS + MKL_FLAGS + PAPI_COMPILE_FLAGS, cwd=codegen_dir+"_"+str(thread), check=True, timeout=COMPILE_TIMEOUT)
                        time2 = time.time_ns() // 1_000_000
                        compile_time = time2-time1
                    except subprocess.TimeoutExpired:
                        f.write(f"{fname},{codegen_time}ms,ERROR,ERROR\n")
                        f.flush()
                        print(f"Compiler failed for {fname}")
                        continue
                    if thread == 1:
                        f.write(f"{fname},{partition_time},{VBR_convert_time},{compress_time},{codegen_time},{compile_time}\n")
                        f.flush()
                    print(f"Done {fname} for {thread} threads")
            except Exception as e:
                f.write(f"{fname},ERROR3,ERROR3\n")
                f.flush()
                print(f"Errored {fname} with exception: {e}")
