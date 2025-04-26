import os
import pathlib
import subprocess

import os
import pathlib
import subprocess
import time

import scipy

from src.codegen import gen_single_threaded_spmv, gen_multi_threaded_spmv
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from src.autopartition import cut_indices2_fast, similarity2_numba, my_convert_dense_to_vbrc, cut_indices2, similarity2
from utils.fileio import write_dense_vector, read_vbr, read_vbrc
from utils.convert_real_to_vbr import convert_vbr_to_compressed, convert_sparse_to_vbr
from bench_parallel_launcher import THREADS as THREADS
from bench_parallel_launcher import mtx_dir as mtx_dir
from bench_parallel_launcher import codegen_dir as codegen_dir
from bench_parallel_launcher import vbr_dir as vbr_dir

from utils.utils import check_file_matches_parent_dir
import scipy

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV")
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR"))

cut_indices = cut_indices2_fast
similarity = similarity2_numba
cut_threshold = 0.2

if __name__ == "__main__":
    eval = ["eris1176",
    "std1_Jac3",
    "lp_wood1p",
    "jendrec1",
    "lowThrust_5",
    "hangGlider_4",
    "brainpc2",
    "hangGlider_3",
    "lowThrust_7",
    "lowThrust_11",
    "lowThrust_3",
    "lowThrust_6",
    "lowThrust_12",
    "hangGlider_5",
    "Journals",
    "bloweybl",
    "heart1",
    "TSOPF_FS_b9_c6",
    "Sieber",
    "case9",
    "c-30",
    "c-32",
    "freeFlyingRobot_10",
    "freeFlyingRobot_11",
    "freeFlyingRobot_12",
    "lowThrust_10",
    "lowThrust_13",
    "lowThrust_4",
    "lowThrust_8",
    "lowThrust_9",
    "lp_fit2p",
    "nd12k",
    "std1_Jac2",
    "vsp_c-30_data_data"]
    with open(os.path.join(BASE_PATH, "results", "suitesparse_inspect.csv"), "w") as f:
        f.write("Filename,Codegen(ms),Compile(ms)\n")
        for file_path in mtx_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                if fname not in eval:
                    continue
                try:
                    A = scipy.io.mmread(file_path)
                except:
                    print("Error reading file:", file_path)
                    continue
                print(f"Processing {fname}")
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                A = scipy.sparse.csc_matrix(A, copy=False)
                time1 = time.time_ns() // 1_000_000
                cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
                time2 = time.time_ns() // 1_000_000
                partition_time = time2-time1
                time1 = time.time_ns() // 1_000_000
                val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(A, rpntr, cpntr, fname, os.path.join(vbr_dir,fname))
                time2 = time.time_ns() // 1_000_000
                VBR_convert_time = time2-time1
                time1 = time.time_ns() // 1_000_000
                val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, os.path.join(vbr_dir,fname))
                time2 = time.time_ns() // 1_000_000
                compress_time = time2-time1
                write_dense_vector(1.0, cpntr[-1])
                for thread in THREADS:
                    codegen_time = 0
                    if thread == 1:
                        codegen_time = gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(thread), fname, os.path.join(vbr_dir, fname), bench=100)
                    else:
                        gen_multi_threaded_spmv(thread, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(thread), fname, os.path.join(vbr_dir, fname), bench=100)
                    try:
                        time1 = time.time_ns() // 1_000_000
                        subprocess.run(["gcc", f"{fname}.c", "-o", fname] + CFLAGS + MKL_FLAGS, cwd=codegen_dir+"_"+str(thread), check=True, timeout=COMPILE_TIMEOUT)
                        time2 = time.time_ns() // 1_000_000
                        compile_time = time2-time1
                    except subprocess.TimeoutExpired:
                        f.write(f"{fname},{codegen_time}ms,ERROR\n")
                        f.flush()
                        print(f"Compiler failed for {fname}")
                        continue
                    if thread == 1:
                        f.write(f"{fname},{partition_time + VBR_convert_time + compress_time + codegen_time},{compile_time}\n")
                        f.flush()
                    print(f"Done {fname} for {thread} threads")