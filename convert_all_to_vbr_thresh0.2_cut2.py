import os
import pathlib
import subprocess
import time

from tqdm import tqdm
import statistics

from src.autopartition import my_convert_dense_to_vbr, cut_indices2, similarity2
from src.codegen import vbr_spmv_codegen
from studies.check_partition import check_partition_iter2
from studies.full_pipeline import check_file_matches_parent_dir
from utils.fileio import write_dense_vector

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 30

eval = [
    "dwt_209",
    "p0548",
    "lshp_406",
    "bfwa398",
    "lp_degen2",
    "cavity04",
    "iprob",
    "rbsa480",
    "mycielskian11",
    "kl02",
]

def bench_spmv():
    # Iterate over all files in Suitesparse
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr_thresh0.2_cut2_final"))
    # codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse_thresh0.2_cut1")
    # Get total number of files to process for the progress bar
    # total_files = sum(1 for file_path in mtx_dir.rglob("*") 
    #                  if file_path.is_file() and 
    #                  file_path.suffix == ".mtx" and 
    #                  check_file_matches_parent_dir(file_path))
    # threads = [1]
    # file_handles = {}
    # for thread in threads:
    #     file_handles[thread] = open(os.path.join(BASE_PATH, "results", f"benchmarks_spmv_suitesparse_thresh0.2_cut1_{thread}thread.csv"), "w")
    #     file_handles[thread].write("Filename,SABLE no unroll,SABLE unroll,PSC,No unroll speedup,Unroll speedup\n")
    # matrices = []
    # pbar = tqdm(mtx_dir.rglob("*"), total=total_files, desc="Processing suitesparse matrices", unit="matrix")
    # for file_path in pbar:
    with open(os.path.join(BASE_PATH, "suitesparse_insp_conv.csv"), "w") as f:
        f.write("Filename,Time\n")
        for file_path in mtx_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                if fname not in eval:
                    continue
                # if fname == "LFAT5000":
                    # pbar.set_description(f"Processing {fname}")
                print(f"Processing {fname}")
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                # Where to write the VBR file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Add nested progress bars for benchmarking
                # with tqdm(total=(len(threads)*7) + 2, desc="Steps", leave=False) as step_bar:
                # try:
                    # Convert matrix to VBR format
                    # step_bar.set_description("Converting to VBR")
                print("Converting to VBR")
                time1 = time.time_ns() // 1_000
                my_convert_dense_to_vbr((str(file_path), str(dest_path)), 0.2, cut_indices2, similarity2)
                time2 = time.time_ns() // 1_000
                # step_bar.update(1)
                # Check partitioning
                # step_bar.set_description("Checking partition")
                # print("Checking partition")
                # perc_one_by_one, perc_one_d_blocks, avg_block_size, avg_density, rows, cols, nnz_blocks, total_sparsity, new_sparsity = check_partition_iter2(dest_path)
                f.write(f"{fname},{time2-time1}\n")
                f.flush()
                    # step_bar.update(1)
                    # Evaluate using SABLE
                #     write_dense_vector(1.0, cols)
                #     for i, thread in enumerate(threads):
                #         codegen_dir_iter = codegen_dir + f"_{thread}thread"
                #         # step_bar.set_description(f"Generating code for {thread} thread(s) with no unroll")
                #         print("Generating code for 1 thread with no unroll")
                #         vbr_spmv_codegen(fname, density=0, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
                #         # step_bar.update(1)
                #         # step_bar.set_description(f"Compiling code for {thread} thread(s) with no unroll")
                #         print("Compiling code for 1 thread with no unroll")
                #         try:
                #             subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-pthread", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                #         except subprocess.TimeoutExpired:
                #             print(f"SABLE: {thread} thread(s) compilation failed for {fname} with no unroll")
                #             for remaining_thread in threads[i:]:
                #                 file_handles[remaining_thread].write(f"{fname},ERROR1,ERROR1,ERROR1,ERROR1\n")
                #             break
                #         # step_bar.update(1)
                #         # step_bar.set_description("Benchmarking SABLE with no unroll")
                #         print("Benchmarking SABLE with no unroll")
                #         subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                #         execution_times = []
                #         for _ in range(BENCHMARK_FREQ):
                #             output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                #             execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                #             execution_times.append(execution_time)
                #         # step_bar.update(1)
                #         # step_bar.set_description(f"Generating code for {thread} thread(s) with unroll")
                #         print("Generating code for 1 thread with unroll")
                #         vbr_spmv_codegen(fname, density=8, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
                #         # step_bar.update(1)
                #         # step_bar.set_description(f"Compiling code for {thread} thread(s) with unroll")
                #         print("Compiling code for 1 thread with unroll")
                #         try:
                #             subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-pthread", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                #         except subprocess.TimeoutExpired:
                #             print(f"SABLE: {thread} thread(s) compilation failed for {fname} with unroll")
                #             for remaining_thread in threads[i:]:
                #                 file_handles[remaining_thread].write(f"{fname},ERROR2,ERROR2,ERROR2,ERROR2\n")
                #             break
                #         # step_bar.update(1)
                #         # step_bar.set_description("Benchmarking SABLE with unroll")
                #         print("Benchmarking SABLE with unroll")
                #         subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                #         execution_time_unroll = []
                #         for _ in range(BENCHMARK_FREQ):
                #             output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                #             execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                #             execution_time_unroll.append(execution_time)
                #         # step_bar.update(1)
                #         # step_bar.set_description("Benchmarking PSC\n")
                #         print("Benchmarking PSC")
                #         output = subprocess.run([f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                #         psc_times = output.split("\n")[:-1]
                #         # step_bar.update(1)
                #         if float(statistics.median(execution_times)) == 0 or float(statistics.median(execution_time_unroll)) == 0:
                #             file_handles[thread].write(f"{fname},ERROR3,ERROR3,ERROR3,ERROR3\n")
                #         else:
                #             file_handles[thread].write(f"{fname},{statistics.median(execution_times)},{statistics.median(execution_time_unroll)},{statistics.median(psc_times)},{round(float(statistics.median(psc_times))/float(statistics.median(execution_times)), 2)},{round(float(statistics.median(psc_times))/float(statistics.median(execution_time_unroll)), 2)}\n")
                #         file_handles[thread].flush()
                # except:
                #     file_handles[thread].write(f"{fname},ERROR4,ERROR4,ERROR4,ERROR4\n")
                #     file_handles[thread].flush()
    # for thread in threads:
    #     file_handles[thread].close()

if __name__ == "__main__":
    bench_spmv()
