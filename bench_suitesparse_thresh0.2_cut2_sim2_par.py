import os
import pathlib
import subprocess
import time

from tqdm import tqdm
import statistics

from src.codegen import vbr_spmv_codegen
from studies.full_pipeline import check_file_matches_parent_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 45

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

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr_thresh0.2_cut2_sim2"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse_thresh0.2_cut2_sim2_par")

threads = [2, 4, 8]
file_handles = {}

pid = os.getpid()

# Get the CPU affinity of the current process
cpu_affinity = os.sched_getaffinity(pid)


def foo(file_path):
    if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
        fname = pathlib.Path(file_path).resolve().stem
        if fname not in eval:
            return
        print(f"Processing {fname}")
        relative_path = file_path.relative_to(mtx_dir)
        dest_path = vbr_dir / relative_path.with_suffix(".vbr")
        # Where to write the VBR file
        # dest_path.parent.mkdir(parents=True, exist_ok=True)
        # my_convert_dense_to_vbr((str(file_path), str(dest_path)), 0.2, cut_indices2, similarity2)
        # _, _, _, _, cpntr, _, _ = read_vbr(dest_path)
        # write_dense_vector(1.0, cpntr[-1])
        for i, thread in enumerate(threads):
            codegen_dir_iter = codegen_dir + f"_{thread}thread"
            # step_bar.set_description(f"Generating code for {thread} thread(s) with no unroll")
            # vbr_spmv_codegen(fname, density=0, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
            # step_bar.update(1)
            # step_bar.set_description(f"Compiling code for {thread} thread(s) with no unroll")
            # try:
            #     subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-pthread", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
            # except subprocess.TimeoutExpired:
            #     print(f"SABLE: {thread} thread(s) compilation failed for {fname} with no unroll")
            #     for remaining_thread in threads[i:]:
            #         file_handles[remaining_thread].write(f"{fname},ERROR1,ERROR1,ERROR1,ERROR1\n")
            #     break
            # step_bar.update(1)
            # step_bar.set_description("Benchmarking SABLE with no unroll")
            # subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
            # execution_times = []
            # for _ in range(BENCHMARK_FREQ):
            #     output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
            #     execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
            #     execution_times.append(execution_time)
            # step_bar.update(1)
            # step_bar.set_description(f"Generating code for {thread} thread(s) with unroll")
            try:
                codegen_time = vbr_spmv_codegen(fname, density=8, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
            except FileNotFoundError:
                for remaining_thread in threads[i:]:
                    file_handles[remaining_thread].write(f"{fname},ERROR3,ERROR3,ERROR3,ERROR3\n")
                    file_handles[remaining_thread].flush()
                break
            # step_bar.update(1)
            # step_bar.set_description(f"Compiling code for {thread} thread(s) with unroll")
            try:
                time1 = time.time_ns() // 1_000
                subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)), "gcc", "-O3", "-march=native", "-pthread", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                time2 = time.time_ns() // 1_000
                compile_time = time2-time1
            except subprocess.TimeoutExpired:
                for remaining_thread in threads[i:]:
                    file_handles[remaining_thread].write(f"{fname},ERROR2,ERROR2,ERROR2,ERROR2\n")
                    file_handles[remaining_thread].flush()
                break
            # step_bar.update(1)
            # step_bar.set_description("Benchmarking SABLE with unroll")
            try:
                subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)), "./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                execution_time_unroll = []
                for _ in range(BENCHMARK_FREQ):
                    output = subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)), "./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_time_unroll.append(execution_time)
            except subprocess.CalledProcessError:
                file_handles[thread].write(f"{fname},ERROR4,ERROR4,ERROR4,ERROR4\n")
                file_handles[thread].flush()
                break
            # step_bar.update(1)
            # step_bar.set_description("Benchmarking PSC\n")
            output = subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)), f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(thread)], capture_output=True, check=True).stdout.decode("utf-8")
            psc_times = output.split("\n")[:-1]
            if float(statistics.median(execution_time_unroll)) == 0:
                file_handles[thread].write(f"{fname},{codegen_time+compile_time},{statistics.median(execution_time_unroll)},{statistics.median(psc_times)},{2*statistics.median([float(time) for time in psc_times])}\n")
            else:
            # step_bar.update(1)
                file_handles[thread].write(f"{fname},{codegen_time+compile_time},{statistics.median(execution_time_unroll)},{statistics.median(psc_times)},{round(float(statistics.median(psc_times))/float(statistics.median(execution_time_unroll)), 2)}\n")
            file_handles[thread].flush()
        print(f"Done {fname}")

def bench_spmv():
    # Iterate over all files in Suitesparse
    # Get total number of files to process for the progress bar
    total_files = sum(1 for file_path in mtx_dir.rglob("*") 
                     if file_path.is_file() and 
                     file_path.suffix == ".mtx" and 
                     check_file_matches_parent_dir(file_path))

    pbar = tqdm(mtx_dir.rglob("*"), total=total_files, desc="Processing suitesparse matrices", unit="matrix")
    for file_path in pbar:
        foo(file_path)
        

if __name__ == "__main__":
    for thread in threads:
        file_handles[thread] = open(os.path.join(BASE_PATH, "results", f"benchmarks_spmv_suitesparse_thresh0.2_cut2_sim2_par_{thread}thread.csv"), "w")
        file_handles[thread].write("Filename,Codegen+Compile,SABLE unroll,PSC,Unroll speedup\n")
    bench_spmv()
    for thread in threads:
        file_handles[thread].close()
