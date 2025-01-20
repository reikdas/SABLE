import os
import pathlib
import subprocess

from tqdm import tqdm
import statistics

from src.autopartition import my_convert_dense_to_vbr
from src.codegen import vbr_spmv_codegen
from studies.check_partition import check_partition_iter2
from utils.fileio import write_dense_vector
from utils.smtx_to_mtx import parallel_dispatch, process_file

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 30

def bench_spmv():
    src_dir = pathlib.Path(os.path.join(BASE_PATH, "dlmc"))
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "dlmc_mtx"))
    vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "dlmc_vbr"))
    codegen_dir = os.path.join(BASE_PATH, "dlmc_spmv")

    print("Converting all dlmc smtx to mtx")
    parallel_dispatch(src_dir, mtx_dir, 1, process_file, ".smtx", ".mtx")

    total_files = sum(1 for file_path in mtx_dir.rglob("*") if file_path.is_file() and file_path.suffix == ".mtx")

    threads = [1]
    file_handles = {}
    for thread in threads:
        file_handles[thread] = open(os.path.join(BASE_PATH, "results", f"benchmarks_spmv_dlmc_{thread}.csv"), "w")
        file_handles[thread].write("Filename,SABLE no unroll,SABLE unroll,PSC,No unroll speedup,Unroll speedup\n")
    with open(os.path.join(BASE_PATH, "stats_spmv_dlmc.csv"), "w") as fstats:
        fstats.write("Filename,Percentage of 1x1 nnz_blocks,Percentage of 1D blocks,Mean size of larger nnz_blocks,Mean Density of larger nnz_blocks,Rows,Cols,nnz_blocks,Actual sparsity,New sparsity\n")
        pbar = tqdm(mtx_dir.rglob("*"), total=total_files, desc="Processing matrices", unit="matrix")
        for file_path in pbar:
            if file_path.is_file() and file_path.suffix == ".mtx":
                fname = pathlib.Path(file_path).resolve().stem
                pbar.set_description(f"Processing {fname}")
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                # Where to write the VBR file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Add nested progress bars for benchmarking
                with tqdm(total=(len(threads)*4) + 2, desc="Steps", leave=False) as step_bar:
                    # Convert matrix to VBR format
                    step_bar.set_description("Converting to VBR")
                    my_convert_dense_to_vbr((str(file_path), str(dest_path)))
                    step_bar.update(1)
                    # Check partitioning
                    step_bar.set_description("Checking partition")
                    perc_one_by_one, perc_one_d_blocks, avg_block_size, avg_density, rows, cols, nnz_blocks, total_sparsity, new_sparsity = check_partition_iter2(dest_path)
                    fstats.write(f"{os.path.join(relative_path.parent, fname)},{perc_one_by_one},{perc_one_d_blocks},{avg_block_size},{avg_density},{rows},{cols},{nnz_blocks},{total_sparsity},{new_sparsity}\n")
                    fstats.flush()
                    step_bar.update(1)
                    # Evaluate using SABLE
                    write_dense_vector(1.0, cols)
                    for i, thread in enumerate(threads):
                        codegen_dir_iter = codegen_dir + f"_{thread}thread"
                        step_bar.set_description(f"Generating code for {thread} thread(s) with no unroll")
                        vbr_spmv_codegen(fname, density=0, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
                        step_bar.update(1)
                        step_bar.set_description(f"Compiling code for {thread} thread(s) with no unroll")
                        try:
                            subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-pthread", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                        except subprocess.TimeoutExpired:
                            print(f"SABLE: {thread} thread(s) compilation failed for {os.path.join(relative_path.parent, fname)} with no unroll")
                            for remaining_thread in threads[i:]:
                                file_handles[remaining_thread].write(f"{os.path.join(relative_path.parent, fname)}, ERROR, ERROR, ERROR, ERROR\n")
                            break
                        step_bar.update(1)
                        step_bar.set_description("Benchmarking SABLE with no unroll")
                        subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                        execution_times = []
                        for _ in range(BENCHMARK_FREQ):
                            output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                            execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                            execution_times.append(execution_time)
                        step_bar.update(1)
                        step_bar.set_description(f"Generating code for {thread} thread(s) with unroll")
                        vbr_spmv_codegen(fname, density=8, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
                        step_bar.update(1)
                        step_bar.set_description(f"Compiling code for {thread} thread(s) with unroll")
                        try:
                            subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-pthread", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                        except subprocess.TimeoutExpired:
                            print(f"SABLE: {thread} thread(s) compilation failed for {os.path.join(relative_path.parent, fname)} with unroll")
                            for remaining_thread in threads[i:]:
                                file_handles[remaining_thread].write(f"{os.path.join(relative_path.parent, fname)}, ERROR, ERROR, ERROR, ERROR\n")
                            break
                        step_bar.update(1)
                        step_bar.set_description("Benchmarking SABLE with unroll")
                        subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                        execution_time_unroll = []
                        for _ in range(BENCHMARK_FREQ):
                            output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                            execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                            execution_time_unroll.append(execution_time)
                        step_bar.update(1)
                        step_bar.set_description("Benchmarking PSC\n")
                        subprocess.call([f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], stdout=subprocess.PIPE)
                        output = subprocess.run([f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                        psc_times = output.split("\n")[:-1]
                        step_bar.update(1)
                        if float(statistics.median(execution_times)) == 0 or float(statistics.median(execution_time_unroll)) == 0:
                            file_handles[thread].write(f"{os.path.join(relative_path.parent, fname)},ERROR,ERROR,ERROR,ERROR\n")
                        else:
                            file_handles[thread].write(f"{os.path.join(relative_path.parent, fname)},{statistics.median(execution_times)},{statistics.median(execution_time_unroll)},{statistics.median(psc_times)},{round(float(statistics.median(psc_times))/float(statistics.median(execution_times)), 2)},{round(float(statistics.median(psc_times))/float(statistics.median(execution_time_unroll)), 2)}\n")
                        file_handles[thread].flush()
        for thread in threads:
            file_handles[thread].close()

if __name__ == "__main__":
    bench_spmv()