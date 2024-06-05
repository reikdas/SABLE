import os
import subprocess

from src.codegen import vbr_spmv_codegen, vbr_spmm_codegen
from src.fileio import write_dense_matrix, write_dense_vector

BENCHMARK_FREQ = 5

def bench_spmv():
    vbr_dir = "manual_vbr"
    codegen_dir = "Generated_SpMV_manual"
    vbr_files = os.listdir(vbr_dir)
    for thread in [1, 2, 4, 8, 16]:
        with open(os.path.join("results", f"benchmarks_spmv_manual_{thread}.csv"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmv_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmv_codegen(fname, dense_blocks_only=False, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=thread)
                subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-funroll-all-loops", "-o", fname, spmv_file], cwd=codegen_dir)
                output = subprocess.run(["./"+fname], capture_output=True, cwd=codegen_dir)
                execution_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking threads={thread} executor iteration", i, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=codegen_dir)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_times.append(execution_time)
                # save execution times to file
                p = f"{fname},{','.join(execution_times)}\n"
                print(p, flush=True)
                fMy.write(p)


def bench_spmm():
    vbr_dir = "manual_vbr"
    codegen_dir = "Generated_SpMM_manual"
    vbr_files = os.listdir(vbr_dir)
    for thread in [1, 2, 4, 8, 16]:
        with open(os.path.join("results", f"benchmarks_spmm_manual_{thread}.csv"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmm_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmm_codegen(fname, dense_blocks_only=True, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=thread)
                subprocess.run(["gcc", "-O3", "-pthread", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd=codegen_dir)
                output = subprocess.run(["./"+fname], capture_output=True, cwd=codegen_dir)
                execution_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking threads={thread} executor iteration", i, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=codegen_dir)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_times.append(execution_time)
                # save execution times to file
                p = f"{fname},{','.join(execution_times)}\n"
                print(p, flush=True)
                fMy.write(p)


if __name__ == "__main__":
    # write_dense_matrix(1.0, 662, 512)
    # write_dense_matrix(1.0, 4000, 512)
    # write_dense_matrix(1.0, 1454, 512)
    write_dense_vector(1.0, 14109)
    bench_spmv()
    # bench_spmm()
