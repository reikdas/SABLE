import os
import subprocess

from src.codegen import vbr_spmm_codegen, vbr_spmv_codegen

BENCHMARK_FREQ = 5

def bench_spmv(bad: bool):
    if bad:
        gen_dir = "Generated_SpMV_Sparse_Bad"
        benchfile_name = "benchmarks_spmv_1_rel_bad.csv"
        density = 0
    else:
        gen_dir = "Generated_SpMV_Sparse"
        benchfile_name = "benchmarks_spmv_1_rel.csv"
        density = 15
    vbr_files = os.listdir("Generated_VBR_Sparse")
    print("Benchmarking executor")
    for thread in [1]:
        with open(os.path.join("results", benchfile_name), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmv_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmv_codegen(fname, density, dir_name=gen_dir, vbr_dir="Generated_VBR_Sparse", threads=thread)
                subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-funroll-all-loops", "-o", fname, spmv_file], cwd=gen_dir)
                output = subprocess.run(["./"+fname], capture_output=True, cwd=gen_dir)
                execution_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking threads={thread} executor iteration", i, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=gen_dir)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_times.append(execution_time)
                # save execution times to file
                p = f"{fname},{','.join(execution_times)}\n"
                print(p, flush=True)
                fMy.write(p)

def bench_spmm(bad: bool):
    if bad:
        gen_dir = "Generated_SpMM_Sparse_Bad"
        benchfile_name = "benchmarks_spmm_1_rel_bad.csv"
        density = 0
    else:
        gen_dir = "Generated_SpMM_Sparse"
        benchfile_name = "benchmarks_spmm_1_rel.csv"
        density = 15
    vbr_files = os.listdir("Generated_VBR_Sparse")
    print("Benchmarking executor")
    for thread in [1]:
        with open(os.path.join("results", benchfile_name), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmm_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmm_codegen(fname, density, dir_name=gen_dir, vbr_dir="Generated_VBR_Sparse", threads=thread)
                subprocess.run(["gcc", "-O3", "-pthread", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd=gen_dir)
                output = subprocess.run(["./"+fname], capture_output=True, cwd=gen_dir)
                execution_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking threads={thread} executor iteration", i, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=gen_dir)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_times.append(execution_time)
                # save execution times to file
                p = f"{fname},{','.join(execution_times)}\n"
                print(p, flush=True)
                fMy.write(p)

if __name__ == "__main__":
    bench_spmv(bad=True)
    bench_spmv(bad=False)
    bench_spmm(bad=True)
    bench_spmm(bad=False)
