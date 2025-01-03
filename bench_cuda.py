import os
import subprocess
import time

from src.codegen import vbr_spmm_cuda_codegen, vbr_spmv_cuda_codegen

BENCHMARK_FREQ = 5

def bench_spmv():
    vbr_files = os.listdir("Generated_VBR")
    print("Benchmarking inspector")
    with open(os.path.join("results", "benchmarks_inspector_spmv_cuda.csv"), "w") as fInspector:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".cu"
            print(filename, flush=True)
            inspector_times = []
            vbr_spmv_cuda_codegen(fname, density=0, dir_name="Generated_SpMV_cuda", vbr_dir="Generated_VBR")
            subprocess.run(["nvcc", "-O3", "-o", fname, spmv_file], cwd="Generated_SpMV_cuda")
            for i in range(BENCHMARK_FREQ):
                # SpMV code generation by inspecting the VBR matrix
                print("Benchmarking inspector iteration", i, flush=True)
                spmv_codegen_time = vbr_spmv_cuda_codegen(fname, density=0, dir_name="Generated_SpMV_cuda", vbr_dir="Generated_VBR")
                time1 = time.time_ns() // 1_000
                # compile the generated code for SpMV operation
                subprocess.run(["nvcc", "-O3", "-o", fname, spmv_file], cwd="Generated_SpMV_cuda")
                time2 = time.time_ns() // 1_000
                compilation_time = time2 - time1
                inspector_time = spmv_codegen_time + compilation_time
                inspector_times.append(inspector_time)
            # save inspector time (code generation + compilation) to file
            p = f"{fname},{','.join([str(x) for x in inspector_times])}\n"
            print(p, flush = True)
            fInspector.write(p)
    print("Benchmarking executor")
    with open(os.path.join("results", f"benchmarks_spmv_cuda.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".cu"
            print(filename, flush=True)
            # compile the generated code for SpMV operation
            vbr_spmv_cuda_codegen(fname, density=0, dir_name="Generated_SpMV_cuda", vbr_dir="Generated_VBR")
            subprocess.run(["nvcc", "-O3", "-o", fname, spmv_file], cwd="Generated_SpMV_cuda")
            output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMV_cuda")
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMV_cuda")
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

def bench_spmm():
    vbr_files = os.listdir("Generated_VBR")
    print("Benchmarking inspector")
    with open(os.path.join("results", "benchmarks_inspector_spmm_cuda.csv"), "w") as fInspector:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".cu"
            print(filename, flush=True)
            inspector_times = []
            vbr_spmm_cuda_codegen(fname, density=0, dir_name="Generated_SpMM_cuda", vbr_dir="Generated_VBR")
            subprocess.run(["nvcc", "-O3", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd="Generated_SpMM_cuda")
            for i in range(BENCHMARK_FREQ):
                # SpMV code generation by inspecting the VBR matrix
                print("Benchmarking inspector iteration", i, flush=True)
                spmm_codegen_time = vbr_spmm_cuda_codegen(fname, density=0, dir_name="Generated_SpMM_cuda", vbr_dir="Generated_VBR")
                time1 = time.time_ns() // 1_000
                # compile the generated code for SpMV operation
                subprocess.run(["nvcc", "-O3", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd="Generated_SpMM_cuda")
                time2 = time.time_ns() // 1_000
                compilation_time = time2 - time1
                inspector_time = spmm_codegen_time + compilation_time
                inspector_times.append(inspector_time)
            # save inspector time (code generation + compilation) to file
            p = f"{fname},{','.join([str(x) for x in inspector_times])}\n"
            print(p, flush = True)
            fInspector.write(p)
    print("Benchmarking executor")
    with open(os.path.join("results", f"benchmarks_spmm_cuda.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".cu"
            print(filename, flush=True)
            # compile the generated code for SpMV operation
            vbr_spmm_cuda_codegen(fname, density=0, dir_name="Generated_SpMM_cuda", vbr_dir="Generated_VBR")
            subprocess.run(["nvcc", "-O3", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd="Generated_SpMM_cuda")
            output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_cuda")
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print(f"Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_cuda")
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

if __name__ == "__main__":
    bench_spmv()
    bench_spmm()
