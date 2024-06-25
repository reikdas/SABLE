import os
import subprocess
import time

from src.codegen import vbr_spmv_codegen, vbr_spmm_codegen
from src.fileio import write_dense_matrix, write_dense_vector

BENCHMARK_FREQ = 10

def bench_spmv():
    vbr_dir = "manual_vbr"
    codegen_dir = "Generated_SpMV_manual"
    vbr_files = os.listdir(vbr_dir)
    print("Benchmarking inspector")
    with open(os.path.join("results", "benchmarks_inspector_spmv_manual.csv"), "w") as fInspector:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".c"
            print(filename, flush=True)
            inspector_times = []
            vbr_spmv_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=1)
            subprocess.run(["gcc", "-O3", "-funroll-all-loops", "-march=native", "-o", fname, spmv_file], cwd=codegen_dir)
            for i in range(BENCHMARK_FREQ):
                # SpMV code generation by inspecting the VBR matrix
                print("Benchmarking inspector iteration", i, flush=True)
                spmv_codegen_time = vbr_spmv_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=1)
                time1 = time.time_ns() // 1_000
                # compile the generated code for SpMV operation
                subprocess.run(["gcc", "-O3", "-funroll-all-loops", "-march=native", "-o", fname, spmv_file], cwd=codegen_dir)
                time2 = time.time_ns() // 1_000
                compilation_time = time2 - time1
                inspector_time = spmv_codegen_time + compilation_time
                inspector_times.append(inspector_time)
            # save inspector time (code generation + compilation) to file
            p = f"{fname},{','.join([str(x) for x in inspector_times])}\n"
            print(p, flush = True)
            fInspector.write(p)
    print("Benchmarking executor")
    for thread in [1, 2, 4, 8, 16]:
        with open(os.path.join("results", f"benchmarks_spmv_manual_{thread}.csv"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmv_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmv_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=thread)
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
    print("Benchmarking inspector")
    with open(os.path.join("results", "benchmarks_inspector_spmm_manual.csv"), "w") as fInspector:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".c"
            print(filename, flush=True)
            inspector_times = []
            vbr_spmm_codegen(fname, density=0, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=1)
            subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd=codegen_dir)
            for i in range(BENCHMARK_FREQ):
                # SpMV code generation by inspecting the VBR matrix
                print("Benchmarking inspector iteration", i, flush=True)
                spmm_codegen_time = vbr_spmm_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=1)
                time1 = time.time_ns() // 1_000
                # compile the generated code for SpMV operation
                subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd=codegen_dir)
                time2 = time.time_ns() // 1_000
                compilation_time = time2 - time1
                inspector_time = spmm_codegen_time + compilation_time
                inspector_times.append(inspector_time)
            # save inspector time (code generation + compilation) to file
            p = f"{fname},{','.join([str(x) for x in inspector_times])}\n"
            print(p, flush = True)
            fInspector.write(p)
    print("Benchmarking executor")
    for thread in [1, 2, 4, 8, 16]:
        with open(os.path.join("results", f"benchmarks_spmm_manual_{thread}.csv"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmm_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmm_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=thread)
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
    write_dense_vector(1.0, 1454)
    write_dense_matrix(1.0, 1454, 512)
    write_dense_vector(1.0, 132)
    write_dense_matrix(1.0, 132, 512)
    write_dense_vector(1.0, 36)
    write_dense_matrix(1.0, 36, 512)
    write_dense_vector(1.0, 55)
    write_dense_matrix(1.0, 55, 512)
    write_dense_vector(1.0, 66)
    write_dense_matrix(1.0, 66, 512)
    bench_spmv()
    bench_spmm()
