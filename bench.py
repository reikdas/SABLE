import subprocess
import os
import time

from src.codegen import vbr_spmv_codegen, vbr_spmm_codegen

BENCHMARK_FREQ = 5


def bench_spmv(dense_blocks_only: bool = True):
    '''
    Here, dense_blocks_only denote that the blocks are mostly dense.
    '''

    inspector_output = "benchmarks_inspector_dense" if dense_blocks_only else "benchmarks_inspector_sparse"
    executor_output = "benchmarks_spmv_dense" if dense_blocks_only else "benchmarks_spmv_sparse"
    input_dir = "Generated_VBR_Dense" if dense_blocks_only else "Generated_VBR_Sparse"
    output_dir = "Generated_SpMV"
    vbr_files = os.listdir(input_dir)
    
    print("Benchmarking inspector")
    with open(os.path.join("results", f"{inspector_output}.txt"), "w") as fInspector:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".c"
            print(filename, flush=True)
            inspector_times = []
            for i in range(BENCHMARK_FREQ):
                # SpMV code generation by inspecting the VBR matrix
                print("Benchmarking inspector iteration", i, flush=True)
                spmv_codegen_time = vbr_spmv_codegen(fname, dense_blocks_only, vbr_dir=input_dir, dir_name=output_dir)
                time1 = time.time_ns() // 1_000
                # compile the generated code for SpMV operation
                subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-o", fname, spmv_file], cwd="Generated_SpMV")
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
        with open(os.path.join("results", f"{executor_output}_{thread}.txt"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmv_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmv_codegen(fname, dense_blocks_only, dir_name=output_dir, vbr_dir=input_dir, threads=thread)
                subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-o", fname, spmv_file], cwd="Generated_SpMV")
                execution_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking threads={thread} executor iteration", i, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMV")
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_times.append(execution_time)
                # save execution times to file
                p = f"{fname},{','.join(execution_times)}\n"
                print(p, flush=True)
                fMy.write(p)

def bench_spmm(dense_blocks_only: bool = True):        
    vbr_mat_input_dir = "Generated_VBR_Dense" if dense_blocks_only else "Generated_VBR_Sparse"
    vbr_files = os.listdir(vbr_mat_input_dir)
    output = "benchmarks_spmm_dense" if dense_blocks_only else "benchmarks_spmm_sparse"
    output_dir = "Generated_SpMM"
    
    for thread in [1, 2, 4, 8, 16]:
        with open(os.path.join("results", f"{output}_{thread}.txt"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmm_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmm_codegen(fname, vbr_dir=vbr_mat_input_dir, dir_name=output_dir, threads=thread)
                # Only execute on Tim Rogers' machine since it has AVX-512 instructions
                subprocess.run(["/usr/bin/gcc-8", "-O3", "-pthread", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd="Generated_SpMM")
                execution_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking threads={thread} executor iteration", i, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM")
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_times.append(execution_time)
                # save execution times to file
                p = f"{fname},{','.join(execution_times)}\n"
                print(p, flush=True)
                fMy.write(p)

if __name__ == "__main__":
    bench_spmv(dense_blocks_only=True)
    bench_spmv(dense_blocks_only=False)
    bench_spmm(dense_blocks_only=True)
    bench_spmm(dense_blocks_only=False)
