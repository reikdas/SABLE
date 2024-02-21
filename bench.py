import subprocess
import os
import time

from src.codegen import vbr_spmv_codegen

BENCHMARK_FREQ = 5

if __name__ == "__main__":
    vbr_files = os.listdir("Generated_VBR")
    print("Benchmarking inspector")
    with open(os.path.join("results", "benchmarks_inspector.txt"), "w") as fInspector:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".c"
            print(filename, flush=True)
            inspector_times = []
            for i in range(BENCHMARK_FREQ):
                # SpMV code generation by inspecting the VBR matrix
                print("Benchmarking inspector iteration", i, flush=True)
                spmv_codegen_time = vbr_spmv_codegen(fname, threads=1)
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
        with open(os.path.join("results", f"benchmarks_my_{thread}.txt"), "w") as fMy:
            for filename in vbr_files:
                fname = filename[:-len(".vbr")]
                spmv_file = fname + ".c"
                print(filename, flush=True)
                # compile the generated code for SpMV operation
                vbr_spmv_codegen(fname, threads=thread)
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
