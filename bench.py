import subprocess
import os
import time

from src.spmv_codegen import vbr_spmv_codegen

BENCHMARK_FREQ = 5

if __name__ == "__main__":

    vbr_files = os.listdir("Generated_Data")
    
    with open("benchmarks_my.txt", "w") as fMy:
        with open("benchmarks_dense.txt", "w") as fDense:
            with open("benchmarks_inspector.txt", "w") as fInspector:

                for filename in vbr_files:
                    fname = filename[:-5]
                    spmv_file = fname + ".c"

                    print(filename, flush=True)
                    print("Benchmarking mine", flush=True)

                    inspector_times = []        
                    for i in range(BENCHMARK_FREQ):
                        # SpMV code generation by inspecting the VBR matrix
                        print("Benchmarking inspector iteration", i, flush=True)
                        spmv_codegen_time = vbr_spmv_codegen(filename)
                        time1 = time.time_ns() // 1_000
                        # compile the generated code for SpMV operation 
                        subprocess.run(["gcc", "-O3", "-o", fname, spmv_file], cwd="Generated_SpMV")
                        time2 = time.time_ns() // 1_000
                        compilation_time = time2 - time1
                        inspector_time = spmv_codegen_time + compilation_time
                        inspector_times.append(inspector_time)
                    # save inspector time (code generation + compilation) to file
                    p = f"{fname},{','.join([str(x) for x in inspector_times])}\n"
                    print(p, flush = True)
                    fInspector.write(p)
                    
                    execution_times = []
                    for i in range(BENCHMARK_FREQ):
                        output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMV")
                        execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                        execution_times.append(execution_time)
                    # save execution times to file
                    p = f"{fname},{','.join(execution_times)}\n"
                    print(p)
                    fMy.write(p)
                    
                    # print("Benchmarking dense")
                    # mtx_file = os.path.join("Generated_Matrix", filename[:-2]+".mtx")
                    # subprocess.run(["gcc", "-O3", "-o", filename[:-2], "dense.c", "-lcblas"])
                    # subprocess.call(["./"+filename[:-2], mtx_file], stdout=subprocess.PIPE)
                    # sum = 0
                    # for i in range(5):
                    #     output = subprocess.run(["./"+filename[:-2], mtx_file], capture_output=True)
                    #     sum += float(output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1])
                    # avg = sum/5
                    # fDense.write(f"{filename[:-2]}: {avg}ms\n")
