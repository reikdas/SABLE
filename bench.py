import subprocess
import os
import time

from src.spmv_codegen import vbr_spmv_codegen

BENCHMARK_FREQ = 5

if __name__ == "__main__":
    fMy = open("benchmarks_my.txt", "w")
    fDense = open("benchmarks_dense.txt", "w")
    fInspector = open("benchmarks_inspector.txt", "w")
    vbr_files = set(os.listdir("Generated_Data"))
    spmv_code_files = filter(
        lambda filename: filename.endswith(".c"), 
        os.listdir("Generated_SpMV")
    )
    
    for filename in spmv_code_files:
        vbr_file = filename[:-2]+".data"
        if vbr_file not in vbr_files:
            raise Exception(f"File {vbr_file+'.data'} not found in Generated_Data directory")
        
        print(filename, flush=True)
        print("Benchmarking mine", flush=True)

        inspector_times = []        
        for i in range(BENCHMARK_FREQ):
            # SpMV code generation by inspecting the VBR matrix
            spmv_codegen_time = vbr_spmv_codegen(vbr_file)
            time1 = time.time_ns() // 1_000
            # compile the generated code for SpMV operation 
            subprocess.run(["gcc", "-O3", "-o", filename[:-2], filename], cwd="Generated_SpMV")
            time2 = time.time_ns() // 1_000
            compilation_time = time2 - time1
            inspector_time = spmv_codegen_time + compilation_time
            inspector_times.append(inspector_time)
        # save inspector time (code generation + compilation) to file
        fInspector.write(f"{filename[:-2]},{','.join([str(x) for x in inspector_times])}\n")
        fInspector.flush()
            
        # execute the generated code for SpMV operation, and measure the execution time
        # first call is executed to bring the code into memory
        # TODO - I think subprocesses are fresh executions, and this call here does not 
        # persist the code in memory. Maybe remove this call and just run the loop below
        subprocess.call(["./"+filename[:-2]], stdout=subprocess.PIPE, cwd="Generated_SpMV")
        execution_times = []
        for i in range(BENCHMARK_FREQ):
            output = subprocess.run(["./"+filename[:-2]], capture_output=True, cwd="Generated_SpMV")
            execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
            execution_times.append(execution_time)
        # save execution times to file
        fMy.write(f"{filename[:-2]},{','.join(execution_times)}\n")
        fMy.flush()
        
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
                        
    fMy.close()
    fDense.close()
    fInspector.close()
