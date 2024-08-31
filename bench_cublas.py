import subprocess
import os

from gen_cublas import gen_spmm_cublas_file

BENCHMARK_FREQ = 5

def bench_spmm():
    vbr_files = os.listdir("Generated_VBR")
    with open(os.path.join("results", f"benchmarks_spmm_cublas.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".cu"
            print(filename, flush=True)
            # compile the generated code for SpMM operation
            gen_spmm_cublas_file(fname, dir_name="Generated_SpMM_cuBLAS", vbr_dir="Generated_VBR", mm_dir="Generated_MMarket", testing=False)
            subprocess.run(["nvcc", "-O3", "-lcublas", "-o", fname, spmm_file], cwd="Generated_SpMM_cuBLAS")
            output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_cuBLAS")
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_cuBLAS")
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

if __name__ == "__main__":
    bench_spmm()
