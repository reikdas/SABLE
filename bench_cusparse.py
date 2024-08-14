import subprocess
import os

from gen_cusparse import gen_spmv_cusparse_file, gen_spmm_cusparse_file

BENCHMARK_FREQ = 5

def bench_spmv():
    vbr_files = os.listdir("Generated_VBR")
    with open(os.path.join("results", f"benchmarks_spmv_cusparse.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".c"
            print(filename, flush=True)
            # compile the generated code for SpMV operation
            gen_spmv_cusparse_file(fname, dir_name="Generated_SpMV_cuSparse", vbr_dir="Generated_VBR", testing=False)
            subprocess.run(["nvcc", "-O3", "-o", fname, spmv_file, "-Wno-deprecated-declarations"], cwd="Generated_SpMV_cuSparse")
            output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMV_cuSparse")
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMV_cuSparse")
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

def bench_spmm():
    vbr_files = os.listdir("Generated_VBR")
    with open(os.path.join("results", f"benchmarks_spmm_cusparse.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".c"
            print(filename, flush=True)
            # compile the generated code for SpMM operation
            gen_spmm_cusparse_file(fname, dir_name="Generated_SpMM_cuSparse", vbr_dir="Generated_VBR", testing=False)
            subprocess.run(["nvcc", "-O3", "-o", fname, spmm_file, "-Wno-deprecated-declarations"], cwd="Generated_SpMM_cuSparse")
            output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_cuSparse")
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_cuSparse")
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

if __name__ == "__main__":
    bench_spmv()
    bench_spmm()
