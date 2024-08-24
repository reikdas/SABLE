import subprocess
import os

from gen_cusparse import gen_spmv_cusparse_file, gen_spmm_cusparse_file

BENCHMARK_FREQ = 5

vbr_dir = "Generated_VBR_Sparse"
mm_dir = "Generated_MMarket_Sparse"

def bench_spmv():
    dir_name = "Generated_SpMV_cuSparse_Sparse"
    vbr_files = os.listdir(vbr_dir)
    with open(os.path.join("results", f"benchmarks_spmv_cusparse_sparse.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmv_file = fname + ".c"
            print(filename, flush=True)
            # compile the generated code for SpMV operation
            gen_spmv_cusparse_file(fname, dir_name=dir_name, vbr_dir=vbr_dir, mm_dir=mm_dir, testing=False)
            subprocess.run(["nvcc", "-O3", "-lcusparse", "-o", fname, spmv_file, "-Wno-deprecated-declarations"], cwd=dir_name)
            output = subprocess.run(["./"+fname], capture_output=True, cwd=dir_name)
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd=dir_name)
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

def bench_spmm():
    dir_name = "Generated_SpMM_cuSparse_Sparse"
    vbr_files = os.listdir(vbr_dir)
    with open(os.path.join("results", f"benchmarks_spmm_cusparse_sparse.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".c"
            print(filename, flush=True)
            # compile the generated code for SpMM operation
            gen_spmm_cusparse_file(fname, dir_name=dir_name, vbr_dir=vbr_dir, mm_dir=mm_dir, testing=False)
            subprocess.run(["nvcc", "-O3", "-lcusparse", "-o", fname, spmm_file, "-Wno-deprecated-declarations"], cwd=dir_name)
            output = subprocess.run(["./"+fname], capture_output=True, cwd=dir_name)
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["./"+fname], capture_output=True, cwd=dir_name)
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

if __name__ == "__main__":
    bench_spmv()
    bench_spmm()
