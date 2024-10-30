import subprocess
import os

from gen_scorch import gen_spmm_scorch_file

BENCHMARK_FREQ = 5

def bench_spmm():
    vbr_files = os.listdir("Generated_VBR")
    gen_dir = "Generated_SpMM_scorch"
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    with open(os.path.join("results", f"benchmarks_spmm_scorch.csv"), "w") as fMy:
        for filename in vbr_files:
            fname = filename[:-len(".vbr")]
            spmm_file = fname + ".py"
            print(filename, flush=True)
            # compile the generated code for SpMM operation
            gen_spmm_scorch_file(fname, dir_name=gen_dir, mm_dir="Generated_MMarket", testing=False)
            output = subprocess.run(["python3", spmm_file], capture_output=True, cwd=gen_dir)
            execution_times = []
            for i in range(BENCHMARK_FREQ):
                print("Benchmarking executor iteration", i, flush=True)
                output = subprocess.run(["python3", spmm_file], capture_output=True, cwd=gen_dir)
                execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                execution_times.append(execution_time)
            # save execution times to file
            p = f"{fname},{','.join(execution_times)}\n"
            print(p, flush=True)
            fMy.write(p)

if __name__ == "__main__":
    bench_spmm()
