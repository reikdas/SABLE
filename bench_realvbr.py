import os
import subprocess
import time

from src.codegen import vbr_spmv_codegen, vbr_spmm_codegen
from src.fileio import write_dense_matrix, write_dense_vector
from src.fileio import read_vbr, cleanup
import pathlib
import multiprocessing as mp
from multiprocessing import cpu_count
import tqdm
from functools import partial

FILEPATH = pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 1

def bench_spmv():
    vbr_dir = "Real_vbr"
    codegen_dir = "Generated_SpMV_real"
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
        with open(os.path.join("results", f"benchmarks_spmv_real_{thread}.csv"), "w") as fMy:
            for filename in vbr_files:
                val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(vbr_dir, filename))
                write_dense_vector(1.0, rpntr[-1])
                fname = filename[:-len(".vbr")]
                if fname[-1] == "_":
                    print("Skipping char", fname[:-1])
                    continue
                print(fname)
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
    cleanup("Generated_SpMV_real", "Generated_dense_tensor")

def spmm_inspect_files(vbr_dir, codegen_dir, root, filename):
    fname = os.path.splitext(filename)[0]
    spmm_file = fname + ".c"
    full_path = os.path.join(root, filename)
    source, pruning_method, sparsity, _ = full_path.split(os.path.sep)[-4:]
    vbr_dir_for_codegen = os.path.join(vbr_dir, source, pruning_method, sparsity)
    codegen_dir_path = os.path.join(codegen_dir, source, pruning_method, sparsity)
    os.makedirs(codegen_dir_path, exist_ok=True)
    # print(full_path, flush=True)
    inspector_times = []
    for i in range(BENCHMARK_FREQ):
        # print("Benchmarking inspector iteration", i, flush=True)
        spmm_codegen_time = vbr_spmm_codegen(fname, density=1, dir_name=codegen_dir_path, vbr_dir=vbr_dir_for_codegen, threads=1)
        time1 = time.time_ns() // 1_000
        # compile the generated code for SpMV operation
        subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd=codegen_dir_path)
        time2 = time.time_ns() // 1_000
        compilation_time = time2 - time1
        inspector_time = spmm_codegen_time + compilation_time
        inspector_times.append(inspector_time)
    # save inspector time (code generation + compilation) to file
    p = f"{source}/{pruning_method}/{sparsity},{fname},{','.join([str(x) for x in inspector_times])}\n"
    # print(p, flush=True)
    return p

def bench_spmm():
    src_dir = pathlib.Path(os.path.join(FILEPATH, "Real_vbr"))
    dest_dir = pathlib.Path(os.path.join(FILEPATH, "Generated_SpMM_real"))
    print("Benchmarking inspector")

    tasks = []
    for root, _, files in os.walk(src_dir):
        for filename in files:
            if filename.endswith(".vbr"):
                tasks.append((src_dir, dest_dir, root, filename))

    with mp.Pool(processes=cpu_count()) as pool, tqdm.tqdm(total=len(tasks)) as pbar:
        results = [pool.apply_async(spmm_inspect_files, args=task, callback=lambda x: pbar.update(1)) for task in tasks]
        with open(os.path.join("results", "benchmarks_inspector_spmm_real.csv"), "w") as fInspector:
            for result in results:
                fInspector.write(result.get())
                fInspector.flush()
    
    # Save results to file
    vbr_dir = "Real_vbr"
    codegen_dir = "Generated_SpMM_real"
    vbr_files = os.listdir(vbr_dir)
    print("Benchmarking executor")
    for thread in [1]:
        with open(os.path.join("results", f"benchmarks_spmm_real_{thread}.csv"), "w") as fMy:
            for filename in vbr_files:
                val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(vbr_dir, filename))
                fname = filename[:-len(".vbr")]
                if fname[-1] == "_":
                    continue
                spmm_file = fname + ".c"
                print(filename, flush=True)
                write_dense_matrix(1.0, rpntr[-1], 512)
                if thread > 1 and count == 0:
                    vbr_spmm_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=vbr_dir, threads=thread)
                    subprocess.run(["gcc", "-O3", "-pthread", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, spmm_file], cwd=codegen_dir)
                    count += 1
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
    cleanup("Generated_SpMM_real", "Generated_dense_tensor")

if __name__ == "__main__":
    bench_spmv()
    bench_spmm()
