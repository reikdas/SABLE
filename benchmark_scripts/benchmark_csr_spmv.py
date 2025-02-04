import os
import pathlib
import subprocess
import time
import statistics
import psutil
import sys
import tqdm as tqdm

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)
BASE_PATH = os.path.join(BASE_PATH, "..")

MTX_DIR = "/local/scratch/a/Suitesparse/"

BENCHMARK_RESULTS = os.path.join(BASE_PATH, "benchmark_results")

eval = []
skip = []

if __name__ == "__main__":
    
    # get number of threads from the arguments
    threads = int(sys.argv[1])
    
    files = []    
    for filename in os.listdir(MTX_DIR):
        mtx_file = os.path.join(MTX_DIR, filename, filename + ".mtx")
        if os.path.isfile(mtx_file):
            files.append(mtx_file)
        else:
            print(f"Skipping {filename} because it is not a file")
            
    # select matrices to evaulate from eval and skip lists
    # if both are empty, evaluate all matrices
    files_to_process = []
    for mtx_file in files:
        # get basename and remove .mtx extension
        fname = os.path.basename(mtx_file)
        fname = os.path.splitext(fname)[0]
        if len(eval) > 0 and fname not in eval:
            continue
        if fname in skip:
            continue
        files_to_process.append(mtx_file)
    
    print(f"Base path: {BASE_PATH}")
    output = subprocess.run(["mkdir", "-p", f"tmp"], cwd=BASE_PATH, check=True, capture_output=True, text=True)
    output = subprocess.run(["mkdir", "-p", f"benchmark_results"], cwd=BASE_PATH, check=True, capture_output=True, text=True)
    
    if threads > 1:
        output = subprocess.run(["g++", "-DOPENMP", "-O3", f"src/csr-spmv.cpp", "-o", f"tmp/csr-spmv", "-fopenmp"], cwd=BASE_PATH, check=True, capture_output=True, text=True)
    else:
        output = subprocess.run(["g++", "-O3", f"src/csr-spmv.cpp", "-o", f"tmp/csr-spmv", "-fopenmp"], cwd=BASE_PATH, check=True, capture_output=True, text=True)
    
    # write "Filename, Time" to the benchmark file
    # change benchmark file to have the number of threads in the name
    BENCHMARK_FILE = os.path.join(BENCHMARK_RESULTS, f"csr_spmv_{threads}.csv")
    with open(BENCHMARK_FILE, "w") as f:
        f.write("Filename, Time(us)\n")
    
    i: int = 0
    for mtx_file in tqdm.tqdm(files_to_process):
        
        fname = os.path.basename(mtx_file)
        fname = os.path.splitext(fname)[0]
        
        i += 1
        
        # execute this command
        # ./csr-spmv <matrix> <threads>'
        # to get the execution time
        output = subprocess.run([f"{BASE_PATH}/tmp/csr-spmv", mtx_file, f"{threads}"], capture_output=True, check=True, text=True)
        
        # output is in the format "Time: 587.048 us"
        # split the string and get the time taken
        time_taken = float(output.stdout.split(" ")[1])
        
        # write the filename and time to the benchmark file
        with open(BENCHMARK_FILE, "a") as f:
            f.write(f"{fname}, {time_taken}\n")

    print(f"Processed {i} files")
