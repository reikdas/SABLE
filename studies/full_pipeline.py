import os
import pathlib
import subprocess
from argparse import ArgumentParser
import statistics
import sys

from tqdm import tqdm

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from studies.check_partition import check_partition_iter
from src.autopartition import my_convert_dense_to_vbr
from src.codegen_model import vbr_spmm_codegen, vbr_spmv_codegen
from studies.testbench import Operation
from utils.fileio import read_vbr, write_dense_matrix, write_dense_vector

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 30

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH.parent)

def avg(l):
    return round((sum(float(x) for x in l)/len(l)), 2)

def check_file_matches_parent_dir(filepath):
    """
    Check if a file's name (without suffix) matches its parent directory name.
    
    Args:
        filepath (str): Full path to the file
        
    Returns:
        bool: True if file name (without suffix) matches parent directory name
        
    Example:
        >>> path = '/local/scratch/a/das160/SABLE/Suitesparse/GD96_a/GD96_a.mtx'
        >>> check_file_matches_parent_dir(path)
        True
    """
    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Get the parent directory name
    parent_dir = os.path.basename(os.path.dirname(filepath))
    
    return file_name == parent_dir

def full_suitesparse_spmm():
    # Iterate over all files in Suitesparse
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr_fullnums"))
    codegen_dir = os.path.join(BASE_PATH, "Generated_SpMM_fullnums")
    # Get total number of files to process for the progress bar
    total_files = sum(1 for file_path in mtx_dir.rglob("*") 
                     if file_path.is_file() and 
                     file_path.suffix == ".mtx" and 
                     check_file_matches_parent_dir(file_path))
    with open(os.path.join(BASE_PATH,"full_spmm_nums.csv"), "w") as f:
        f.write("Filename,SABLE,SABLE unroll,SpReg,PSC\n")
         # Create progress bar
        pbar = tqdm(mtx_dir.rglob("*"), total=total_files, 
                   desc="Processing matrices", unit="matrix")
        for file_path in pbar:
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                pbar.set_description(f"Processing {fname}")
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                # Where to write the VBR file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Add nested progress bars for benchmarking
                with tqdm(total=9, desc="Steps", leave=False) as step_bar:
                    # Convert matrix to VBR format
                    step_bar.set_description("Converting to VBR")
                    my_convert_dense_to_vbr((str(file_path), str(dest_path)))
                    step_bar.update(1)
                    # Check partitioning
                    step_bar.set_description("Checking partition")
                    rpntr, cpntr, nnz_blocks, mean_nnz, var_nnz, mean_density, var_density, mean_size, var_size, total_sparsity, new_sparsity = check_partition_iter(dest_path)
                    step_bar.update(1)
                    # Evaluate using SABLE
                    # Generate code
                    step_bar.set_description("Generating code for SABLE no unroll")
                    vbr_spmm_codegen(fname, density=0, dir_name=codegen_dir, vbr_dir=os.path.dirname(dest_path), threads=1)
                    write_dense_matrix(1.0, cpntr[-1], 512)
                    step_bar.update(1)
                    # Compile it
                    step_bar.set_description("Compiling SABLE no unroll")
                    try:
                        subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir, check=True, timeout=COMPILE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        f.write(f"{fname},ERROR,ERROR,ERROR,ERROR\n")
                        f.flush()
                        continue
                    step_bar.update(1)
                    # Benchmark
                    step_bar.set_description("Benchmarking SABLE no unroll")
                    subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
                    execution_times = []
                    for i in range(BENCHMARK_FREQ):
                        output = subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
                        execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                        execution_times.append(execution_time)
                    step_bar.update(1)
                    step_bar.set_description("Generating code for SABLE unroll")
                    vbr_spmm_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=os.path.dirname(dest_path), threads=1)
                    step_bar.update(1)
                    # Compile it
                    step_bar.set_description("Compiling SABLE unroll")
                    try:
                        subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir, check=True,timeout=COMPILE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        f.write(f"{fname},{statistics.median(execution_times)},ERROR,ERROR,ERROR\n")
                        f.flush()
                        continue
                    step_bar.update(1)
                    # Benchmark
                    step_bar.set_description("Benchmarking SABLE unroll")
                    subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
                    execution_times_unroll = []
                    for i in range(BENCHMARK_FREQ):
                        output = subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
                        execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                        execution_times_unroll.append(execution_time)
                    step_bar.update(1)
                    # Benchmark SpReg
                    step_bar.set_description("Benchmarking SpReg")
                    spreg_times = []
                    start = -7
                    spreg_dir = os.path.join(BASE_PATH, "..", "sparse-register-tiling")
                    output = subprocess.check_output(["python3", "run_matrix.py", "-m", str(file_path), "-t", str(1), "-b", "512", "-o", "temp.csv"], cwd=spreg_dir).decode("utf-8").split("\n")
                    for i in range(5):
                        spreg_times.append(output[start+i])
                    step_bar.update(1)
                    # Benchmark PSC
                    try:
                        output = subprocess.run([f"{BASE_PATH}/../partially-strided-codelet/build/demo/spmm_demo", "-m", str(file_path), "-n", "SPMM", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                    except subprocess.CalledProcessError:
                        f.write(f"{fname},{statistics.median(execution_times)},{statistics.median(execution_times_unroll)},{statistics.median(spreg_times)},ERROR1\n")
                        f.flush()
                        continue
                    psc_times = output.split("\n")[:-1]
                    if not all(item.isdigit() for item in psc_times):
                        f.write(f"{fname},{statistics.median(execution_times)},{statistics.median(execution_times_unroll)},{statistics.median(spreg_times)},ERROR2\n")
                        f.flush()
                        continue
                    f.write(f"{fname},{statistics.median(execution_times)},{statistics.median(execution_times_unroll)},{statistics.median(spreg_times)},{statistics.median(psc_times)}\n")
                    f.flush()

def full_suitesparse_spmv():
    # Iterate over all files in Suitesparse
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr"))
    codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse")
    # Get total number of files to process for the progress bar
    total_files = sum(1 for file_path in mtx_dir.rglob("*") 
                     if file_path.is_file() and 
                     file_path.suffix == ".mtx" and 
                     check_file_matches_parent_dir(file_path))
    with open(f"stats_spmv.csv", "w") as f:
        f.write("Filename, SABLE, PSC\n")
         # Create progress bar
        pbar = tqdm(mtx_dir.rglob("*"), total=total_files, 
                   desc="Processing matrices", unit="matrix")
        for file_path in pbar:
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                if fname not in rel_matrices:
                    continue
                pbar.set_description(f"Processing {fname}")
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                # Where to write the VBR file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Add nested progress bars for benchmarking
                with tqdm(total=4, desc="Steps", leave=False) as step_bar:
                    # Convert matrix to VBR format
                    step_bar.set_description("Converting to VBR")
                    my_convert_dense_to_vbr((str(file_path), str(dest_path)))
                    step_bar.update(1)
                    # Evaluate using SABLE
                    # Generate code
                    step_bar.set_description("Generating code")
                    vbr_spmv_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=os.path.dirname(dest_path), threads=1)
                    # Get num cols
                    _, _, _, _, cpntr, _, _ = read_vbr(dest_path)
                    write_dense_vector(1.0, cpntr[-1])
                    step_bar.update(1)
                    # Compile it
                    step_bar.set_description("Compiling")
                    try:
                        subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir, check=True, timeout=COMPILE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print("SABLE: Compilation failed for ", fname)
                        continue
                    step_bar.update(1)
                    # Benchmark
                    step_bar.set_description("Benchmarking SABLE")
                    subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
                    execution_times = []
                    for i in range(BENCHMARK_FREQ):
                        output = subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
                        execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                        execution_times.append(execution_time)
                    step_bar.update(1)
                    # Benchmark PSC
                    subprocess.call([f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], stdout=subprocess.PIPE)
                    output = subprocess.run([f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                    psc_times = output.split("\n")[:-1]
                f.write(f"{fname}, {avg(execution_times)}, {avg(psc_times)}\n")
                f.flush()

def matrix_spmv(fname: str, mtx_dir: str):
    file_path = os.path.join(mtx_dir, fname)
    fname = pathlib.Path(file_path).resolve().stem
    
    codegen_dir = pathlib.Path(os.path.join("/tmp", "SABLE"))
    dest_path = pathlib.Path(os.path.join("/tmp", "SABLE", fname+".vbr"))
    # Where to write the VBR file
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add nested progress bars for benchmarking
    with tqdm(total=6, desc="Steps", leave=False) as step_bar:
        # Convert matrix to VBR format
        step_bar.set_description("Converting to VBR")
        my_convert_dense_to_vbr((str(file_path), str(dest_path)))
        step_bar.update(1)
        # Check partitioning
        step_bar.set_description("Checking partition")
        rpntr, cpntr, nnz_blocks, mean_nnz, var_nnz, mean_density, var_density, mean_size, var_size, total_sparsity, new_sparsity = check_partition_iter(dest_path)
        step_bar.update(1)
        # Evaluate using SABLE
        # Generate code
        step_bar.set_description("Generating code")
        vbr_spmv_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=os.path.dirname(dest_path), threads=1)
        write_dense_vector(1.0, rpntr[-1])
        step_bar.update(1)
        # Compile it
        step_bar.set_description("Compiling")
        subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir, check=True)
        step_bar.update(1)
        # Benchmark
        step_bar.set_description("Benchmarking SABLE")
        subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
        execution_times = []
        for _ in range(BENCHMARK_FREQ):
            output = subprocess.run(["./" + fname], cwd=codegen_dir, capture_output=True, check=True)
            execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
            execution_times.append(execution_time)
        step_bar.update(1)
    print(f"{fname}, {rpntr[-1]}, {cpntr[-1]}, {nnz_blocks}, {avg(execution_times)}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--operation", type=Operation, choices=list(Operation), required=True)
    parser.add_argument("-m", "--matrix", type=str, required=True, help="'Suitesparse' if you want the full suitesparse suit; explicit matrix name otherwise")
    parser.add_argument("-d", "--dir", type=str, help="If explicit matrix supplied, where is it?")
    args = parser.parse_args()
    if args.operation == Operation.SPMM:
        if args.matrix.lower() == "suitesparse":
            full_suitesparse_spmm()
        else:
            raise NotImplementedError
    else:
        if args.matrix.lower() == "suitesparse":
            full_suitesparse_spmv()
        else:
            if args.dir is None:
                raise ValueError("Need to provide directory for explicit matrix")
            matrix_spmv(args.matrix, args.dir)
