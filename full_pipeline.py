import os
import pathlib
import subprocess

from check_partition import check_partition_iter
from scripts.autopartition import my_convert_dense_to_vbr
from src.codegen import vbr_spmm_codegen
from src.fileio import write_dense_matrix
from tqdm import tqdm

BENCHMARK_FREQ = 5

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

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

if __name__ == "__main__":
    # Iterate over all files in Suitesparse
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr"))
    codegen_dir = os.path.join(BASE_PATH, "Generated_SpMM_real")
    # Get total number of files to process for the progress bar
    total_files = sum(1 for file_path in mtx_dir.rglob("*") 
                     if file_path.is_file() and 
                     file_path.suffix == ".mtx" and 
                     check_file_matches_parent_dir(file_path))
    with open(f"stats.csv", "w") as f:
        f.write("Filename, Rows, Cols, num_nnz_blocks, Mean of nnz/nnz_block, Var of nnz/nnz_block, Mean of density/nnz_block, Var of density/nnz_block,  Mean of size/nnz_block, Var of size/nnz_block, Total sparsity, New sparsity, SABLE, SpReg, PSC\n")
         # Create progress bar
        pbar = tqdm(mtx_dir.rglob("*"), total=total_files, 
                   desc="Processing matrices", unit="matrix")
        for file_path in pbar:
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                pbar.set_description(f"Processing {fname}")
                try:
                    relative_path = file_path.relative_to(mtx_dir)
                    dest_path = vbr_dir / relative_path.with_suffix(".vbr")
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
                        vbr_spmm_codegen(fname, density=1, dir_name=codegen_dir, vbr_dir=os.path.dirname(dest_path), threads=1)
                        write_dense_matrix(1.0, rpntr[-1], 512)
                        step_bar.update(1)
                        # Compile it
                        step_bar.set_description("Compiling")
                        subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir, check=True)
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
                        # Benchmark SpReg
                        step_bar.set_description("Benchmarking SpReg")
                        spreg_times = []
                        start = -7
                        spreg_dir = os.path.join(BASE_PATH, "sparse-register-tiling")
                        output = subprocess.check_output(["python3", "run_matrix.py", "-m", str(file_path), "-t", str(1), "-b", "512", "-o", "temp.csv"], cwd=spreg_dir).decode("utf-8").split("\n")
                        for i in range(5):
                            spreg_times.append(output[start+i])
                        step_bar.update(1)
                        # Benchmark PSC
                        subprocess.call([f"{BASE_PATH}/partially-strided-codelet/build/demo/spmm_demo", "-m", str(file_path), "-n", "SPMM", "-s", "CSR", "--bench_executor", "-t", str(1)], stdout=subprocess.PIPE)
                        output = subprocess.run([f"{BASE_PATH}/partially-strided-codelet/build/demo/spmm_demo", "-m", str(file_path), "-n", "SPMM", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                        psc_times = output.split("\n")[:-1]
                    f.write(f"{fname}, {rpntr[-1]}, {cpntr[-1]}, {nnz_blocks}, {mean_nnz}, {var_nnz}, {mean_density}, {var_density}, {mean_size}, {var_size}, {total_sparsity}, {new_sparsity}, {avg(execution_times)}, {avg(spreg_times)}, {avg(psc_times)}\n")
                    f.flush()
                except:
                    f.write(f"{fname}, ERROR\n")
                    f.flush()
