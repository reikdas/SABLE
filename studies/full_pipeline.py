import os
import pathlib
import subprocess
from argparse import ArgumentParser

from check_partition import check_partition_iter
from tqdm import tqdm

from scripts.testbench import Operation
from src.autopartition import my_convert_dense_to_vbr
from src.codegen import vbr_spmm_codegen, vbr_spmv_codegen
from utils.fileio import write_dense_matrix, write_dense_vector, read_vbr

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 30

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH.parent)

rel_matrices=[ 
'nemsemm2', 
'ch7-8-b2', 
'ulevimin', 
'LFAT5000', 
'lpi_gran', 
'jagmesh8', 
'baxter', 
'lowThrust_12', 
'aa5', 
'freeFlyingRobot_9', 
'lnsp3937', 
'hangGlider_4', 
'ex24', 
'ex10', 
'lp_maros', 
'bcsstm10', 
'aa03', 
'TF14', 
'cis-n4c6-b13', 
'ex12', 
'qh1484', 
'lp_ship08l', 
'nasa2146', 
'delaunay_n11', 
'bcsstk11', 
'delaunay_n13', 
'lp_d6cube', 
'mycielskian11', 
'skirt', 
'freeFlyingRobot_6', 
'delaunay_n14', 
'lowThrust_7', 
'air03', 
'poli_large', 
'CAG_mat1916', 
'nasa2910', 
'scrs8-2r', 
'lpi_ceria3d', 
'brainpc2', 
'bcsstk26', 
'cegb3306', 
'nug08-3rd', 
'lp_pilot_ja', 
'bayer07', 
'ex10hs', 
'cegb3024', 
'dwt_1242', 
'air06', 
'ch7-7-b2', 
'n4c5-b7', 
'lp_pilotnov', 
'freeFlyingRobot_10', 
'lowThrust_3', 
'lshp1561', 
'freeFlyingRobot_7', 
'eris1176', 
's1rmt3m1', 
'c-25', 
'lp_ship12l', 
'can_1072', 
'n3c6-b8', 
'freeFlyingRobot_5', 
'dwt_1005', 
'whitaker3_dual', 
'ex36', 
'rajat01', 
'ukerbe1_dual', 
'lp_woodw', 
'ex18', 
'jagmesh5', 
'lowThrust_5', 
'g7jac010sc', 
'sc205-2r', 
'bcsstk27', 
'delaunay_n12', 
'dwt_2680', 
'bcsstm39', 
'freeFlyingRobot_12', 
'bcsstk13', 
'lock2232', 
'n2c6-b6', 
'kl02', 
'foldoc', 
'g7jac020sc', 
'bcsstk12', 
'coater2', 
'ex7', 
'ch7-6-b2', 
'ex37', 
'hangGlider_2', 
'lock1074', 
'hvdc1', 
'hydr1', 
'onetone2', 
'lp_pilot_we', 
'freeFlyingRobot_2', 
'mk12-b2', 
'delaunay_n10', 
'lowThrust_9', 
'bcsstk15', 
'lowThrust_4', 
'primagaz', 
'scrs8-2b', 
'nasa1824', 
'lowThrust_8', 
'ford1', 
'lowThrust_11', 
'n3c6-b9', 
'ex8', 
'c-23', 
'lowThrust_13', 
'c-19', 
'ncvxqp1', 
'n4c6-b13', 
'hangGlider_5', 
'bcsstm24', 
'bcsstk14', 
'n4c5-b6', 
'p6000', 
'dynamicSoaringProblem_2', 
'aa3', 
'spaceShuttleEntry_3', 
'complex', 
'air04', 
'c-40', 
'lp_stocfor3', 
'cr42', 
'lock3491', 
'psse1', 
'sherman3', 
'lowThrust_10', 
'bcsstm27', 
'barth4', 
'piston', 
'struct4', 
'c-29', 
'spiral', 
'ch6-6-b2', 
'sherman5', 
'ex35', 
'TS', 
'reorientation_3', 
'lpi_cplex1', 
'c-33', 
'lowThrust_6', 
'spaceStation_10', 
'bayer05', 
'pf2177', 
'sts4098', 
'bcsstk23', 
'raefsky6', 
'nsir', 
'ex9', 
'ex23', 
'OPF_3754', 
'aa01', 
'dynamicSoaringProblem_6', 
'lp_stocfor2', 
'can_1054', 
'ch7-8-b3', 
'ex14', 
'delaunay_n15', 
'lp_ship08s', 
'data', 
'testbig', 
'pgp2', 
'c-21', 
'c-38', 
'nasa4704', 
'jagmesh7', 
'freeFlyingRobot_11', 
'freeFlyingRobot_8', 
'ex3', 
'seymourl', 
'model5', 
'Sieber', 
'hydr1c', 
'aa6', 
'c-34', 
'freeFlyingRobot_3', 
'msc01050', 
'lp_osa_07', 
'c-35', 
'TF13', 
'freeFlyingRobot_4', 
'bcsstk24', 
'scrs8-2c', 
]

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
                    # Get num rows
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
