import os
import pathlib
import subprocess
from multiprocessing import Pool
import time

from tqdm import tqdm
import statistics

from src.codegen2 import vbr_spmv_codegen
from src.autopartition import cut_indices2, similarity2, my_convert_dense_to_vbr
# from studies.full_pipeline import check_file_matches_parent_dir
from utils.fileio import read_vbr, write_dense_vector

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

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 45

done = [
    "GD96_a", "lp_ship08l", "g7jac020", "cis-n4c6-b1", "fpga_dcop_50", "adder_dcop_50", 
    "adder_dcop_15", "nemsemm2", "rail_20209", "coater1", "cavity15", "n3c6-b6", 
    "worms20_10NN", "fpga_dcop_11", "oscil_dcop_13", "ch7-8-b2", "ulevimin", "bfwa62", 
    "gre_216b", "bibd_12_5", "EX1", "G39", "G27", "bfwa398", "G38", "lpi_gran", 
    "jagmesh8", "jan99jac120", "oscil_dcop_18", "lpi_qual", "tumorAntiAngiogenesis_2", 
    "fpga_dcop_43", "n4c5-b5", "fpga_dcop_26", "G53", "adder_dcop_06", "oscil_dcop_57", 
    "Kohonen", "n2c6-b10", "klein-b1", "lp_pds_10", "g7jac060sc", "oscil_dcop_02", 
    "bcsstm38", "Franz8", "can_715", "aircraft", "nasa2146", "n4c6-b15", "adder_dcop_33", 
    "delaunay_n11", "minnesota", "bcsstk11", "Trec6", "GD95_c", "Muu", "G8", 
    "adder_dcop_39", "sstmodel", "usps_norm_5NN", "dwt_209", "west0067", 
    "spaceStation_11", "orsirr_1", "G7", "fpga_trans_02", "GD98_a", "aug2d", 
    "lshp_406", "G44", "nnc261", "dwt_512", "delaunay_n13", "cavity21", "lpi_itest2", 
    "oscil_dcop_33", "lp_d6cube"
]

skip = [
    'LFAT5000',
    'poli4',
    'aug3dcqp',
    'rajat10',
    'jnlbrng1',
    'LF10000',
    'torsion1',
    'chem_master1',
    'ccc',
    'cz20468',
    'nmos3',
    'obstclae',
    'baxter',
    'copter1',
    'coupled',
    'Franz6',
]

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr_thresh0.2_cut2_sim2"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse_thresh0.2_cut2_sim2_fastc")

threads = [1]
file_handles = {}

pid = os.getpid()

# Get the CPU affinity of the current process
cpu_affinity = os.sched_getaffinity(pid)

def foo(file_path):
    if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
        fname = pathlib.Path(file_path).resolve().stem
        if fname in skip or fname in done:
            return
        print(f"Processing {fname}")
        relative_path = file_path.relative_to(mtx_dir)
        dest_path = vbr_dir / relative_path.with_suffix(".vbr")
        # Where to write the VBR file
        # dest_path.parent.mkdir(parents=True, exist_ok=True)
        # my_convert_dense_to_vbr((str(file_path), str(dest_path)), 0.2, cut_indices2, similarity2)
        # _, _, _, _, cpntr, _, _ = read_vbr(dest_path)
        # write_dense_vector(1.0, cpntr[-1])
        for i, thread in enumerate(threads):
            codegen_dir_iter = codegen_dir + f"_{thread}thread"
            try:
                codegen_time = vbr_spmv_codegen(fname, density=8, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
            except FileNotFoundError:
                # for remaining_thread in threads[i:]:
                #     file_handles[remaining_thread].write(f"{fname},ERROR3,ERROR3,ERROR3,ERROR3\n")
                #     file_handles[remaining_thread].flush()
                break
            # codegen_time = 0
            
            
            ###############################
            #### COMPILE USING SABLE ######
            ###############################
            try:
                time1 = time.time_ns() // 1_000
                # subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)), "./split.sh", "64", codegen_dir_iter + "/" + fname + ".c"], cwd=BASE_PATH, check=True, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
                time2 = time.time_ns() // 1_000
                compile_time = time2-time1
                # print(result.stdout)
                # print(result.stderr)
                print(f"Compile time: {compile_time}")
            except subprocess.CalledProcessError:
                for remaining_thread in threads[i:]:
                    file_handles[remaining_thread].write(f"{fname},ERROR1,ERROR1,ERROR1,ERROR1\n")
                    file_handles[remaining_thread].flush()
                break
            except subprocess.TimeoutExpired:
                for remaining_thread in threads[i:]:
                    file_handles[remaining_thread].write(f"{fname},ERROR2,ERROR2,ERROR2,ERROR2\n")
                    file_handles[remaining_thread].flush()
                break
            
            ###############################
            #### BENCHMARK USING SABLE ####
            ###############################
            try:
                subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)),f"{BASE_PATH}/split-and-binaries/{fname}/{fname}"], cwd=BASE_PATH, capture_output=True, check=True)
                execution_time_unroll = []
                for _ in range(BENCHMARK_FREQ):
                    output = subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)),f"{BASE_PATH}/split-and-binaries/{fname}/{fname}"], cwd=codegen_dir_iter, capture_output=True, check=True)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_time_unroll.append(execution_time)
            except subprocess.CalledProcessError:
                file_handles[thread].write(f"{fname},ERROR4,ERROR4,ERROR4,ERROR4\n")
                file_handles[thread].flush()
                break
            
            ###############################
            #### BENCHMARK USING PSC ######
            ###############################
            output = subprocess.run(["taskset", "-a", "-c", ",".join(map(str, cpu_affinity)),f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(thread)], capture_output=True, check=True).stdout.decode("utf-8")
            psc_times = output.split("\n")[:-1]
            
            
            if float(statistics.median(execution_time_unroll)) == 0:
                file_handles[thread].write(f"{fname},{codegen_time+compile_time},{statistics.median(execution_time_unroll)},{statistics.median(psc_times)},{statistics.median(psc_times)}\n")
            else:
            # step_bar.update(1)
                file_handles[thread].write(f"{fname},{codegen_time+compile_time},{statistics.median(execution_time_unroll)},{statistics.median(psc_times)},{round(float(statistics.median(psc_times))/float(statistics.median(execution_time_unroll)), 2)}\n")
            file_handles[thread].flush()
        print(f"Done {fname}")

def bench_spmv():
    # Iterate over all files in Suitesparse
    # Get total number of files to process for the progress bar
    total_files = sum(1 for file_path in mtx_dir.rglob("*") 
                     if file_path.is_file() and 
                     file_path.suffix == ".mtx" and 
                     check_file_matches_parent_dir(file_path))

    pbar = tqdm(mtx_dir.rglob("*"), total=total_files, desc="Processing suitesparse matrices", unit="matrix")
    # for file_path in pbar:
    #     foo(file_path)
    with Pool(2) as p:
        p.map(foo, [file_path for file_path in pbar])
    # for file_path in pbar:
    #     foo(file_path)
        

if __name__ == "__main__":    
    for thread in threads:
        file_handles[thread] = open(os.path.join(BASE_PATH, "results", f"benchmarks_spmv_suitesparse_thresh0.2_cut2_sim2_fastc_{thread}thread.csv"), "w")
        file_handles[thread].write("Filename,Codegen+Compile,SABLE unroll,PSC,Unroll speedup\n")
    bench_spmv()
    for thread in threads:
        file_handles[thread].close()