import os
import pathlib
import subprocess
from multiprocessing import Pool
import time

from tqdm import tqdm
import statistics

from src.codegen import vbr_spmv_codegen
from src.autopartition import cut_indices2, similarity2, my_convert_dense_to_vbr
from studies.full_pipeline import check_file_matches_parent_dir
from utils.fileio import read_vbr, write_dense_vector

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 45

done = [
    "gre_216b", "bibd_12_5", "GD96_a", "lp_ship08l", "EX1", "G39", "g7jac020", 
    "cis-n4c6-b1", "fpga_dcop_50", "adder_dcop_50", "adder_dcop_15", "G27", 
    "bfwa398", "G38", "lpi_gran", "jagmesh8", "nemsemm2", "rail_20209", 
    "coater1", "cavity15", "n3c6-b6", "nasa2146", "n4c6-b15", "adder_dcop_33", 
    "delaunay_n11", "minnesota", "bcsstk11", "Trec6", "GD95_c", "worms20_10NN", 
    "fpga_dcop_11", "oscil_dcop_13", "ch7-8-b2", "jan99jac120", "oscil_dcop_18", 
    "lpi_qual", "tumorAntiAngiogenesis_2", "fpga_dcop_43", "n4c5-b5", 
    "fpga_dcop_26", "G53", "adder_dcop_06", "oscil_dcop_57", "ulevimin", 
    "bfwa62", "freeFlyingRobot_14", "fpga_dcop_20", "D_11", "Kohonen", 
    "n2c6-b10", "klein-b1", "Muu", "G8", "adder_dcop_39", "sstmodel", 
    "lp_pds_10", "usps_norm_5NN", "dwt_209", "west0067", "spaceStation_11", 
    "orsirr_1", "G7", "fpga_trans_02", "GD98_a", "g7jac060sc", "oscil_dcop_02", 
    "bcsstm38", "aug2d", "lshp_406", "G44", "nnc261", "dwt_512", "delaunay_n13", 
    "TSOPF_FS_b9_c6", "adder_dcop_25", "iiasa", "lock2232", "adder_dcop_30", 
    "ch7-6-b3", "cavity07", "Franz8", "can_715", "aircraft", "lp_greenbea", 
    "ukerbe1", "dermatology_5NN", "mesh3e1", "IG5-8", "n2c6-b6", "ch4-4-b3", 
    "G26", "gre_216a", "kl02", "lp_truss", "dwt_361", "lp_degen2", "cs4", 
    "cz10228", "ch6-6-b1", "foldoc", "adder_dcop_40", "deter3", 
    "kineticBatchReactor_7", "n3c6-b1", "bcsstk06", "ch7-9-b1", "Tina_DisCog", 
    "c-49", "mark3jac020sc", "Sandi_authors", "mycielskian9", "rbsa480", 
    "oscil_dcop_22", "adder_dcop_68", "fpga_dcop_03", "bayer03", "ch7-7-b2", 
    "n3c6-b9", "Tina_AskCal", "M80PI_n1", "Zewail", "lp_sctap1", "xingo3012", 
    "r05", "nemswrld", "lp_gfrd_pnc", "oscil_dcop_49", "D_10", "iprob", 
    "fs_541_1", "mesh2e1", "lp_bnl1", "g7jac020sc", "bcsstk12", "lshp_265", 
    "lpi_box1", "mark3jac060", "lp_etamacro", "west1505", "n2c6-b8", "mhdb416", 
    "n4c5-b7", "ex1", "bips07_2476", "GD02_a", "n3c6-b10", "TS", "EVA", 
    "GD06_theory", "bips07_3078", "fpga_dcop_38", "rel5", "reorientation_3", 
    "deter8", "mk10-b2", "Journals", "lpi_cplex1", "lpi_woodinfe", 
    "adder_dcop_64", "Cities", "utm300", "adder_dcop_28", "bcsstk22", 
    "GL6_D_8", "cdde3", "can_96", "mhda416", "lshp1882", "fpga_dcop_27", 
    "bcsstm06", "bcsstm07", "lp_pilotnov", "oscil_dcop_45", "bfwb62", 
    "plsk1919", "freeFlyingRobot_10", "dw1024", "lns_3937", "N_biocarta", 
    "rosen7", "dataset12mfeatfactors_10NN", "cz5108", "pde900", "coater2", 
    "ex7", "mice_10NN", "ch7-6-b2", "flower_7_4", "west0381", "circuit204", 
    "wing_nodal", "G4", "fd12"
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
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_suitesparse_thresh0.2_cut2_sim2")

threads = [1]
file_handles = {}

def foo(file_path):
    if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
        fname = pathlib.Path(file_path).resolve().stem
        if fname in skip or fname in done:
            return
        print(f"Processing {fname}")
        relative_path = file_path.relative_to(mtx_dir)
        dest_path = vbr_dir / relative_path.with_suffix(".vbr")
        # Where to write the VBR file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        my_convert_dense_to_vbr((str(file_path), str(dest_path)), 0.2, cut_indices2, similarity2)
        _, _, _, _, cpntr, _, _ = read_vbr(dest_path)
        write_dense_vector(1.0, cpntr[-1])
        for i, thread in enumerate(threads):
            codegen_dir_iter = codegen_dir + f"_{thread}thread"
            # step_bar.set_description(f"Generating code for {thread} thread(s) with no unroll")
            # vbr_spmv_codegen(fname, density=0, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
            # step_bar.update(1)
            # step_bar.set_description(f"Compiling code for {thread} thread(s) with no unroll")
            # try:
            #     subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-pthread", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
            # except subprocess.TimeoutExpired:
            #     print(f"SABLE: {thread} thread(s) compilation failed for {fname} with no unroll")
            #     for remaining_thread in threads[i:]:
            #         file_handles[remaining_thread].write(f"{fname},ERROR1,ERROR1,ERROR1,ERROR1\n")
            #     break
            # step_bar.update(1)
            # step_bar.set_description("Benchmarking SABLE with no unroll")
            # subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
            # execution_times = []
            # for _ in range(BENCHMARK_FREQ):
            #     output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
            #     execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
            #     execution_times.append(execution_time)
            # step_bar.update(1)
            # step_bar.set_description(f"Generating code for {thread} thread(s) with unroll")
            codegen_time = vbr_spmv_codegen(fname, density=8, dir_name=codegen_dir_iter, vbr_dir=dest_path.parent, threads=thread)
            # step_bar.update(1)
            # step_bar.set_description(f"Compiling code for {thread} thread(s) with unroll")
            try:
                time1 = time.time_ns() // 1_000
                subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mavx", "-mprefer-vector-width=512", "-o", fname, fname+".c"], cwd=codegen_dir_iter, check=True, timeout=COMPILE_TIMEOUT)
                time2 = time.time_ns() // 1_000
                compile_time = time2-time1
            except subprocess.TimeoutExpired:
                for remaining_thread in threads[i:]:
                    file_handles[remaining_thread].write(f"{fname},ERROR2,ERROR2,ERROR2,ERROR2\n")
                    file_handles[remaining_thread].flush()
                break
            # step_bar.update(1)
            # step_bar.set_description("Benchmarking SABLE with unroll")
            try:
                subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                execution_time_unroll = []
                for _ in range(BENCHMARK_FREQ):
                    output = subprocess.run(["./" + fname], cwd=codegen_dir_iter, capture_output=True, check=True)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    execution_time_unroll.append(execution_time)
            except subprocess.CalledProcessError:
                file_handles[thread].write(f"{fname},ERROR4,ERROR4,ERROR4,ERROR4\n")
                file_handles[thread].flush()
                break
            # step_bar.update(1)
            # step_bar.set_description("Benchmarking PSC\n")
            output = subprocess.run([f"{BASE_PATH}/../partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(thread)], capture_output=True, check=True).stdout.decode("utf-8")
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
    with Pool(21) as p:
        p.map(foo, [file_path for file_path in pbar])
        

if __name__ == "__main__":
    for thread in threads:
        file_handles[thread] = open(os.path.join(BASE_PATH, "results", f"benchmarks_spmv_suitesparse_thresh0.2_cut2_sim2_{thread}thread.csv"), "w")
        file_handles[thread].write("Filename,Codegen+Compile,SABLE unroll,PSC,Unroll speedup\n")
    bench_spmv()
    for thread in threads:
        file_handles[thread].close()
