import os
import pathlib
import subprocess

from tqdm import tqdm

from full_pipeline import avg, check_file_matches_parent_dir
from scripts.autopartition import my_convert_dense_to_vbr
from src.codegen import vbr_spmv_codegen
from src.fileio import read_vbr, write_dense_vector

BENCHMARK_FREQ = 5

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 30

rel_matrices=[ 
'nemsemm2', 
'ch7-8-b2', 
'ulevimin', 
'LFAT5000', 
'lpi_gran', 
'jagmesh8', 
'baxter', 
'g7jac060sc', 
'lowThrust_12', 
'lp_bnl2', 
'aa5', 
'freeFlyingRobot_9', 
'lnsp3937', 
'hangGlider_4', 
'cavity17', 
'ex24', 
'lp22', 
'ex10', 
'lp_maros', 
'c-52', 
'bcsstm10', 
'aa03', 
'TF14', 
'cis-n4c6-b13', 
'cq5', 
'hangGlider_3', 
'c-18', 
'ex12', 
'qh1484', 
'lp_ship08l', 
'nasa2146', 
'delaunay_n11', 
'bcsstk11', 
'delaunay_n13', 
'cavity21', 
'lp_d6cube', 
'mycielskian11', 
'skirt', 
'lpi_greenbea', 
'freeFlyingRobot_6', 
'delaunay_n14', 
'lowThrust_7', 
'model9', 
'air03', 
'lpl3', 
'poli_large', 
'CAG_mat1916', 
'nasa2910', 
'scrs8-2r', 
'TSOPF_FS_b9_c1', 
'lpi_ceria3d', 
'brainpc2', 
'g7jac050sc', 
'bcsstk26', 
'cegb3306', 
'nug08-3rd', 
'jan99jac040sc', 
'netz4504', 
'lp_pilot_ja', 
'bayer07', 
'ex10hs', 
'cegb3024', 
'dwt_1242', 
'cavity19', 
'air06', 
'TSOPF_FS_b9_c6', 
'lp_greenbea', 
'bayer03', 
'ch7-7-b2', 
'n4c5-b7', 
'mk10-b2', 
'lp_pilotnov', 
'freeFlyingRobot_10', 
'c-41', 
'lowThrust_3', 
'lshp1561', 
'jan99jac100sc', 
'c-26', 
'freeFlyingRobot_7', 
'reorientation_5', 
'eris1176', 
's1rmt3m1', 
'c-25', 
'lp_ship12l', 
'can_1072', 
'n3c6-b8', 
'c-48', 
'freeFlyingRobot_5', 
'dwt_1005', 
'whitaker3_dual', 
'ex36', 
'rajat01', 
'rajat12', 
'TF15', 
'ukerbe1_dual', 
'lp_woodw', 
'c-24', 
'ex18', 
'jagmesh5', 
'lowThrust_5', 
'g7jac010sc', 
'sc205-2r', 
'c-31', 
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
'nemswrld', 
'iprob', 
'lp_bnl1', 
'g7jac020sc', 
'bcsstk12', 
'coater2', 
'ex7', 
'ch7-6-b2', 
'ex37', 
'hangGlider_2', 
'progas', 
'lock1074', 
'hvdc1', 
'hydr1', 
'c-32', 
'powersim', 
'lp_sierra', 
'c-46', 
'onetone2', 
'rajat22', 
'lp_pilot_we', 
'wb-cs-stanford', 
'freeFlyingRobot_2', 
'Pd', 
'mk12-b2', 
'model7', 
'delaunay_n10', 
'lpl2', 
'co9', 
'model4', 
'lowThrust_9', 
'cavity22', 
'bcsstk15', 
'nemspmm1', 
'lowThrust_4', 
'cavity20', 
'primagaz', 
'scrs8-2b', 
'model3', 
'nasa1824', 
'lowThrust_8', 
'reorientation_7', 
'ford1', 
'nemscem', 
'lowThrust_11', 
'c-30', 
'n3c6-b9', 
'ex8', 
'c-23', 
'stormg2-8', 
'poli3', 
'lowThrust_13', 
'c-19', 
'ncvxqp1', 
'jan99jac120sc', 
'n4c6-b13', 
'hangGlider_5', 
'bcsstm24', 
'stormg2-27', 
'bcsstk14', 
'n4c5-b6', 
'p6000', 
'cq9', 
'dynamicSoaringProblem_2', 
'aa3', 
'jan99jac080sc', 
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
'cavity18', 
'bcsstm27', 
'barth4', 
'piston', 
'struct4', 
'bayer04', 
'c-29', 
'spiral', 
'ch6-6-b2', 
'sherman5', 
'ex35', 
'lp_cre_d', 
'TS', 
'EVA', 
'reorientation_3', 
'lpi_cplex1', 
'reorientation_6', 
'c-33', 
'lowThrust_6', 
'poli', 
'lp_degen3', 
'case9', 
'spaceStation_10', 
'bayer05', 
'rajat26', 
'pf2177', 
'soc-sign-bitcoin-alpha', 
'sts4098', 
'bcsstk23', 
'raefsky6', 
'nsir', 
'ex9', 
'ex23', 
'OPF_3754', 
'c-28', 
'nemspmm2', 
'aa01', 
'dynamicSoaringProblem_6', 
'lp_stocfor2', 
'can_1054', 
'model10', 
'reorientation_2', 
'cavity16', 
'ch7-8-b3', 
'ex14', 
'delaunay_n15', 
'lp_ship08s', 
'ncvxqp9', 
'data', 
'c-36', 
'testbig', 
'pgp2', 
'c-21', 
'c-38', 
'mk11-b2', 
'reorientation_4', 
'nasa4704', 
'jagmesh7', 
'freeFlyingRobot_11', 
'reorientation_8', 
'freeFlyingRobot_8', 
'ex3', 
'seymourl', 
'g7jac040sc', 
'model5', 
'N_reactome', 
'co5', 
'lp_greenbeb', 
'Sieber', 
'lp_cycle', 
'bayer09', 
'hydr1c', 
'aa6', 
'c-34', 
'freeFlyingRobot_3', 
'jan99jac020sc', 
'msc01050', 
'lp_osa_07', 
'jan99jac060sc', 
'TF16', 
'c-35', 
'TF13', 
'freeFlyingRobot_4', 
'bcsstk24', 
'scrs8-2c', 
]

if __name__ == "__main__":
    # Iterate over all files in Suitesparse
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse_vbr"))
    codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_real")
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
                    subprocess.call([f"{BASE_PATH}/partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], stdout=subprocess.PIPE)
                    output = subprocess.run([f"{BASE_PATH}/partially-strided-codelet/build/DDT", "-m", str(file_path), "-n", "SPMV", "-s", "CSR", "--bench_executor", "-t", str(1)], capture_output=True, check=True).stdout.decode("utf-8")
                    psc_times = output.split("\n")[:-1]
                f.write(f"{fname}, {avg(execution_times)}, {avg(psc_times)}\n")
                f.flush()