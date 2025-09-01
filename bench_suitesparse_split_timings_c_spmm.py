import json
import os
import pathlib
import re
import statistics
import subprocess

import scipy

from src.autopartition import cut_indices2_fast, similarity2_numba
from src.codegen import gen_single_threaded_spmv, gen_single_threaded_spmm
from src.consts import CFLAGS as CFLAGS
from studies.find_threshold import predict_speedup
from utils.convert_real_to_vbr import convert_sparse_to_vbrc
from utils.fileio import read_vbrc, write_dense_vector, write_dense_matrix
from utils.utils import (check_file_matches_parent_dir,
                         remove_outliers_deciles, set_ulimit)

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_C_split")
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR_split"))

cut_indices = cut_indices2_fast
similarity = similarity2_numba
cut_threshold = 0.2

def _calculate_weighted_speedup_stats(individual_dense_block_timings):
    """Helper function to calculate weighted speedup statistics from dense block timings"""
    total_weighted_speedup = 0.0
    total_dense_nnz_percentage = 0.0
    
    for block_id, block_data in individual_dense_block_timings.items():
        nnz_percentage = block_data.get("percentage_of_total_nnz", 0)
        predicted_speedup = block_data.get("predicted_speedup", 0)
        
        # Weight the predicted speedup by the percentage of nnz it handles
        weighted_speedup = predicted_speedup * (nnz_percentage / 100.0)
        total_weighted_speedup += weighted_speedup
        total_dense_nnz_percentage += nnz_percentage
    
    return total_weighted_speedup, total_dense_nnz_percentage


def estimate_total_speedup(individual_dense_block_timings):
    """Estimate total speedup across the entire matrix based on individual block characteristics"""
    total_weighted_speedup, total_dense_nnz_percentage = _calculate_weighted_speedup_stats(individual_dense_block_timings)
    
    # If no dense blocks, speedup is 1.0 (no improvement)
    if total_dense_nnz_percentage == 0:
        return 1.0
    
    estimated_speedup = 1.0 + (total_weighted_speedup - (total_dense_nnz_percentage / 100.0))
    
    return estimated_speedup


def estimate_dense_speedup(individual_dense_block_timings):
    """Estimate speedup specifically for dense regions only"""
    total_weighted_speedup, total_dense_nnz_percentage = _calculate_weighted_speedup_stats(individual_dense_block_timings)
    
    # If no dense blocks, speedup is 1.0 (no improvement)
    if total_dense_nnz_percentage == 0:
        return 1.0

    estimated_dense_speedup = total_weighted_speedup / (total_dense_nnz_percentage / 100.0)
    
    return estimated_dense_speedup


def compile_c_program(c_file_path, output_dir):
    """Compile the C program using gcc"""
    c_file = os.path.basename(c_file_path)
    output_name = os.path.splitext(c_file)[0]
    output_path = os.path.join(output_dir, output_name)
    
    # Compile with gcc, including MKL flags if needed
    compile_cmd = ["gcc", c_file_path, "-o", output_path] + CFLAGS

    try:
        result = subprocess.run(compile_cmd, cwd=output_dir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        if result.returncode != 0:
            print(f"Compilation failed for {c_file}: {result.stderr}")
            return None
        return output_path
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout for {c_file}")
        return None
    except Exception as e:
        print(f"Compilation error for {c_file}: {e}")
        return None


def eval_single_file_split_timings(fname, codegen_dir, bench_freq: int, extract_indiv_blocks: bool = True):
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)

    sparse_times = []
    dense_times = []
    individual_dense_block_times = {}  # Dictionary to store individual block timings
    
    # First, compile the C program
    c_file_path = os.path.join(codegen_dir, f"{fname}.c")
    executable_path = compile_c_program(c_file_path, codegen_dir)
    
    if executable_path is None:
        print(f"Failed to compile {fname}, skipping evaluation")
        return 0, 0, {}
    
    for i in range(bench_freq):
        try:
            output = subprocess.check_output(["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), executable_path], cwd=codegen_dir, preexec_fn=set_ulimit).decode("utf-8").split("\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running {fname}: {e}")
            continue
        
        # Skip warning lines if present
        start_idx = 0
        if len(output) > 0 and "warning" in output[0].lower():
            start_idx = 1
        
        # Extract Sparse and Dense timing from output
        if len(output[start_idx:]) >= 2:
            sparse_line = output[start_idx:][0]
            dense_line = output[start_idx:][1]
            
            # Handle sparse times (may be empty)
            sparse_values = []
            if sparse_line.startswith('Sparse: '):
                sparse_content = sparse_line[8:].strip()  # Remove 'Sparse: ' prefix
                if sparse_content:  # Only parse if there's content
                    sparse_values = [float(x.strip()) for x in sparse_content.rstrip(',').split(',')]
            
            # Handle dense times
            dense_values = []
            dense_match = re.search(r'Dense: (.+)', dense_line)
            if dense_match:
                dense_values = [float(x.strip()) for x in dense_match.group(1).rstrip(',').split(',')]
            
            # Add times if we found any
            if sparse_values or dense_values:
                sparse_times.extend(sparse_values)
                dense_times.extend(dense_values)
            
            # Extract individual dense block timings
            if extract_indiv_blocks:
                for line in output[start_idx:]:
                    dense_block_match = re.search(r'Dense Block (\d+): (.+)', line)
                    if dense_block_match:
                        block_id = int(dense_block_match.group(1))
                        block_times_str = dense_block_match.group(2)
                        block_times = [float(x.strip()) for x in block_times_str.rstrip(',').split(',')]
                        
                        if block_id not in individual_dense_block_times:
                            individual_dense_block_times[block_id] = []
                        individual_dense_block_times[block_id].extend(block_times)
    
    # Remove outliers and calculate averages
    sparse_times = remove_outliers_deciles(sparse_times)
    dense_times = remove_outliers_deciles(dense_times)
    
    avg_sparse_time = statistics.mean(sparse_times) if sparse_times else 0
    avg_dense_time = statistics.mean(dense_times) if dense_times else 0
    
    # Calculate averages for individual dense blocks
    avg_individual_block_times = {}
    if extract_indiv_blocks:
        for block_id, times in individual_dense_block_times.items():
            times_clean = remove_outliers_deciles(times)
            avg_individual_block_times[block_id] = statistics.mean(times_clean) if times_clean else 0
    
    return avg_sparse_time, avg_dense_time, avg_individual_block_times

def analyze_dense_blocks(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks):
    """Analyze dense blocks in a matrix using VBR data structure"""
    dense_blocks = []
    
    # Count dense blocks using same logic as code generation
    nnz_block = 0
    count = 0
    
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1: 
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                if nnz_block not in ublocks:
                    # This is a dense block
                    i_extent = rpntr[a+1] - rpntr[a]
                    j_extent = cpntr[b+1] - cpntr[b]
                    
                    # Calculate density based on the block size and nnz
                    block_size = i_extent * j_extent
                    block_nnz = indx[count+1] - indx[count] if count+1 < len(indx) else len(val) - indx[count]
                    calc_density = (block_nnz / block_size) * 100
                    
                    predicted_speedup = predict_speedup(i_extent, j_extent, calc_density)
                    dense_blocks.append({
                        "rows": i_extent,
                        "cols": j_extent,
                        "density_percent": calc_density,
                        "nnz": block_nnz,
                        "predicted_speedup": predicted_speedup
                    })
                    count += 1
                nnz_block += 1
    
    return dense_blocks

eval = [
    "eris1176",
    "std1_Jac3",
    "lp_wood1p",
    "jendrec1",
    "lowThrust_5",
    "hangGlider_4",
    "brainpc2",
    "hangGlider_3",
    "lowThrust_7",
    "lowThrust_11",
    "lowThrust_3",
    "lowThrust_6",
    "lowThrust_12",
    "hangGlider_5",
    "bloweybl",
    "heart1",
    "TSOPF_FS_b9_c6",
    "Sieber",
    "case9",
    "c-30",
    "c-32",
    "freeFlyingRobot_10",
    "freeFlyingRobot_11",
    "freeFlyingRobot_12",
    "lowThrust_10",
    "lowThrust_13",
    "lowThrust_4",
    "lowThrust_8",
    "lowThrust_9",
    "lp_fit2p",
    "nd12k",
    "std1_Jac2",
    "vsp_c-30_data_data"
    ]

if __name__ == "__main__":
    output_file = os.path.join(BASE_PATH, "results", "matrix_analysis_results_c_spmm.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Start with empty results (don't load existing results)
    all_results = []
    
    for file_path in mtx_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
            fname = pathlib.Path(file_path).resolve().stem
            if fname not in eval:
                continue
            try:
                A = scipy.io.mmread(file_path)
            except:
                print("Error reading file:", file_path)
                continue
            print(f"Processing {fname}")
            
            # Get matrix dimensions and non-zeros
            matrix_rows = A.shape[0]
            matrix_cols = A.shape[1]
            matrix_nnz = A.nnz
            
            relative_path = file_path.relative_to(mtx_dir)
            dest_path = vbr_dir / relative_path.with_suffix(".vbr")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
            val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc(A, rpntr, cpntr, fname, os.path.join(vbr_dir,fname))
            # val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(vbr_dir,f"{fname}/{fname}.vbrc"))

            # Analyze dense blocks after reading VBR data
            dense_blocks = analyze_dense_blocks(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks)

            write_dense_matrix(1.0, cpntr[-1], 512)
            
            # Use C backend instead of Python backend
            gen_single_threaded_spmm(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir, fname, os.path.join(vbr_dir, fname), bench=50)

            # print(f"Done {fname}")
            
            # Evaluate the generated program immediately
            print(f"Evaluating {fname}...")
            avg_sparse_time, avg_dense_time, avg_individual_block_times = eval_single_file_split_timings(fname, codegen_dir, 15)
            
            # Benchmark fully sparse variant
            print(f"Benchmarking fully sparse variant for {fname}...")
            sparse_codegen_dir = os.path.join(BASE_PATH, "Generated_SpMM_C_sparse")
            sparse_vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR_Sparse"))
            
            # Create sparse variant
            sparse_dest_path = sparse_vbr_dir / relative_path.with_suffix(".vbr")
            sparse_dest_path.parent.mkdir(parents=True, exist_ok=True)
            # val_sparse, indx_sparse, bindx_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse = convert_sparse_to_vbrc(A, rpntr, cpntr, fname, os.path.join(sparse_vbr_dir,fname), density=100)
            val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse = read_vbrc(os.path.join(sparse_vbr_dir,f"{fname}/{fname}.vbrc"))

            # Assert that the sparse variant has no dense blocks
            assert len(val_sparse) == 0, f"Expected fully sparse variant for {fname}, but found {len(val_sparse)} dense blocks"
            
            # Use C backend for sparse variant too
            gen_single_threaded_spmm(val_sparse, indx_sparse, bindx_sparse, rpntr, cpntr, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse, sparse_codegen_dir, fname, os.path.join(sparse_vbr_dir, fname), bench=50)
            
            sparse_avg_sparse_time, _, _ = eval_single_file_split_timings(fname, sparse_codegen_dir, 15)
            
            # Calculate percentages
            total_time = avg_sparse_time + avg_dense_time
            sparse_percentage = (avg_sparse_time / total_time * 100) if total_time > 0 else 0
            dense_percentage = (avg_dense_time / total_time * 100) if total_time > 0 else 0
            speedup = (sparse_avg_sparse_time / total_time) if total_time > 0 else 0
            
            # Calculate nnz statistics
            dense_all = sum(block.get("rows", 0) * block.get("cols", 0) for block in dense_blocks)
            dense_nnz = sum(block.get("nnz", 0) for block in dense_blocks)
            sparse_nnz = matrix_nnz - dense_nnz
            extra_zeros = dense_all - dense_nnz
            dense_nnz_perc = (dense_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
            sparse_nnz_perc = (sparse_nnz / matrix_nnz * 100) if matrix_nnz > 0 else 0
            
            density_calculation = matrix_nnz / (matrix_rows * matrix_cols)
            
            # Store results in memory
            matrix_result = {
                "matrix_name": fname,
                "matrix_dimensions": {
                    "rows": matrix_rows,
                    "cols": matrix_cols,
                    "nnz": matrix_nnz,
                    "density": density_calculation
                },
                "timing": {
                    "sparse_time_ns": avg_sparse_time,
                    "dense_time_ns": avg_dense_time,
                    "total_time_ns": total_time,
                    "sparse_percentage": sparse_percentage,
                    "dense_percentage": dense_percentage,
                    "fully_sparse_time": sparse_avg_sparse_time,
                    "speedup": speedup,
                    "expected_sparse_time_ns": ((100-dense_nnz_perc)/100)*sparse_avg_sparse_time,
                    "dense_if_sparse_time_ns": sparse_avg_sparse_time - ((100-dense_nnz_perc)/100)*sparse_avg_sparse_time
                },
                "nnz": {
                    "sparse_nnz": sparse_nnz,
                    "dense_all": dense_all,
                    "dense_nnz": dense_nnz,
                    "extra_zeros": extra_zeros,
                    "dense_nnz_perc": dense_nnz_perc,
                    "sparse_nnz_perc": sparse_nnz_perc
                },
                "individual_dense_block_timings": {}
            }
            
            # Add individual dense block timings and merge with dense block analysis
            for block_id, block_time in avg_individual_block_times.items():
                # Find corresponding dense block info (block_id is 1-indexed, dense_blocks is 0-indexed)
                block_info = dense_blocks[block_id - 1] if block_id - 1 < len(dense_blocks) else {}
                
                matrix_result["individual_dense_block_timings"][f"block_{block_id}"] = {
                    "time_ns": block_time,
                    "percentage_of_total_time": (block_time / total_time * 100) if total_time > 0 else 0,
                    "percentage_of_dense_time": (block_time / avg_dense_time * 100) if avg_dense_time > 0 else 0,
                    "percentage_of_total_nnz": (block_info.get("nnz", 0) / matrix_nnz * 100) if matrix_nnz > 0 else 0,
                    "percentage_of_dense_nnz": (block_info.get("nnz", 0) / sum(block.get("nnz", 0) for block in dense_blocks) * 100) if sum(block.get("nnz", 0) for block in dense_blocks) > 0 else 0,
                    "rows": block_info.get("rows", 0),
                    "cols": block_info.get("cols", 0),
                    "density_percent": block_info.get("density_percent", 0),
                    "nnz": block_info.get("nnz", 0),
                    # "predicted_speedup": block_info.get("predicted_speedup", 0)
                    "predicted_speedup": 0
                }
            
            # Calculate estimated speedup based on individual block characteristics
            estimated_speedup = estimate_total_speedup(matrix_result["individual_dense_block_timings"])
            estimated_dense_speedup = estimate_dense_speedup(matrix_result["individual_dense_block_timings"])
            
            # Add the estimated speedup to the timing section
            # matrix_result["timing"]["expected_speedup"] = estimated_speedup
            matrix_result["timing"]["expected_speedup"] = 0
            # matrix_result["timing"]["expected_dense_speedup"] = estimated_dense_speedup
            matrix_result["timing"]["expected_dense_speedup"] = 0
            
            # Check if this matrix result already exists and replace it, otherwise append
            existing_index = None
            for i, existing_result in enumerate(all_results):
                if existing_result["matrix_name"] == fname:
                    existing_index = i
                    break
            
            if existing_index is not None:
                all_results[existing_index] = matrix_result
                print(f"Updated existing result for {fname}")
            else:
                all_results.append(matrix_result)
                print(f"Added new result for {fname}")
            
            # Write results to file after each matrix is processed
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Evaluation complete for {fname}: {len(dense_blocks)} dense blocks")
            print(f"Results written to {output_file} ({len(all_results)} matrices total)")
    
    print(f"All processing complete. Final results in {output_file}")
    
    # Print summary
    for result in all_results:
        num_dense_blocks = len(result['individual_dense_block_timings'])
        print(f"  {result['matrix_name']}: {num_dense_blocks} dense blocks, "
              f"sparse: {result['timing']['sparse_time_ns']:.0f}ns, "
              f"dense: {result['timing']['dense_time_ns']:.0f}ns, "
              f"estimated_speedup: {result['timing']['expected_speedup']:.3f}, "
              f"estimated_dense_speedup: {result['timing']['expected_dense_speedup']:.3f}")
