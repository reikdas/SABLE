import json
import os
import pathlib
import re
import statistics
import subprocess
from typing import Any, Dict, List

import numpy as np
import scipy

from src.autopartition import cut_indices2_fast, similarity2_numba
from src.codegen import gen_single_threaded_spmv_spv8
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from studies.find_threshold import predict_speedup
from utils.convert_real_to_vbr import convert_sparse_to_vbrc, convert_sparse_to_vbrc_with_blocks, _write_vbrc_file
from utils.fileio import read_vbrc, write_dense_vector
from utils.utils import (check_file_matches_parent_dir,
                         remove_outliers_deciles, set_ulimit)

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))

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
    compile_cmd = ["gcc", c_file_path, "-o", output_path] + CFLAGS + MKL_FLAGS

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
    
    for _ in range(bench_freq):
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

def analyze_dense_blocks(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, op: str)->List[Dict[str, Any]]:
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
                    
                    # Calculate density based on the block size and actual non-zero count
                    block_size = i_extent * j_extent
                    # Determine slice of `val` for this dense block (indx is ordered by dense blocks)
                    start = indx[count]
                    end = indx[count+1] if (count+1) < len(indx) else len(val)
                    block_vals = val[start:end]
                    # `val` contains explicit zeros for dense blocks; count real non-zeros
                    block_nnz = int(np.count_nonzero(block_vals))
                    calc_density = (block_nnz / block_size) * 100 if block_size > 0 else 0
                    
                    predicted_speedup = predict_speedup(i_extent, j_extent, calc_density, op)
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

eval = {
    "TSOPF_FS_b9_c6": [
        (24, 3630, 24, 27),  # Block 1: Rows: [24, 3630), Cols: [24, 27)
    ],
    "bloweybl": [
        (10001, 20002, 20002, 20003),  # Block 1: Rows: [10001, 20002), Cols: [20002, 20003)
        (20002, 20003, 10001, 20002),  # Block 2: Rows: [20002, 20003), Cols: [10001, 20002)
    ],
    "brainpc2": [
        (13808, 20707, 13803, 13807),  # Block 1: Rows: [13808, 20707), Cols: [13803, 13807)
        (13803, 13807, 13808, 20707),  # Block 2: Rows: [13803, 13807), Cols: [13808, 20707)
    ],
    "c-30": [
        (4280, 5319, 0, 14),  # Block 1: Rows: [4280, 5319), Cols: [0, 14)
        (0, 14, 4280, 5319),  # Block 2: Rows: [0, 14), Cols: [4280, 5319)
    ],
    "case9": [
        (24, 3630, 24, 27),  # Block 1: Rows: [24, 3630), Cols: [24, 27)
    ],
    "eris1176": [
        (1025, 1116, 1025, 1116),  # Block 1: Rows: [1025, 1116), Cols: [1025, 1116)
        (782, 844, 782, 855),  # Block 2: Rows: [782, 844), Cols: [782, 855)
        (2, 53, 2, 53),  # Block 3: Rows: [2, 53), Cols: [2, 53)
    ],
    "hangGlider_3": [
        (2279, 10260, 5697, 5698),  # Block 1: Rows: [2279, 10260), Cols: [5697, 5698)
    ],
    "hangGlider_4": [
        (3457, 15561, 8642, 8643),  # Block 1: Rows: [3457, 15561), Cols: [8642, 8643)
    ],
    "hangGlider_5": [
        (3557, 16011, 8892, 8893),  # Block 1: Rows: [3557, 16011), Cols: [8892, 8893)
    ],
    "heart1": [
        (1433, 1657, 1433, 1657),  # Block 1: Rows: [1433, 1657), Cols: [1433, 1657)
        (149, 436, 149, 436),  # Block 2: Rows: [149, 436), Cols: [149, 436)
        (538, 709, 538, 709),  # Block 3: Rows: [538, 709), Cols: [538, 709)
        (436, 488, 1101, 1153),  # Block 4: Rows: [436, 488), Cols: [1101, 1153)
        (2044, 2194, 2044, 2194),  # Block 5: Rows: [2044, 2194), Cols: [2044, 2194)
        (2575, 2860, 2575, 2860),  # Block 6: Rows: [2575, 2860), Cols: [2575, 2860)
        (3009, 3109, 3009, 3109),  # Block 7: Rows: [3009, 3109), Cols: [3009, 3109)
        (2860, 2931, 2860, 2931),  # Block 8: Rows: [2860, 2931), Cols: [2860, 2931)
        (2933, 3008, 2933, 3008),  # Block 9: Rows: [2933, 3008), Cols: [2933, 3008)
        (3143, 3318, 3489, 3514),  # Block 10: Rows: [3143, 3318), Cols: [3489, 3514)
        (3318, 3368, 3318, 3368),  # Block 11: Rows: [3318, 3368), Cols: [3318, 3368)
        (3371, 3396, 3168, 3293),  # Block 12: Rows: [3371, 3396), Cols: [3168, 3293)
    ],
    "jendrec1": [
        (30, 2107, 0, 68),  # Block 1: Rows: [30, 2107), Cols: [0, 68)
    ],
    "lowThrust_10": [
        (10143, 17241, 10139, 10141),  # Block 1: Rows: [10143, 17241), Cols: [10139, 10141)
        (10139, 10141, 10143, 17241),  # Block 2: Rows: [10139, 10141), Cols: [10143, 17241)
    ],
    "lowThrust_11": [
        (10203, 17343, 10199, 10201),  # Block 1: Rows: [10203, 17343), Cols: [10199, 10201)
        (10199, 10201, 10203, 17343),  # Block 2: Rows: [10199, 10201), Cols: [10203, 17343)
    ],
    "lowThrust_12": [
        (10253, 17428, 10249, 10251),  # Block 1: Rows: [10253, 17428), Cols: [10249, 10251)
        (10249, 10251, 10253, 17428),  # Block 2: Rows: [10249, 10251), Cols: [10253, 17428)
    ],
    "lowThrust_13": [
        (10263, 17445, 10259, 10261),  # Block 1: Rows: [10263, 17445), Cols: [10259, 10261)
        (10259, 10261, 10263, 17445),  # Block 2: Rows: [10259, 10261), Cols: [10263, 17445)
    ],
    "lowThrust_3": [
        (3923, 6667, 3919, 3921),  # Block 1: Rows: [3923, 6667), Cols: [3919, 3921)
        (3919, 3921, 3923, 6667),  # Block 2: Rows: [3919, 3921), Cols: [3923, 6667)
    ],
    "lowThrust_4": [
        (7533, 12804, 7529, 7531),  # Block 1: Rows: [7533, 12804), Cols: [7529, 7531)
        (7529, 7531, 7533, 12804),  # Block 2: Rows: [7529, 7531), Cols: [7533, 12804)
    ],
    "lowThrust_5": [
        (9033, 15354, 9029, 9031),  # Block 1: Rows: [9033, 15354), Cols: [9029, 9031)
        (9029, 9031, 9033, 15354),  # Block 2: Rows: [9029, 9031), Cols: [9033, 15354)
    ],
    "lowThrust_6": [
        (9403, 15983, 9399, 9401),  # Block 1: Rows: [9403, 15983), Cols: [9399, 9401)
        (9399, 9401, 9403, 15983),  # Block 2: Rows: [9399, 9401), Cols: [9403, 15983)
    ],
    "lowThrust_7": [
        (9653, 16408, 9649, 9651),  # Block 1: Rows: [9653, 16408), Cols: [9649, 9651)
        (9649, 9651, 9653, 16408),  # Block 2: Rows: [9649, 9651), Cols: [9653, 16408)
    ],
    "lowThrust_8": [
        (9833, 16714, 9829, 9831),  # Block 1: Rows: [9833, 16714), Cols: [9829, 9831)
        (9829, 9831, 9833, 16714),  # Block 2: Rows: [9829, 9831), Cols: [9833, 16714)
    ],
    "lowThrust_9": [
        (10023, 17037, 10019, 10021),  # Block 1: Rows: [10023, 17037), Cols: [10019, 10021)
        (10019, 10021, 10023, 17037),  # Block 2: Rows: [10019, 10021), Cols: [10023, 17037)
    ],
    "lp_fit2p": [
        (0, 3000, 0, 1),  # Block 1: Rows: [0, 3000), Cols: [0, 1)
    ],
    "lp_wood1p": [
        (242, 243, 2, 2585),  # Block 1: Rows: [242, 243), Cols: [2, 2585)
    ],
    "nd12k": [
        (672, 861, 672, 861),  # Block 1: Rows: [672, 861), Cols: [672, 861)
        (35109, 35306, 35109, 35306),  # Block 2: Rows: [35109, 35306), Cols: [35109, 35306)
        (30996, 31104, 30996, 31104),  # Block 3: Rows: [30996, 31104), Cols: [30996, 31104)
        (12375, 12507, 12375, 12507),  # Block 4: Rows: [12375, 12507), Cols: [12375, 12507)
        (4918, 5049, 4918, 5049),  # Block 5: Rows: [4918, 5049), Cols: [4918, 5049)
        (3648, 3780, 3648, 3780),  # Block 6: Rows: [3648, 3780), Cols: [3648, 3780)
        (6676, 6807, 6676, 6807),  # Block 7: Rows: [6676, 6807), Cols: [6676, 6807)
        (6109, 6159, 6109, 6159),  # Block 8: Rows: [6109, 6159), Cols: [6109, 6159)
        (6457, 6510, 6457, 6510),  # Block 9: Rows: [6457, 6510), Cols: [6457, 6510)
        (14256, 14346, 14256, 14346),  # Block 10: Rows: [14256, 14346), Cols: [14256, 14346)
        (13032, 13148, 13032, 13148),  # Block 11: Rows: [13032, 13148), Cols: [13032, 13148)
        (13637, 13690, 13637, 13690),  # Block 12: Rows: [13637, 13690), Cols: [13637, 13690)
        (34044, 34157, 34044, 34157),  # Block 13: Rows: [34044, 34157), Cols: [34044, 34157)
    ],
    "std1_Jac2": [
        (12875, 12921, 15406, 15660),  # Block 1: Rows: [12875, 12921), Cols: [15406, 15660)
        (3389, 3439, 4094, 4344),  # Block 2: Rows: [3389, 3439), Cols: [4094, 4344)
        (2480, 2530, 3179, 3429),  # Block 3: Rows: [2480, 2530), Cols: [3179, 3429)
        (1571, 1617, 2264, 2504),  # Block 4: Rows: [1571, 1617), Cols: [2264, 2504)
        (1873, 1923, 2569, 2827),  # Block 5: Rows: [1873, 1923), Cols: [2569, 2827)
        (2782, 2812, 3484, 3742),  # Block 6: Rows: [2782, 2812), Cols: [3484, 3742)
        (10467, 10517, 11452, 11710),  # Block 7: Rows: [10467, 10517), Cols: [11452, 11710)
        (5207, 5257, 5924, 6174),  # Block 8: Rows: [5207, 5257), Cols: [5924, 6174)
        (4298, 4328, 5009, 5303),  # Block 9: Rows: [4298, 4328), Cols: [5009, 5303)
        (3691, 3721, 4399, 4693),  # Block 10: Rows: [3691, 3721), Cols: [4399, 4693)
        (4600, 4630, 5314, 5608),  # Block 11: Rows: [4600, 4630), Cols: [5314, 5608)
        (8347, 8397, 9317, 9567),  # Block 12: Rows: [8347, 8397), Cols: [9317, 9567)
        (7438, 7488, 8402, 8652),  # Block 13: Rows: [7438, 7488), Cols: [8402, 8652)
        (5509, 5539, 6229, 6523),  # Block 14: Rows: [5509, 5539), Cols: [6229, 6523)
        (7740, 7770, 8707, 9001),  # Block 15: Rows: [7740, 7770), Cols: [8707, 9001)
        (9256, 9306, 10232, 10488),  # Block 16: Rows: [9256, 9306), Cols: [10232, 10488)
        (8649, 8679, 9622, 9880),  # Block 17: Rows: [8649, 8679), Cols: [9622, 9880)
        (10165, 10215, 11147, 11397),  # Block 18: Rows: [10165, 10215), Cols: [11147, 11397)
        (9558, 9588, 10537, 10831),  # Block 19: Rows: [9558, 9588), Cols: [10537, 10831)
        (11376, 11406, 12367, 12661),  # Block 20: Rows: [11376, 11406), Cols: [12367, 12661)
        (11074, 11120, 12062, 12302),  # Block 21: Rows: [11074, 11120), Cols: [12062, 12302)
        (11983, 12029, 12977, 13217),  # Block 22: Rows: [11983, 12029), Cols: [12977, 13217)
        (12285, 12315, 13282, 13576),  # Block 23: Rows: [12285, 12315), Cols: [13282, 13576)
        (14411, 14457, 16941, 17196),  # Block 24: Rows: [14411, 14457), Cols: [16941, 17196)
        (13643, 13689, 16173, 16428),  # Block 25: Rows: [13643, 13689), Cols: [16173, 16428)
        (13131, 13177, 15663, 15901),  # Block 26: Rows: [13131, 13177), Cols: [15663, 15901)
        (13899, 13945, 16431, 16669),  # Block 27: Rows: [13899, 13945), Cols: [16431, 16669)
        (14667, 14713, 17199, 17437),  # Block 28: Rows: [14667, 14713), Cols: [17199, 17437)
    ],
    "std1_Jac3": [
        (7438, 7468, 8402, 8641),  # Block 1: Rows: [7438, 7468), Cols: [8402, 8641)
        (2480, 2510, 3179, 3418),  # Block 2: Rows: [2480, 2510), Cols: [3179, 3418)
        (1873, 1903, 2569, 2809),  # Block 3: Rows: [1873, 1903), Cols: [2569, 2809)
        (1571, 1601, 2264, 2503),  # Block 4: Rows: [1571, 1601), Cols: [2264, 2503)
        (3389, 3419, 4094, 4333),  # Block 5: Rows: [3389, 3419), Cols: [4094, 4333)
        (2782, 2812, 3484, 3724),  # Block 6: Rows: [2782, 2812), Cols: [3484, 3724)
        (3691, 3721, 4399, 4639),  # Block 7: Rows: [3691, 3721), Cols: [4399, 4639)
        (4600, 4630, 5314, 5554),  # Block 8: Rows: [4600, 4630), Cols: [5314, 5554)
        (4298, 4328, 5009, 5248),  # Block 9: Rows: [4298, 4328), Cols: [5009, 5248)
        (5207, 5237, 5924, 6164),  # Block 10: Rows: [5207, 5237), Cols: [5924, 6164)
        (5509, 5539, 6229, 6469),  # Block 11: Rows: [5509, 5539), Cols: [6229, 6469)
        (8347, 8377, 9317, 9556),  # Block 12: Rows: [8347, 8377), Cols: [9317, 9556)
        (7740, 7770, 8707, 8947),  # Block 13: Rows: [7740, 7770), Cols: [8707, 8947)
        (14411, 14441, 16941, 17180),  # Block 14: Rows: [14411, 14441), Cols: [16941, 17180)
        (9256, 9286, 10232, 10471),  # Block 15: Rows: [9256, 9286), Cols: [10232, 10471)
        (8649, 8679, 9622, 9862),  # Block 16: Rows: [8649, 8679), Cols: [9622, 9862)
        (10165, 10195, 11147, 11386),  # Block 17: Rows: [10165, 10195), Cols: [11147, 11386)
        (9558, 9588, 10537, 10777),  # Block 18: Rows: [9558, 9588), Cols: [10537, 10777)
        (11074, 11104, 12062, 12301),  # Block 19: Rows: [11074, 11104), Cols: [12062, 12301)
        (10467, 10497, 11452, 11692),  # Block 20: Rows: [10467, 10497), Cols: [11452, 11692)
        (12875, 12905, 15405, 15644),  # Block 21: Rows: [12875, 12905), Cols: [15405, 15644)
        (12285, 12315, 13282, 13522),  # Block 22: Rows: [12285, 12315), Cols: [13282, 13522)
        (11983, 12013, 12977, 13216),  # Block 23: Rows: [11983, 12013), Cols: [12977, 13216)
        (11376, 11406, 12367, 12607),  # Block 24: Rows: [11376, 11406), Cols: [12367, 12607)
        (13643, 13673, 16173, 16412),  # Block 25: Rows: [13643, 13673), Cols: [16173, 16412)
        (13131, 13161, 15661, 15901),  # Block 26: Rows: [13131, 13161), Cols: [15661, 15901)
        (13899, 13929, 16429, 16669),  # Block 27: Rows: [13899, 13929), Cols: [16429, 16669)
    ],
    "vsp_c-30_data_data": [
        (4280, 5319, 0, 14),  # Block 1: Rows: [4280, 5319), Cols: [0, 14)
        (0, 14, 4280, 5319),  # Block 2: Rows: [0, 14), Cols: [4280, 5319)
    ],
}

if __name__ == "__main__":
    output_file = os.path.join(BASE_PATH, "results", "new_new_partitioner_sable_spv8_std1_Jac3.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Start with empty results (don't load existing results)
    all_results = []
    
    # Process matrices in the order specified by eval
    for fname in eval.keys():
        # Find the corresponding .mtx file for this matrix name
        file_path = None
        for candidate_path in mtx_dir.rglob(f"{fname}.mtx"):
            if candidate_path.is_file() and check_file_matches_parent_dir(candidate_path):
                file_path = candidate_path
                break
        
        if file_path is None:
            print(f"Warning: Could not find file for matrix {fname}, skipping")
            continue
        try:
            A = scipy.io.mmread(file_path)
        except:
            print("Error reading file:", file_path)
            continue
        # Ensure the matrix supports indexing — convert read matrix (often COO) to CSC
        A = scipy.sparse.csc_matrix(A, copy=False)
        print(f"Processing {fname}")
        
        # Get matrix dimensions and non-zeros
        matrix_rows = A.shape[0]
        matrix_cols = A.shape[1]

        # Iterate over matrix to count nnz and assert matrix_nnz correctness
        matrix_nnz = 0
        for i in range(matrix_rows):
            for j in range(matrix_cols):
                if A[i, j] != 0:
                    matrix_nnz += 1

        codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_C_split")
        vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR_split"))
        relative_path = file_path.relative_to(mtx_dir)
        dest_path = vbr_dir / relative_path.with_suffix(".vbr")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
        dense_block_coords = eval.get(fname, [])
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc_with_blocks(A, dense_block_coords)
        # Write the VBRC file so the C code can read it
        _write_vbrc_file(fname, os.path.join(vbr_dir, fname), val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
        # val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(vbr_dir,f"{fname}/{fname}.vbrc"))

        # Analyze dense blocks after reading VBR data
        dense_blocks = analyze_dense_blocks(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, "spmv")

        write_dense_vector(1.0, cpntr[-1])
        if len(val) == 0:
            continue
        
        # Use C backend instead of Python backend
        gen_single_threaded_spmv_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir, fname, os.path.join(vbr_dir, fname), bench=100)

        print(f"Done {fname}")
        
        # Evaluate the generated program immediately
        print(f"Evaluating {fname}...")
        avg_sparse_time, avg_dense_time, avg_individual_block_times = eval_single_file_split_timings(fname, codegen_dir, 100)
        
        # Benchmark fully sparse variant
        print(f"Benchmarking fully sparse variant for {fname}...")
        sparse_codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_C_sparse")
        sparse_vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR_Sparse"))
        
        # Create sparse variant (no dense blocks - all converted to CSR)
        sparse_dest_path = sparse_vbr_dir / relative_path.with_suffix(".vbr")
        sparse_dest_path.parent.mkdir(parents=True, exist_ok=True)
        val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse = convert_sparse_to_vbrc_with_blocks(A, [])
        # val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse = read_vbrc(os.path.join(sparse_vbr_dir,f"{fname}/{fname}.vbrc"))

        # Write the sparse VBRC file so the C code can read it
        _write_vbrc_file(fname, os.path.join(sparse_vbr_dir, fname), val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse)

        # Assert that the sparse variant has no dense blocks
        assert len(val_sparse) == 0, f"Expected fully sparse variant for {fname}, but found {len(val_sparse)} dense blocks"
        
        # Use C backend for sparse variant too
        gen_single_threaded_spmv_spv8(val_sparse, indx_sparse, bindx_sparse, rpntr_sparse, cpntr_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse, sparse_codegen_dir, fname, os.path.join(sparse_vbr_dir, fname), bench=100)
        
        sparse_avg_sparse_time, _, _ = eval_single_file_split_timings(fname, sparse_codegen_dir, 100)
        # sparse_avg_sparse_time = 0
        
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
                "predicted_speedup": block_info.get("predicted_speedup", 0)
            }
        
        # Calculate estimated speedup based on individual block characteristics
        estimated_speedup = estimate_total_speedup(matrix_result["individual_dense_block_timings"])
        estimated_dense_speedup = estimate_dense_speedup(matrix_result["individual_dense_block_timings"])
        
        # Add the estimated speedup to the timing section
        matrix_result["timing"]["expected_speedup"] = estimated_speedup
        matrix_result["timing"]["expected_dense_speedup"] = estimated_dense_speedup
        
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
