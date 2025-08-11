#!/usr/bin/env python3

import os
import pathlib
import sys

# Add the parent directory to the path to import the eval_single_file_split_timings function
FILEPATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH))

from bench_suitesparse_split_timings import eval_single_file_split_timings

def time_file(fname, codegen_dir):
    bench_freq = 100  # Number of benchmark runs
    
    print(f"Timing {fname} with {bench_freq} benchmark runs...")
    print(f"Codegen directory: {codegen_dir}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if the file exists
    file_path = os.path.join(codegen_dir, f"{fname}.py")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return
    
    # Check if the directory exists
    if not os.path.exists(codegen_dir):
        print(f"Error: Directory {codegen_dir} does not exist!")
        return
    
    try:
        # Run the timing evaluation
        avg_sparse_time, avg_dense_time, avg_individual_block_times = eval_single_file_split_timings(
            fname, codegen_dir, bench_freq, extract_indiv_blocks=False
        )
        
        # Print results
        print("\n" + "="*50)
        print("TIMING RESULTS")
        print("="*50)
        print(f"Matrix: {fname}")
        print(f"Benchmark runs: {bench_freq}")
        print()
        
        print("AVERAGE TIMES (nanoseconds):")
        print(f"  Sparse time: {avg_sparse_time:.2f}")
        print(f"  Dense time:  {avg_dense_time:.2f}")
        print(f"  Total time:  {avg_sparse_time + avg_dense_time:.2f}")
        print()
        
        if avg_sparse_time + avg_dense_time > 0:
            sparse_percentage = (avg_sparse_time / (avg_sparse_time + avg_dense_time)) * 100
            dense_percentage = (avg_dense_time / (avg_sparse_time + avg_dense_time)) * 100
            print("PERCENTAGES:")
            print(f"  Sparse: {sparse_percentage:.1f}%")
            print(f"  Dense:  {dense_percentage:.1f}%")
            print()
        
        if avg_individual_block_times:
            print("INDIVIDUAL DENSE BLOCK TIMES:")
            for block_id, block_time in avg_individual_block_times.items():
                print(f"  Block {block_id}: {block_time:.2f} ns")
            print()
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during timing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    time_file("eris1176-3", "Generated_SpMV_Python_split")
    # time_file("Matrix_51_51_1_1_1_0_uniform", "Generated_SpMV_threshold_Dense_2")
    # time_file("Matrix_51_51_1_1_1_0_uniform2", "Generated_SpMV_threshold_Dense_2")