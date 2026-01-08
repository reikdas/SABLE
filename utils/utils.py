import os
import resource
import signal
import numpy as np
from functools import wraps


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

def extract_mul_nums(output) -> list[int]:
    output = output.split("=")[1].split(",")
    output = [x for x in output if x!=""]
    return output

def timeout(timeout_secs: int):
    def wrapper(func):
        @wraps(func)
        def time_limited(*args, **kwargs):
            # Register an handler for the timeout
            def handler(signum, frame):
                raise Exception(f"Timeout for function '{func.__name__}'")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)

            # Define a timeout for your function
            signal.alarm(timeout_secs)

            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                raise exc
            finally:
                # disable the signal alarm
                signal.alarm(0)

            return result

        return time_limited

    return wrapper

def set_ulimit():
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

def remove_outliers_deciles(data):
    """
    Remove outliers from data using deciles (10th and 90th percentiles).
    
    Args:
        data (list): List of numerical values
        
    Returns:
        list: Data with outliers removed
    """
    if len(data) < 10:  # Ensure enough data points for deciles
        return data
    
    D1 = np.percentile(data, 10)  # 10th percentile
    D9 = np.percentile(data, 90)  # 90th percentile

    return [x for x in data if D1 <= x <= D9]


def calculate_weighted_speedup_stats(individual_dense_block_timings):
    """
    Helper function to calculate weighted speedup statistics from dense block timings.
    
    Args:
        individual_dense_block_timings: Dictionary mapping block_id to block data containing
                                       'percentage_of_total_nnz' and 'predicted_speedup' keys
    
    Returns:
        Tuple of (total_weighted_speedup, total_dense_nnz_percentage)
    """
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
    """
    Estimate total speedup across the entire matrix based on individual block characteristics.
    
    Args:
        individual_dense_block_timings: Dictionary mapping block_id to block data
    
    Returns:
        Estimated total speedup (float)
    """
    total_weighted_speedup, total_dense_nnz_percentage = calculate_weighted_speedup_stats(
        individual_dense_block_timings
    )
    
    # If no dense blocks, speedup is 1.0 (no improvement)
    if total_dense_nnz_percentage == 0:
        return 1.0
    
    estimated_speedup = 1.0 + (total_weighted_speedup - (total_dense_nnz_percentage / 100.0))
    
    return estimated_speedup


def estimate_dense_speedup(individual_dense_block_timings):
    """
    Estimate speedup specifically for dense regions only.
    
    Args:
        individual_dense_block_timings: Dictionary mapping block_id to block data
    
    Returns:
        Estimated dense speedup (float)
    """
    total_weighted_speedup, total_dense_nnz_percentage = calculate_weighted_speedup_stats(
        individual_dense_block_timings
    )
    
    # If no dense blocks, speedup is 1.0 (no improvement)
    if total_dense_nnz_percentage == 0:
        return 1.0

    estimated_dense_speedup = total_weighted_speedup / (total_dense_nnz_percentage / 100.0)
    
    return estimated_dense_speedup

