import resource
import signal
import numpy as np
from functools import wraps




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


def calculate_weighted_speedup_stats(individual_format_region_timings):
    """
    Helper function to calculate weighted speedup statistics from format-region timings.
    
    Args:
        individual_format_region_timings: Dictionary mapping region_id to region data containing
                                         'percentage_of_total_nnz' and 'predicted_speedup' keys
    
    Returns:
        Tuple of (total_weighted_speedup, total_format_nnz_percentage)
    """
    total_weighted_speedup = 0.0
    total_format_nnz_percentage = 0.0
    
    for region_id, region_data in individual_format_region_timings.items():
        nnz_percentage = region_data.get("percentage_of_total_nnz", 0)
        predicted_speedup = region_data.get("predicted_speedup", 0)
        
        # Weight the predicted speedup by the percentage of nnz it handles
        weighted_speedup = predicted_speedup * (nnz_percentage / 100.0)
        total_weighted_speedup += weighted_speedup
        total_format_nnz_percentage += nnz_percentage
    
    return total_weighted_speedup, total_format_nnz_percentage


def estimate_total_speedup(individual_format_region_timings):
    """
    Estimate total speedup across the entire matrix based on individual region characteristics.
    
    Args:
        individual_format_region_timings: Dictionary mapping region_id to region data
    
    Returns:
        Estimated total speedup (float)
    """
    total_weighted_speedup, total_format_nnz_percentage = calculate_weighted_speedup_stats(
        individual_format_region_timings
    )
    
    # If no format regions exist, speedup is 1.0 (no improvement)
    if total_format_nnz_percentage == 0:
        return 1.0
    
    estimated_speedup = 1.0 + (total_weighted_speedup - (total_format_nnz_percentage / 100.0))
    
    return estimated_speedup


def estimate_format_region_speedup(individual_format_region_timings):
    """
    Estimate speedup specifically for extracted format regions only.
    
    Args:
        individual_format_region_timings: Dictionary mapping region_id to region data
    
    Returns:
        Estimated format-region speedup (float)
    """
    total_weighted_speedup, total_format_nnz_percentage = calculate_weighted_speedup_stats(
        individual_format_region_timings
    )
    
    # If no format regions exist, speedup is 1.0 (no improvement)
    if total_format_nnz_percentage == 0:
        return 1.0

    estimated_region_speedup = total_weighted_speedup / (total_format_nnz_percentage / 100.0)
    
    return estimated_region_speedup
