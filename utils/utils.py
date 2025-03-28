import os
import resource
import signal
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