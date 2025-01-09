import os
import pathlib
from collections import namedtuple
from multiprocessing import Pool, cpu_count

import numpy as np
import tqdm
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

CSRPattern = namedtuple('CSRPattern',
                       ['nrows', 'ncols', 'nnz', 'row_ptrs', 'col_indices'])

def read_pattern(filepath):
    with open(filepath) as file:
        lines = [file.readline() for _ in range(3)]
        nrows, ncols, nnz = [int(x) for x in lines[0].split(',')]
        return CSRPattern(nrows=nrows, ncols=ncols, nnz=nnz,
                         row_ptrs=np.fromstring(lines[1], dtype=int, sep=" "),
                         col_indices=np.fromstring(lines[2], dtype=int, sep=" ")
                         )
    return None

def pattern_to_scipy_csr(csr_pattern: CSRPattern):
    nnz = len(csr_pattern.col_indices)
    return csr_matrix(([1] * nnz, csr_pattern.col_indices, csr_pattern.row_ptrs),
                     (csr_pattern.nrows, csr_pattern.ncols))

def process_file(file_info):
    """
    Process a single file. This function will be called by the process pool.
    
    Args:
        file_info (tuple): (source_path, destination_path)
    """
    try:
        src_path, dest_path = file_info
        mmwrite(dest_path, pattern_to_scipy_csr(read_pattern(src_path)))
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")
        return False

def get_file_pairs(src_dir, dest_dir, src_suffix, dst_suffix):
    """
    Generate pairs of source and destination file paths.
    """
    file_pairs = []
    src_dir = pathlib.Path(src_dir)
    dest_dir = pathlib.Path(dest_dir)
    
    for file_path in src_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix == src_suffix:
            relative_path = file_path.relative_to(src_dir)
            dest_path = dest_dir / relative_path.with_suffix(dst_suffix)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            file_pairs.append((str(file_path), str(dest_path)))
    
    return file_pairs

def parallel_dispatch(src_dir, dest_dir, num_processes, f, src_suffix, dst_suffix):
    # Get all file pairs first
    file_pairs = get_file_pairs(src_dir, dest_dir, src_suffix, dst_suffix)

    for file_pair in file_pairs:
        f(file_pair)

if __name__ == "__main__":
    src_dir = pathlib.Path(os.path.join(BASE_PATH, "dlmc"))
    dest_dir = pathlib.Path(os.path.join(BASE_PATH, "Real_mtx"))
    
    parallel_dispatch(src_dir, dest_dir, cpu_count(), process_file, ".smtx", ".mtx")
