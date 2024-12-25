import os
import pathlib
from tqdm import tqdm

from src.fileio import read_vbr

FILEPATH = pathlib.Path(__file__).resolve().parent

def get_mean_var(results):
  # calculate mean
  mean = round(sum(results) / len(results), 2)

  # calculate variance using a list comprehension
  var = round(sum((xi - mean) ** 2 for xi in results) / len(results), 2)
  return mean, var

def check_partition_iter(full_path):
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(full_path)
    nnz_blocks = 0
    nnz_count = []
    size = []
    count = 0
    for a in range(len(rpntr)-1):
        if bpntrb[a] == -1:
            continue
        valid_cols = bindx[bpntrb[a]:bpntre[a]]
        for b in range(len(cpntr)-1):
            if b in valid_cols:
                nnz_count.append(0)
                size.append((rpntr[a+1]-rpntr[a]) * (cpntr[b+1]-cpntr[b]))
                for i in range(count, count + size[-1]):
                    if val[i] != 0:
                        nnz_count[-1] += 1
                count += size[-1]
                nnz_blocks += 1
    assert(len(nnz_count) == len(size))
    mean_nnz, var_nnz = get_mean_var(nnz_count)
    density = [nnz_count[i] / size[i] for i in range(len(nnz_count))]
    mean_density, var_density = get_mean_var(density)
    mean_size, var_size = get_mean_var(size)
    total_size = rpntr[-1] * cpntr[-1]
    total_sparsity = round((total_size - sum(nnz_count)) / total_size, 2)
    new_sparsity = round((total_size - sum(size)) / total_size, 2)

    return rpntr, cpntr, nnz_blocks, mean_nnz, var_nnz, mean_density, var_density, mean_size, var_size, total_sparsity, new_sparsity

def check_partition(which, full: bool):
    src_dir = pathlib.Path(os.path.join(FILEPATH, which))
    
    # Get all .vbr files first to set up progress bar
    vbr_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(src_dir)
        for filename in files
        if filename.endswith(".vbr")
    ]
    
    with open(f"partitioning_{which}.txt", "w") as f:
        f.write("Filename, Rows, Cols, num_nnz_blocks, Mean of nnz/nnz_block, Var of nnz/nnz_block, Mean of density/nnz_block, Var of density/nnz_block,  Mean of size/nnz_block, Var of size/nnz_block, Total sparsity, New sparsity\n")
        # Use tqdm to wrap the file iteration
        for full_path in tqdm(vbr_files, desc="Processing VBR Files", unit="file"):
            if full:
                write_name = full_path
            else:
                write_name = os.path.basename(full_path)

            rpntr, cpntr, nnz_blocks, mean_nnz, var_nnz, mean_density, var_density, mean_size, var_size, total_sparsity, new_sparsity = check_partition_iter(full_path)

            f.write(f"{write_name}: {rpntr[-1]}, {cpntr[-1]}, {nnz_blocks}, {mean_nnz}, {var_nnz}, {mean_density}, {var_density}, {mean_size}, {var_size}, {total_sparsity}, {new_sparsity}\n")
            

if __name__ == "__main__":
    # check_partition("Real_vbr", True)
    # check_partition("Suitesparse_vbr", False)
    check_partition("tests", False)
