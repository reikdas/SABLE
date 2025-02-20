import os
import pathlib
import sys

import joblib
import pandas as pd
import psutil
import scipy
from mpi4py import MPI

FILEPATH = pathlib.Path(__file__).resolve().parent.parent

# Import hack - is there an alternative?
sys.path.append(str(FILEPATH))

from src.autopartition import cut_indices2, similarity2
from utils.convert_real_to_vbr import convert_sparse_to_vbr

# from utils.utils import timeout

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
vbr_dir = os.path.join(BASE_PATH, "partition_test_vbr")

cut_indices = cut_indices2
similarity = similarity2
cut_threshold = 0.2

# PARTITION_TIMEOUT = 60 * 60 * 2
# @timeout(PARTITION_TIMEOUT)
# def vbr_convert_wrapper(A, rpntr, cpntr, fname, vbr_dir):
#     return convert_sparse_to_vbr(A, rpntr, cpntr, fname, vbr_dir)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    rank = comm.Get_rank()
    mat_dist = pd.read_csv(os.path.join(BASE_PATH, "studies", "stats_dist.csv"), header=None)
    matrices = mat_dist.iloc[rank:rank+1].values[~pd.isnull(mat_dist.iloc[rank:rank+1].values)]
    model = joblib.load(os.path.join(BASE_PATH, "models", "density_threshold_spmv.pkl"))
    with open(os.path.join(BASE_PATH,"studies","stats_dist_" + str(rank) + ".csv"), "w") as f:
        f.write("Matrix,nnz,extra nnz,num blocks,num dense blocks,old density,new density\n")
        for file_path in matrices:
            file_path = pathlib.Path(file_path)
            fname = file_path.resolve().stem
            print(f"Rank {rank}: Process {fname} on core {core}")
            try:
                mtx = scipy.io.mmread(file_path)
                mtx_size = mtx.shape[0] * mtx.shape[1]
                mtx_nnz = mtx.nnz
                A = scipy.sparse.csc_matrix(mtx, copy=False)
                cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
                val, indx, bindx, bpntrb, bpntre = convert_sparse_to_vbr(A, rpntr, cpntr, fname, vbr_dir)
                count = 0
                dense_blocks = 0
                num_blocks = 0
                total_nnz = 0
                operated_nnz = 0
                for a in range(len(rpntr)-1):
                    if bpntrb[a] == -1:
                        continue
                    valid_cols = bindx[bpntrb[a]:bpntre[a]]
                    for b in range(len(cpntr)-1):
                        if b in valid_cols:
                            sparse_count = 0
                            count2 = 0
                            dense_count = 0
                            for _ in range(rpntr[a], rpntr[a+1]):
                                for _ in range(cpntr[b], cpntr[b+1]):
                                    if val[indx[count]+count2] == 0.0:
                                        sparse_count+=1
                                    else:
                                        dense_count+=1
                                    count2+=1
                            size = sparse_count + dense_count
                            total_nnz += dense_count
                            density = (dense_count / size) * 100
                            if model.predict([[size, density]])[0] == 1:
                                dense_blocks += 1
                                operated_nnz += size
                            else:
                                operated_nnz += dense_count
                            num_blocks += 1
                            count += 1
                old_density = round(total_nnz/mtx_size, 2)
                new_density = round(operated_nnz/mtx_size, 2)
                f.write(f"{fname},{total_nnz},{operated_nnz-total_nnz},{num_blocks},{dense_blocks},{old_density},{new_density}\n")
                f.flush()
            except Exception:
                f.write(f"{fname},ERROR\n")
                f.flush()
            print(f"Done {fname} on core {core}")
