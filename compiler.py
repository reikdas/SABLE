import argparse
import os
import pathlib

from src.codegen import gen_single_threaded_spmv, gen_multi_threaded_spmv
from utils.fileio import write_dense_vector, read_vbrc

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SpMV code for VBR-C matrices.")
    parser.add_argument("vbrc_mat", type=str, help="Path to the VBR matrix file.")
    parser.add_argument("threads", type=str, help="Number of threads to use.")
    args = parser.parse_args()

    vbrc_mat = args.vbrc_mat
    threads = int(args.threads)

    vbrc_mat_parent = str(pathlib.Path(vbrc_mat).parent)
    vbrc_mat_name = pathlib.Path(vbrc_mat).stem
    codegen_dir = os.path.join(BASE_PATH, "artifact_codegen")

    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbrc_mat)
    write_dense_vector(1.0, cpntr[-1])

    if threads == 1:
        gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(threads), vbrc_mat_name, vbrc_mat_parent, bench=100)
    else:
        gen_multi_threaded_spmv(threads, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(threads), vbrc_mat_name, vbrc_mat_parent, bench=100)