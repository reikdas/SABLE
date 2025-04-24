import argparse
import os
import pathlib

from utils.convert_real_to_vbr import convert_vbr_to_compressed
from utils.fileio import read_vbr

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform VBR to VBR-C")
    parser.add_argument("vbr_file", type=str, help="Path to the VBR file.")
    args = parser.parse_args()
    vbr_file = args.vbr_file

    fname = pathlib.Path(vbr_file).resolve().stem
    dst_dir = os.path.join(BASE_PATH, "artifact_partition")
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_file)
    convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, dst_dir)