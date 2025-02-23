import os
import pathlib
import statistics
import subprocess

import psutil

from src.consts import CFLAGS as CFLAGS
from utils.utils import extract_mul_nums

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5

def eval_single_proc(eval, codegen_dir):
    pid = os.getpid()
    core = psutil.Process(pid).cpu_num()
    with open(os.path.join(BASE_PATH, "results", "res.csv"), "w") as f:
        f.write("Filename,SABLE(ns)\n")
        for fname in eval:
            output = subprocess.check_output([f"./{fname}"], cwd=codegen_dir).decode("utf-8").split("\n")[0]
            output = extract_mul_nums(output)
            median_exec_time_unroll = statistics.median([float(x) for x in output])
            f.write(f"{fname},{median_exec_time_unroll}\n")
            f.flush()
            print(f"Done {fname}")

if __name__ == "__main__":
    eval = [
    # "std1_Jac3",
    # "ts-palko",
    # "dbir2",
    # "Zd_Jac2",
    # "dbir1",
    # "msc23052",
    # "heart1",
    # "nemsemm1",
    # "bcsstk36",
    "karted"
    ]
    codegen_dir = ""
    eval_single_proc(eval, codegen_dir)
