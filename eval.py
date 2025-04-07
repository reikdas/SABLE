import os
import pathlib
import statistics
import subprocess

import numpy as np

from src.consts import CFLAGS as CFLAGS
from utils.utils import extract_mul_nums, set_ulimit

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

def remove_outliers_deciles(data):
    if len(data) < 10:  # Ensure enough data points for deciles
        return data
    
    D1 = np.percentile(data, 10)  # 10th percentile
    D9 = np.percentile(data, 90)  # 90th percentile

    return [x for x in data if D1 <= x <= D9]

def eval_single_proc(eval, codegen_dir, threads):
    for thread in threads:
        with open(os.path.join(BASE_PATH, "results", f"res_{thread}.csv"), "w") as f:
            f.write("Filename,SABLE(ns)\n")
            for fname in eval:
                l = []
                for _ in range(100):
                    output = subprocess.check_output(["taskset", "-a", "-c", "0", f"./{fname}"], cwd=codegen_dir+"_"+str(thread), preexec_fn=set_ulimit).decode("utf-8").split("\n")
                    if "warning" in output[0].lower():
                        output = output[1]
                    else:
                        output = output[0]
                    # print(output)
                    output = extract_mul_nums(output)
                    median_exec_time_unroll = statistics.median([float(x) for x in output])
                    l.append(median_exec_time_unroll)
                l = remove_outliers_deciles(l)
                print(l)
                median_exec_time_unroll = statistics.mean(l)
                print(median_exec_time_unroll)
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
    ]
    codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV")
    eval_single_proc(eval, codegen_dir, [1])
