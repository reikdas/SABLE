import subprocess
import os
from vbrgen import spmv_codegen
import time

if __name__ == "__main__":
    d = spmv_codegen(bench=True)
    with open("benchmarks_my.txt", "w") as f1:
        with open("benchmarks_dense.txt", "w") as f2:
            with open("benchmarks_inspector.txt", "w") as f3:
                for filename in os.listdir("Generated_SpMV"):
                    if filename.endswith(".c"):
                        f1.write(f"{filename[:-2]}")
                        print(filename)
                        print("Benchmarking mine")
                        time1 = time.time_ns() // 1_000_000
                        subprocess.run(["gcc", "-O3", "-o", filename[:-2], filename], cwd="Generated_SpMV")
                        time2 = time.time_ns() // 1_000_000
                        diff = time2 - time1
                        f3.write(f"{filename[:-2]},{d[filename[:-2]]+diff}\n")
                        subprocess.call(["./"+filename[:-2]], stdout=subprocess.PIPE, cwd="Generated_SpMV")
                        for i in range(5):
                            output = subprocess.run(["./"+filename[:-2]], capture_output=True, cwd="Generated_SpMV")
                            f1.write(","+output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1])
                        f1.write("\n")
                        # print("Benchmarking dense")
                        # mtx_file = os.path.join("Generated_Matrix", filename[:-2]+".mtx")
                        # subprocess.run(["gcc", "-O3", "-o", filename[:-2], "dense.c", "-lcblas"])
                        # subprocess.call(["./"+filename[:-2], mtx_file], stdout=subprocess.PIPE)
                        # sum = 0
                        # for i in range(5):
                        #     output = subprocess.run(["./"+filename[:-2], mtx_file], capture_output=True)
                        #     sum += float(output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1])
                        # avg = sum/5
                        # f2.write(f"{filename[:-2]}: {avg}ms\n")
