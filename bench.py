import subprocess
import os
import time

from src.spmv_codegen import spmv_codegen

if __name__ == "__main__":
    d = spmv_codegen(bench=True)
    with open("benchmarks_my.txt", "w") as fMy:
        with open("benchmarks_dense.txt", "w") as fDense:
            with open("benchmarks_inspector.txt", "w") as fInspector:
                for filename in os.listdir("Generated_SpMV"):
                    if filename.endswith(".c"):
                        fMy.write(f"{filename[:-2]}")
                        print(filename)
                        print("Benchmarking mine")
                        time1 = time.time_ns() // 1_000_000
                        subprocess.run(["gcc", "-O3", "-o", filename[:-2], filename], cwd="Generated_SpMV")
                        time2 = time.time_ns() // 1_000_000
                        diff = time2 - time1
                        fInspector.write(f"{filename[:-2]},{d[filename[:-2]]+diff}\n")
                        subprocess.call(["./"+filename[:-2]], stdout=subprocess.PIPE, cwd="Generated_SpMV")
                        for i in range(5):
                            output = subprocess.run(["./"+filename[:-2]], capture_output=True, cwd="Generated_SpMV")
                            fMy.write(","+output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1])
                        fMy.write("\n")
                        # print("Benchmarking dense")
                        # mtx_file = os.path.join("Generated_Matrix", filename[:-2]+".mtx")
                        # subprocess.run(["gcc", "-O3", "-o", filename[:-2], "dense.c", "-lcblas"])
                        # subprocess.call(["./"+filename[:-2], mtx_file], stdout=subprocess.PIPE)
                        # sum = 0
                        # for i in range(5):
                        #     output = subprocess.run(["./"+filename[:-2], mtx_file], capture_output=True)
                        #     sum += float(output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1])
                        # avg = sum/5
                        # fDense.write(f"{filename[:-2]}: {avg}ms\n")
