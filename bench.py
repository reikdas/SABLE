import subprocess
import os

if __name__ == "__main__":
    with open("benchmarks_my.txt", "w") as f1:
        with open("benchmarks_dense.txt", "w") as f2:
            for filename in os.listdir("Generated_SpMV"):
                if filename.endswith(".c"):
                    print(filename)
                    print("Benchmarking mine")
                    subprocess.run(["gcc", "-O3", "-o", filename[:-2], filename], cwd="Generated_SpMV")
                    subprocess.call(["./"+filename[:-2]], stdout=subprocess.PIPE, cwd="Generated_SpMV")
                    sum = 0
                    for i in range(5):
                        output = subprocess.run(["./"+filename[:-2]], capture_output=True, cwd="Generated_SpMV")
                        sum += float(output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1])
                    avg = sum/5
                    f1.write(f"{filename[:-2]}: {avg}ms\n")
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
