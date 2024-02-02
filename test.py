import subprocess
import os
import numpy
# from interpreter import interpret
import ast
from concurrent.futures import ThreadPoolExecutor

from src.spmv_codegen import vbr_spmv_codegen

def is_valid_list_string(s):
    try:
        ast.literal_eval(s)
        return True
    except (ValueError, SyntaxError):
        return False

def load_mtx(mtx_file):
    with open(mtx_file, "r") as f:
        lines = f.readlines()
        start_processing = False

        for line in lines:
            if not line.startswith("%"):
                data = line.strip().split()
                if not start_processing:
                    M = numpy.zeros((int(data[0]), int(data[1])))
                    start_processing = True
                    continue
                M[int(data[0])-1][int(data[1])-1] = float(data[2])
    return M

def cmp_file(file1, file2):
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            count = 0
            for line1, line2 in zip(f1, f2):
                line1 = line1.strip()
                line2 = line2.strip()
                count += 1
                if is_valid_list_string(line1):
                    line1 = ast.literal_eval(line1)
                    line2 = ast.literal_eval(line2)
                else:
                    line1 = float(line1)
                    line2 = float(line2)
                if line1 != line2:
                    print(count, ": ", line1, " ", line2)
                    return False
    return True

def write_canon(mtx_file):
    M = load_mtx(os.path.join("Generated_Matrix", mtx_file))
    output = M.dot([1] * M.shape[1])
    with open(os.path.join(dir_name, mtx_file[:-4]+"_canon.txt"), "w") as f:
        f.writelines(str(elem)+"\n" for elem in output)

def write_vbr(mtx_file):
    for threads in [1, 16]:
        vbr_spmv_codegen(mtx_file[:-4]+".data", threads=threads)
        subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-o", mtx_file[:-4], mtx_file[:-4]+".c"], cwd="Generated_SpMV")
        output = subprocess.run(["./"+mtx_file[:-4]], capture_output=True, cwd="Generated_SpMV")
        output = output.stdout.decode("utf-8").split("\n")[1:]
        with open(os.path.join(dir_name, mtx_file[:-4]+f"_{threads}_my.txt"), "w") as f:
            f.writelines(line+"\n" for line in output)

if __name__ == "__main__":
    dir_name = "tests"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for mtx_file in os.listdir("Generated_Matrix"):
            assert(mtx_file.endswith(".mtx"))
            print(mtx_file[:-4])
            # Python canonical
            executor.submit(write_canon, mtx_file)
            # VBR-Codegen
            executor.submit(write_vbr, mtx_file)
            # Interpreter
            # with open(os.path.join(dir_name, mtx_file[:-4]+"_interp.txt"), "w") as f:
            #     y = interpret(os.path.join("Generated_Data", mtx_file[:-4]+".data"))
            #     for elem in y:
            #         f.write(str(int(elem))+"\n")
            # CBLAS
            # subprocess.run(["gcc", "-O3", "-o", mtx_file[:-4], "dense.c", "-lcblas"])
            # output = subprocess.run(["./"+mtx_file[:-4], os.path.join("Generated_Matrix", mtx_file)], capture_output=True)
            # output = output.stdout.decode("utf-8").split("\n")[1:]
            # with open(os.path.join(dir_name, mtx_file[:-4]+"_dense.txt"), "w") as f:
            #     for line in output:
            #         f.write(line+"\n")
    for filename in os.listdir(dir_name):
        if filename.endswith("_my.txt"):
            print(f"Comparing {filename[:-7]}")
            # Compare mine with canon
            print("Comparing mine with canon")
            parts = filename.split('_')
            assert(cmp_file(os.path.join(dir_name, filename), os.path.join(dir_name, '_'.join(parts[:-2])+"_canon.txt")))
            # Compare cblas with canon
            # assert(cmp_file(os.path.join(dir_name, filename[:-6]+"dense.txt"), os.path.join(dir_name, filename[:-6]+"canon.txt")))
            # Compare interp with canon
            # print("Comparing interpreter with canon")
            # assert(cmp_file(os.path.join(dir_name, filename[:-6]+"interp.txt"), os.path.join(dir_name, filename[:-6]+"canon.txt")))
