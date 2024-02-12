import subprocess
import os
import numpy
# from interpreter import interpret
import ast
from concurrent.futures import ThreadPoolExecutor

from src.codegen import vbr_spmv_codegen, vbr_spmm_codegen

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

def write_canon_spmv(mtx_file):
    M = load_mtx(os.path.join("Generated_MMarket", mtx_file))
    output = M.dot([1] * M.shape[1])
    with open(os.path.join(dir_name_spmv, mtx_file[:-len(".mtx")]+"_canon.txt"), "w") as f:
        f.writelines(str(elem)+"\n" for elem in output)

def write_vbr_spmv(mtx_file):
    for threads in [1, 16]:
        vbr_spmv_codegen(mtx_file[:-len(".mtx")], threads=threads)
        subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-o", mtx_file[:-len(".mtx")], mtx_file[:-len(".mtx")]+".c"], cwd="Generated_SpMV")
        output = subprocess.run(["./"+mtx_file[:-len(".mtx")]], capture_output=True, cwd="Generated_SpMV")
        output = output.stdout.decode("utf-8").split("\n")[1:]
        with open(os.path.join(dir_name_spmv, mtx_file[:-len(".mtx")]+f"_{threads}_my.txt"), "w") as f:
            f.writelines(line+"\n" for line in output)

def write_canon_spmm(mtx_file):
    M = load_mtx(os.path.join("Generated_MMarket", mtx_file))
    output = M.dot(numpy.ones((M.shape[1], 512)))
    with open(os.path.join(dir_name_spmm, mtx_file[:-len(".mtx")]+"_canon.txt"), "w") as f:
        for row in output:
            f.writelines(str(elem)+"\n" for elem in row)

def write_vbr_spmm(mtx_file):
    for threads in [1]:
        vbr_spmm_codegen(mtx_file[:-len(".mtx")], threads=threads)
        subprocess.run(["gcc", "-O3", "-lpthread", "-march=native", "-o", mtx_file[:-len(".mtx")], mtx_file[:-len(".mtx")]+".c"], cwd="Generated_SpMM")
        output = subprocess.run(["./"+mtx_file[:-len(".mtx")]], capture_output=True, cwd="Generated_SpMM")
        output = output.stdout.decode("utf-8").split("\n")[1:]
        with open(os.path.join(dir_name_spmm, mtx_file[:-len(".mtx")]+f"_{threads}_my.txt"), "w") as f:
            f.writelines(line+"\n" for line in output)

if __name__ == "__main__":
    dir_name_spmv = "tests_spmv"
    if not os.path.exists(dir_name_spmv):
        os.mkdir(dir_name_spmv)
    dir_name_spmm = "tests_spmm"
    if not os.path.exists(dir_name_spmm):
        os.mkdir(dir_name_spmm)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for mtx_file in os.listdir("Generated_MMarket"):
            assert(mtx_file.endswith(".mtx"))
            print(mtx_file[:-len(".mtx")])
            # Python canonical
            executor.submit(write_canon_spmv, mtx_file)
            executor.submit(write_canon_spmm, mtx_file)
            # VBR-Codegen
            executor.submit(write_vbr_spmv, mtx_file)
            executor.submit(write_vbr_spmm, mtx_file)
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
    for test_dir in [dir_name_spmv, dir_name_spmm]:
        for filename in os.listdir(test_dir):
            if filename.endswith("_my.txt"):
                print(f"Comparing {filename[:-7]}")
                # Compare mine with canon
                print("Comparing mine with canon")
                parts = filename.split('_')
                assert(cmp_file(os.path.join(test_dir, filename), os.path.join(test_dir, '_'.join(parts[:-2])+"_canon.txt")))
                # Compare cblas with canon
                # assert(cmp_file(os.path.join(dir_name, filename[:-6]+"dense.txt"), os.path.join(dir_name, filename[:-6]+"canon.txt")))
                # Compare interp with canon
                # print("Comparing interpreter with canon")
                # assert(cmp_file(os.path.join(dir_name, filename[:-6]+"interp.txt"), os.path.join(dir_name, filename[:-6]+"canon.txt")))
