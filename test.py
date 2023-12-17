import subprocess
import os

from test_canon import load_mtx

def cmp_file(file1, file2):
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            count = 0
            for line1, line2 in zip(f1, f2):
                line1 = line1.strip()
                line2 = line2.strip()
                count += 1
                if int(float(line1)) != int(float(line2)):
                    print(count, ": ", line1, " ", line2)
                    return False
    return True

if __name__ == "__main__":
    dir_name = "tests"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for mtx_file in os.listdir("Generated_Matrix"):
        assert(mtx_file.endswith(".mtx"))
        print(mtx_file[:-4])
        # Python canonical
        M = load_mtx(os.path.join("Generated_Matrix", mtx_file))
        output = M.dot([1] * M.shape[1])
        with open(os.path.join(dir_name, mtx_file[:-4]+"_canon.txt"), "w") as f:
            for elem in output:
                f.write(str(int(elem))+"\n")
        # VBR-Codegen
        subprocess.run(["gcc", "-O3", "-o", mtx_file[:-4], mtx_file[:-4]+".c"], cwd="Generated_SpMV")
        output = subprocess.run(["./"+mtx_file[:-4]], capture_output=True, cwd="Generated_SpMV")
        output = output.stdout.decode("utf-8").split("\n")[1:]
        with open(os.path.join(dir_name, mtx_file[:-4]+"_my.txt"), "w") as f:
            for line in output:
                f.write(line+"\n")
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
            assert(cmp_file(os.path.join(dir_name, filename), os.path.join(dir_name, filename[:-6]+"canon.txt")))
            # Compare cblas with canon
            # assert(cmp_file(os.path.join(dir_name, filename[:-6]+"dense.txt"), os.path.join(dir_name, filename[:-6]+"canon.txt")))
        
        