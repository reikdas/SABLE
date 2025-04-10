import os
import pathlib
import subprocess

from utils.utils import check_file_matches_parent_dir
import scipy

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "..", "Suitesparse"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV")
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR"))

pid = os.getpid()
cpu_affinity = os.sched_getaffinity(pid)

THREADS = [1, 2, 4, 8]
eval = [
    "hangGlider_3",
    "hangGlider_4",
    "hangGlider_5",
    "lowThrust_7",
    "lowThrust_11",
    "lowThrust_12",
    "TSOPF_FS_b9_c1",
    "brainpc2",
    "TSOPF_FS_b9_c6",
    "Journals",
    "eris1176",
    "freeFlyingRobot_7",
    "freeFlyingRobot_9",
    "Zd_Jac3",
    "freeFlyingRobot_6",
    "lowThrust_3",
    "lowThrust_2",
    "lowThrust_5",
    "bloweybl",
    # "rail4284",
    "std1_Jac2",
]

skip = [
    "xenon1",
    "c-73",
    "boyd1",
    "SiO",
    "crashbasis",
    "2cubes_sphere",
]

done = []

if __name__ == "__main__":
    files = []
    fnames = []
    for file_path in mtx_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
            fname = pathlib.Path(file_path).resolve().stem
            # if fname not in eval:
            #     continue
            if fname in skip or fname in done:
                continue
            try:
                A = scipy.io.mmread(file_path)
            except:
                print("Error reading file:", file_path)
                continue
            if A.nnz > 20_000_000:
                continue
            if A.nnz < 10_000:
                continue
            files.append(file_path)
            fnames.append(fname)
            # if len(files) > 500:
            #     break
    num_cores = len(cpu_affinity)
    # Distribute the files among the cores
    files_per_core = len(files) // num_cores
    remainder = len(files) % num_cores
    start = 0
    end = 0
    with open("mat_dist.csv", "w") as f:
        for core in range(num_cores):
            end = start + files_per_core
            if core < remainder:
                end += 1
            for file in files[start:end]:
                f.write(f"{file},")
            f.write("\n")
            start = end
    subprocess.run(["mpirun", "--cpu-list", ",".join(map(str, cpu_affinity)), "--bind-to", "cpu-list:ordered", "-np", str(num_cores), "python3", "bench_launcher.py"], check=True, env=dict(os.environ) | {"HWLOC_COMPONENTS": "-gl"})
    evaled_fnames = []
    with open(os.path.join(BASE_PATH, "results", "compile_res.csv"), "w") as f:
        f.write("Filename,Partition(ms),VBR_convert(ms),Compress(ms),Codegen(ms),Compile(ms)\n")
        for core in range(num_cores):
            with open(os.path.join(BASE_PATH, f"mat_dist_{core}.csv")) as g:
                for line in g:
                    parts = line.strip().split(',')
                    if parts[0] == "Filename":
                        continue
                    if 'ERROR' not in parts[1] and 'ERROR' not in parts[2]:
                        f.write(line)
                        evaled_fnames.append(parts[0])
                    else:
                        print(parts)
                f.write(g.read())
            os.remove(f"mat_dist_{core}.csv")
    # Print fnames not in evaled_fnames
    print("No = " + str(set(fnames) - set(evaled_fnames)))
    print("Yes = " + str(evaled_fnames))
    with open(os.path.join(BASE_PATH, "matrices.txt"), "w") as f:
        f.write("No = " + str(set(fnames) - set(evaled_fnames)))
        f.write("Yes = " + str(evaled_fnames))
