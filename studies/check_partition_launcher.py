import os
import pathlib
import subprocess
import sys

import scipy

FILEPATH = pathlib.Path(__file__).resolve().parent.parent

# Import hack - is there an alternative?
sys.path.append(str(FILEPATH))

from utils.utils import check_file_matches_parent_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

matrices = [
    "heart1",
    "nemsemm1",
    "bcsstk36",
]
skip = [
    "xenon1", "invextr1_new", "majorbasis", "gyro_k",
    "c-73", "li", "torsion1",
    "boyd1", "bcsstk37", "c-49",
    "Si0", "ex11", "c-52",
    "crashbasis", "aug3dcqp",
    "2cubes_sphere", "rajat10",
    "sparsine",
    "language",
    "raefsky4",
    "boyd2",
    "torso2"
]

mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))

if __name__ == "__main__":
    files = []
    for file_path in mtx_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
            fname = pathlib.Path(file_path).resolve().stem
            if fname not in matrices:
                continue
            # if fname in skip:
            #     continue
            # mtx = scipy.io.mmread(file_path)
            # if mtx.nnz < 1_000_000:
            #     continue
            # if mtx.nnz > 10_000_000:
            #     continue
            files.append(file_path)
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)
    num_cores = len(cpu_affinity)
    files_per_core = len(files) // num_cores
    remainder = len(files) % num_cores
    start = 0
    end = 0
    with open(os.path.join(BASE_PATH, "studies", "stats_dist.csv"), "w") as f:
        for core in range(num_cores):
            end = start + files_per_core
            if core < remainder:
                end += 1
            for file in files[start:end]:
                f.write(f"{file},")
            f.write("\n")
            start = end
    subprocess.run(["mpirun", "--cpu-list", ",".join(map(str, cpu_affinity)), "--bind-to", "cpu-list:ordered", "-np", str(num_cores), "python3", "check_partition.py"], check=True, env=dict(os.environ) | {"HWLOC_COMPONENTS": "-gl"}, cwd=os.path.join(BASE_PATH, "studies"))
    with open(os.path.join(BASE_PATH, "studies", "stats.csv"), "w") as f:
        f.write("Matrix,nnz,extra nnz,num blocks,num dense blocks,old density,new density\n")
        for core in range(num_cores):
            with open(os.path.join(BASE_PATH, "studies", f"stats_dist_{core}.csv")) as g:
                for line in g:
                    if not line.startswith("Matrix,"):
                        f.write(line)
            os.remove(os.path.join(BASE_PATH, "studies", f"stats_dist_{core}.csv"))
    os.remove(os.path.join(BASE_PATH, "studies", "stats_dist.csv"))
