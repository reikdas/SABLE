import os
import pathlib
import subprocess
import sys

FILEPATH = pathlib.Path(__file__).resolve().parent.parent

# Import hack - is there an alternative?
sys.path.append(str(FILEPATH))

from utils.utils import check_file_matches_parent_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

matrices = [
    "orbitRaising_2",
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
            # if fname not in matrices:
            #     continue
            if fname in skip:
                continue
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
