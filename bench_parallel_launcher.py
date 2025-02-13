import os
import pathlib
import subprocess

from eval import eval_single_proc
from utils.utils import check_file_matches_parent_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))

pid = os.getpid()
cpu_affinity = os.sched_getaffinity(pid)

# eval = [
#     "rim",
#     "ns3Da",
#     "olafu",
#     "Maragal_7",
#     "bbmat",
#     "bcsstk37",
#     "nemeth26",
#     "nemeth24",
#     "li",
#     "nemsemm1",
#     "venkat25",
# ]

skip = [
    "xenon1",
    "c-73",
    "boyd1",
    "SiO",
    "crashbasis",
    "2cubes_sphere",
]

if __name__ == "__main__":
    files = []
    for file_path in mtx_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
            fname = pathlib.Path(file_path).resolve().stem
            # if fname not in eval:
            #     continue
            if fname in skip:
                continue
            files.append(file_path)
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
        f.write("Filename,Codegen(ms),Compile(ms)\n")
        for core in range(num_cores):
            with open(f"mat_dist_{core}.csv") as g:
                for line in g:
                    parts = line.strip().split(',')
                    if parts[0] == "Filename":
                        continue
                    if 'ERROR' not in parts[1] and 'ERROR' not in parts[2]:
                        f.write(line)
                        evaled_fnames.append(parts[0])
                f.write(g.read())
            os.remove(f"mat_dist_{core}.csv")
    eval_single_proc(evaled_fnames)
