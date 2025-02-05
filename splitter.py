import os
import sys
import pathlib
import shutil

def recreate_directory(dir_path):
    dir_path = pathlib.Path(dir_path)
    
    if dir_path.exists():
        shutil.rmtree(dir_path)  # Delete the directory and its contents
    
    dir_path.mkdir(parents=True)  # Create the directory


def process_file(filename, chunk_size=2000):
    with open(filename, "r") as file:
        lines = file.readlines()

    # Find indices of key lines

    idx_of_spmv = None
    idx_of_main_start = None
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "int main() {" in line:
            idx_of_main_start = i
        if "void" in line:
            idx_of_spmv = i
        if "long t1s" in line:
            start_idx = i
        if "struct timeval t2;" in line:
            end_idx = i

    # Ensure valid indices were found
    if start_idx is None or end_idx is None:
        raise ValueError("Could not find required markers in the file.")

    # Split file into three parts
    part_1 = lines[: start_idx + 1]  # Before "double lts"
    part_3 = lines[end_idx:]  # From "finished" onward
    part_2 = lines[start_idx + 1 : end_idx]  # Middle part between the markers

    # Split middle part into chunks of 2000 lines each
    middle_chunks = [
        part_2[i : i + chunk_size] for i in range(0, len(part_2), chunk_size)
    ]

    preamble = part_1[(idx_of_spmv):(idx_of_main_start)]

    return (
        (part_1[:idx_of_spmv] + part_1[idx_of_main_start:]),  # Cut out spmv from part 1
        part_3,
        middle_chunks,
        preamble,
    )


if len(sys.argv) != 3:
    print("Usage: python splitter.py <filename> <chunk_size>")
    raise Exception("Invalid number of arguments")


# Example usage
file_path = sys.argv[1]
new_name = pathlib.Path(file_path).resolve().stem
chunk_size = int(sys.argv[2])

FILEPATH = pathlib.Path(__file__).resolve().parent
out_dir = os.path.join(FILEPATH, "split-and-binaries", new_name)
recreate_directory(out_dir)

new_file_name = new_name + "_split"
part_1, part_3, middle_chunks, preamble = process_file(file_path, chunk_size=chunk_size)
include_line = [f'#include "{new_file_name}.h"\n']

calls = []
for i, code in enumerate(middle_chunks):
    name = f"{new_file_name}{i}"
    print(name)

    function_header = f"void {name}(float *y, const float* x, const float* val)" + "{\n"

    end_header = "}"

    new_file = include_line + [function_header] + code + [end_header]
    with open(os.path.join(out_dir, f"{name}.c"), "w") as c_file:
        c_file.writelines(new_file)
    calls.append(name + "(y, x, val);\n")

with open(os.path.join(out_dir, f"start_{new_file_name}.c"), "w") as c_file:
    ## TODO: I need `include_line` in the start file
    c_file.writelines(part_1 + calls + part_3)

with open(f"{out_dir}/{new_file_name}.h", "w") as h_file:
    h_file.write("inline __attribute__((always_inline))\n")
    h_file.writelines(preamble)
