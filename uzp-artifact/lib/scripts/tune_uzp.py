import os
import sys
import subprocess
from script_colors import SCRIPT_TAG, ERROR_TAG

def path_exists(path):
    return os.path.exists(path)

def build_tunner(path):
    original_dir = os.getcwd()
    executor_dir = os.path.abspath(path)

    print(f"{SCRIPT_TAG} Building executor-c...")

    command = f"""
        cd {executor_dir} && \
        make all && \
        cd {original_dir}
    """

    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"{SCRIPT_TAG} Build completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"{ERROR_TAG} Build failed: {e}")

def run_tunner(uzp_file, output_file_name, grouping_points_limit, tuner_mode):
    """Run the tunner with the generated UZP file."""

    # tunner_exec = os.path.abspath("../executor-c/spf_tunner")
    tunner_omp_exec = os.path.abspath("../executor-c/spf_tunner_omp")

    # Validate tunner binary existence
    if not os.path.exists(tunner_omp_exec):
        print(f"{ERROR_TAG} Tunner executable not found: {tunner_omp_exec}")
        return

    # Define command to execute tunner
    command = [
        tunner_omp_exec, uzp_file, output_file_name,
        str(grouping_points_limit), str(tuner_mode)
    ]

    print(f"{SCRIPT_TAG} Running tunner with: {command}")

    try:
        subprocess.run(command, check=True)
        print(f"{SCRIPT_TAG} Tunner execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"{ERROR_TAG} Tunner execution failed: {e}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py <input_uzp_file> <output_directory> <grouping_points_limit> <tuner_mode>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    grouping_points_limit = int(sys.argv[3])
    tuner_mode = int(sys.argv[4])

    # call the tunner:
    build_tunner("../executor-c")

    uzp_file = f"{input_path}"
    output_file = f"{output_path}"
    # grouping_points_limit = 128  # Set this value
    """
        Tuner Mode:
        1 - FIND_MERGE_CONSECUTIVE_ORIGINS
        2 - REORDER_ORIGINS_FAVOUR_LOCALITY
        3 - SHUFFLE_ORDER_OF_ORIGINS
    """
    # tuner_mode = 1  # set to aggregate or group contiguous co-ordinates
    # tuner_mode = 2  # set to recoder origins to achieve favour locality
    run_tunner(uzp_file, output_file, grouping_points_limit, tuner_mode)


if __name__ == "__main__":
    main()


# How to run:
# command: python3 scripts/script_generate_uzp.py <mtx file>
# Current working directory should be pldi25-ae 
