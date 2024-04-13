import pathlib
import os
import csv

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..", "..")

if __name__ == "__main__":
    d = {}
    with open(os.path.join(BASE_PATH, "SABLE", "results", "benchmarks_inspector_spmm.csv"), mode="r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix_name = row[0]
            numbers = list(map(int, row[1:]))
            average = sum(numbers)/len(numbers)/1000
            d[matrix_name] = average
    with open("spmm_inspect.txt", "w") as f:
        f.write("Matrix & SABLE (ms)\n")
        for key, val in d.items():
            f.write(key + " & " + str(val)+"\n")
