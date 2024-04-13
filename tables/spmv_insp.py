import pathlib
import os
import csv

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..", "..")

if __name__ == "__main__":
    d = {}
    with open(os.path.join(BASE_PATH, "SABLE", "results", "benchmarks_inspector_spmv.csv"), mode="r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix_name = row[0]
            numbers = list(map(int, row[1:]))
            average = sum(numbers)/len(numbers)/1000
            d[matrix_name] = [average]
    with open(os.path.join(BASE_PATH, "SABLE", "partially-strided-codelet", "bench_inspector_1thrds.csv"), mode="r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix_name = row[0]
            numbers = list(map(int, row[1:]))
            average = sum(numbers)/len(numbers)/1000
            d[matrix_name].append(average)
    with open(os.path.join(BASE_PATH, "SABLE", "partially-strided-codelet", "bench_inspector_16thrds.csv"), mode="r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix_name = row[0]
            numbers = list(map(int, row[1:]))
            average = sum(numbers)/len(numbers)/1000
            d[matrix_name].append(average)
    with open("spmv_inspect.txt", "w") as f:
        f.write("Matrix & SABLE (ms) & PSC 1 Thread (ms) & PSC 16 Thread (ms)\n")
        for key, val in d.items():
            f.write(key)
            for elem in val:
                f.write(" & " + str(elem))
            f.write("\n")
