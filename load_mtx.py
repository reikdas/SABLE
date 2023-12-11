import sys
import numpy

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 load_mtx.py <mtx_file>")
        sys.exit(1)
    mtx_file = sys.argv[1]
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
    numpy.set_printoptions(linewidth=200)
    for row in M:
        print(row)
    # print(M)
                
