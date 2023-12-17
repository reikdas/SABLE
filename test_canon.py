import numpy

def load_mtx(mtx_file):
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
    return M