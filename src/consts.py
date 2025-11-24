import os

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")


CFLAGS = ["/local/scratch/a/das160/SABLE-spv8/spv8-public/bin/spmv_spv8.o", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-ffast-math", "-lpthread", "-I/local/scratch/a/das160/SABLE-spv8/spv8-public/src"]

SPEEDUP_THRESH = 1.5
