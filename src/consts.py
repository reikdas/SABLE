import os

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")

MKL_PATH = os.path.join(BASE_PATH, "..", "intel", "oneapi", "mkl", "latest")
MKL_FLAGS = [f"-I{MKL_PATH}/include", f"-L{MKL_PATH}/lib/intel64", "-lmkl_rt"]
CFLAGS = ["-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-ffast-math", "-lpthread", "-fopenmp"]
SPEEDUP_THRESH = 1.5
