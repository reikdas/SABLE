CFLAGS = ["-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-ffast-math", "-lpthread", "-fopenmp"]
SPEEDUP_THRESH = 4.3