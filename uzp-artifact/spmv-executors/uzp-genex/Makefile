##
## Written by Louis-Noel Pouchet <pouchet@colostate.edu>, 2023.
##
##


## Optimization flags for all files except generic executors for unincorporated
## points.
GCC_OPT = -O2 -march=native -ffast-math -ftree-vectorize -floop-unroll-and-jam -DREPS=$(KERNEL_REPS) #-fopt-info-vec-optimized -fopt-info-vec-missed
# GCC_OPT = -O2 -march=native -ftree-vectorize -floop-unroll-and-jam -fopt-info-vec-optimized -fopt-info-vec-missed
## Optimization flags for generic executors for unincorporated points.
GCC_OPT_CSR = -O2 -march=native -ffast-math -ftree-vectorize -floop-unroll-and-jam

## Compilers:
GCC = gcc-11
GCC_OMP = gcc-11 -fopenmp

### DEBUG:
#GCC_DEBUG_FLAGS = -fopt-info-vec-all -fopt-info-vec -fopt-info-vec-missed
#GCC_OPT_FLAGS_AGGRESSIVE =  -fvect-cost-model=unlimited -fsimd-cost-model=unlimited
#GCC_OPT_FLAGS_AGGRESSIVE =  -fvect-cost-model=unlimited -fsimd-cost-model=unlimited

#GCC_OPT = -Ofast -floop-unroll-and-jam -march=native $(GCC_OPT_FLAGS_AGGRESSIVE)
#GCC = gcc $(GCC_DEBUG_FLAGS)
#GCC = gcc -g -ggdb
### !DEBUG.

## CPP flags.
# Note: need to select the smpv from either of the following
# -DGEN_EXECUTOR_SPMV_ORIGINAL
# -DGEN_EXECUTOR_SPMV_V1
# -DGEN_EXECUTOR_SPMV_V2
CPPFLAGS = -I . -DGEN_EXECUTOR_SPMV_V2 #-DPOLYBENCH_NO_FLUSH_CACHE #-DCOMPARE_OUTPUT # -DSAVE_OUTPUT #-DPOLYBENCH_NO_FLUSH_CACHE

ifeq ($(SPMV_VERSION), SPMV_ORIGINAL)
	CPPFLAGS += -DGEN_EXECUTOR_SPMV_ORIGINAL
else ifeq ($(SPMV_VERSION), SPMV_V1)
	CPPFLAGS += -DGEN_EXECUTOR_SPMV_V1
else ifeq ($(SPMV_VERSION), SPMV_V2)
	CPPFLAGS += -DGEN_EXECUTOR_SPMV_V2
endif

ifeq ($(OUTPUT_TYPE),GFLOPS)
    CPPFLAGS += -DPOLYBENCH_GFLOPS
	LDFLAGS = # No additional libraries for GFLOPS
else ifeq ($(OUTPUT_TYPE),PAPI)
    CPPFLAGS += -DPOLYBENCH_PAPI #-DPOLYBENCH_PAPI_VERBOSE
	LDFLAGS = -lpapi
endif

# SAVE_OUTPUT and OUTPUT_SIMILARITY_CHECK are mutually exclusive, we cant call both of them at once
ifeq ($(SAVE_OUTPUT),SAVE_OUTPUT)
    CPPFLAGS += -DSAVE_OUTPUT
else ifeq ($(OUTPUT_SIMILARITY_CHECK),COMPARE_OUTPUT)
    CPPFLAGS += -DCOMPARE_OUTPUT
else ifeq ($(OUTPUT_SIMILARITY_CHECK),COMPARE_INDIVIDUAL_OUTPUT_VALUE)
    CPPFLAGS += -DCOMPARE_INDIVIDUAL_OUTPUT_VALUE
endif







#CPPFLAGS = -I . -DPOLYBENCH_TIME

## Binaries.
PROG_BIN_MV = spf_matvec
PROG_BIN_SPF_GROUPING = spf_aggregator
PROG_BIN_MV_OMP = spf_matvec_omp
PROG_BIN_SPF_GROUPING_OMP = spf_aggregator_omp

## Source files.
SRC_FILES =					\
	polybench.c				\
	spf_structure.c				\
	spf_executors.c				\
	spf_executors_uninc.c

## Object files, for every binary, w/o and w/ OpenMP.
OBJ_FILES = $(SRC_FILES:%.c=%.o)
OBJ_FILES_OMP = $(SRC_FILES:%.c=%.openmp.o)
OBJ_FILES_MV = $(OBJ_FILES) spf_matvect.o
OBJ_FILES_SPF_GROUPING = $(OBJ_FILES) spf_aggregator.o
OBJ_FILES_MV_OMP = $(OBJ_FILES_OMP) spf_matvect.openmp.o
OBJ_FILES_MM = $(OBJ_FILES) spf_matmult.o
OBJ_FILES_MM_OMP = $(OBJ_FILES_OMP) spf_matmult.openmp.o
OBJ_FILES_SPF_GROUPING_OMP = $(OBJ_FILES_OMP) spf_aggregator.openmp.o


## Entry point.
all: $(PROG_BIN_MV) $(PROG_BIN_MV_OMP) $(PROG_BIN_SPF_GROUPING) $(PROG_BIN_SPF_GROUPING_OMP)

## Default rule to compile source codes w/ OpenMP support.
%.openmp.o: %.c
	$(GCC_OMP) $(GCC_OPT) $(CPPFLAGS) -c $< -o $@

## Default rule to compile source codes.
.c.o:
	$(GCC) $(GCC_OPT) $(CPPFLAGS) -c $< -o $@

## Special rule to compile CSR/COO executor with different optimization flags.
spf_executors_uninc.o: spf_executors_uninc.c
	$(GCC) $(GCC_OPT_CSR) $(CPPFLAGS) -c spf_executors_uninc.c -o spf_executors_uninc.o
spf_executors_uninc.openmp.o: spf_executors_uninc.c
	$(GCC_OMP) $(GCC_OPT_CSR) $(CPPFLAGS) -c spf_executors_uninc.c -o spf_executors_uninc.openmp.o

## General rules.
$(PROG_BIN_SPF_GROUPING): $(OBJ_FILES_SPF_GROUPING)
	$(GCC) $(GCC_OPT) $(CPPFLAGS) $(OBJ_FILES_SPF_GROUPING) -o $(PROG_BIN_SPF_GROUPING) -lm $(LDFLAGS)

$(PROG_BIN_SPF_GROUPING_OMP): $(OBJ_FILES_SPF_GROUPING_OMP)
	$(GCC_OMP) $(GCC_OPT) $(CPPFLAGS) $(OBJ_FILES_SPF_GROUPING_OMP) -o $(PROG_BIN_SPF_GROUPING_OMP) -lm $(LDFLAGS)

$(PROG_BIN_MV): $(OBJ_FILES_MV)
	$(GCC) $(GCC_OPT) $(CPPFLAGS) $(OBJ_FILES_MV) -o $(PROG_BIN_MV) -lm $(LDFLAGS)

$(PROG_BIN_MV_OMP): $(OBJ_FILES_MV_OMP)
	$(GCC_OMP) $(GCC_OPT) $(CPPFLAGS) $(OBJ_FILES_MV_OMP) -o $(PROG_BIN_MV_OMP) -lm $(LDFLAGS)

## Clean.
clean:
	rm -f $(OBJ_FILES_MM) $(OBJ_FILES_MV) $(OBJ_FILES_MM_OMP) $(OBJ_FILES_MV_OMP) $(OBJ_FILES_SPF_GROUPING) $(OBJ_FILES_SPF_GROUPING_OMP)

distclean:
	rm -f $(PROG_BIN_MV) $(PROG_BIN_MV_OMP) $(PROG_BIN_SPF_GROUPING) $(PROG_BIN_SPF_GROUPING_OMP)
