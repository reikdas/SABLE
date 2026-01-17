# PLDI'25 Artifacts

This repository contains the artifacts for the paper:

**"Modular Construction and Optimization of the UZP Sparse Format for SpMV on CPUs"**



## Download and installation instructions

The artifact is encapsulated inside a Singularity container, included in the Zenodo package with DOI **10.5281/zenodo.15048005**. The artifact has been tested with Singularity CE v4.2.2 and v4.3.0, both on Arch Linux- and Debian-based systems.

The scripts contained in the artifact compile new executables and generate files in order to function, and therefore need the container to be writable. We suggest to use the following commands in order to start an interactive session inside the container and access the artifact folder, located in _/uzp-artifact/_:

```
$ singularity build --sandbox pldi25-uzp-artifact/ pldi25-uzp-artifact.sif
$ sudo singularity shell --writable pldi25-uzp-artifact/
Singularity> cd /uzp-artifact/
```

**IMPORTANT**: the artifact should be run as root to ensure appropriate permissions, particularly when running with `numactl` and `hugectl`.

## Contents

This artifact contains several different tools and scripts to reproduce the results in the submitted paper. The main results in Sec. 5, i.e., experimentation with SpMV kernels, can be reproduced using the scripts found in the `/uzp-artifact/spmv-executors/` folder, described in more detail below. Section [Running the SpMV executors](#running-the-spmv-executors) gives details about how to reproduce experiments. Section [Experimental setup](#experimental-setup) details the experimental setup used in the submitted paper. Note that it is fully expected that using a different setup (e.g., a different CPU) will result in different performance profiles.

For the reviewer wishing to experiment further with the toolchain, the artifact also includes the [Z-Polyhedrator](https://github.com/forcegk/z_polyhedrator) tool, that allows to create UZP files from MTX representations, and two example tuners which modify the layout of the UZP files to, e.g., achieve optimized results for specific computational kernels. The Z-Polyhedrator tool and the tuners are found in the `/uzp-artifact/z_polyhedrator/` and `/uzp-artifact/uzp-tuners/` folders, respectively. Sections [Customizing the pattern-mining process](#customizing-the-pattern-mining-process) and [Customizing the UZP tuning process](#customizing-the-uzp-tuning-process) describe how to customize the matrix creation and tuning process, respectively.

Finally, we have included all the results obtained and used for writing Sec. 5 in the form of Pandas DataFrames, as well as scripts to regenerate plots in Sec. 5 in the paper. These are described in more detail in Section [Performance spreadsheets](#performance-spreadsheets).

## Folders

- `/uzp-artifact/lib/`: this folder contains libraries and scripts required for the different tools in this artifact, and is not intended to be modified by the user.
- `/uzp-artifact/spmv-executors/`: this folder contains codes and scripts to execute the SpMV kernel on different formats (SpMV and the baselines used in the submitted paper). A more complete description on how to replicate the experimentation in the paper is given below in Section [Running the SpMV executors](#running-the-spmv-executors).
- `/uzp-artifact/z_polyhedrator/`: this is the Rust tool mentioned in Sec. 4.1 of the paper that is used to generate UZP files. Refer to the `README.md`file inside this folder for more details and instructions of usage. The tool is used automatically by the execution scripts in the `/uzp-artifact/spmv-executors/` folder, searching the same list of patterns as used for the experiments in the paper. See Section [Customizing the pattern-mining process](#customizing-the-pattern-mining-process) below for more information.
- `/uzp-artifact/uzp-tuners/`: the UZP format can be tuned for specific targets (compression, execution performance of specific kernels, etc.). This folder contains example tuners. A more complete description on it, and how to change the tuner employed for SpMV kernel execution, is found in Section [Customizing the UZP tuning process](#customizing-the-uzp-tuning-process)
- `/uzp-artifact/performance-spreadsheets/`: contains the spreadsheets including all the results obtained using our [Experimental setup](#experimental-setup), as well as scripts to regenerate plots in Sec. 5 in the paper. These are detailed in Section [Performance spreadsheets](#performance-spreadsheets).

## Running the SpMV executors

The `/uzp-artifact/spmv-executors/` folder contains the different SpMV versions used in Section 5 of the paper, and allows to automatically run experiments on any SuiteSparse matrix. This section explains how to invoke the SpMV kernels with specific SuiteSparse matrices. To re-execute all the experiments included in the paper, check the section [Running all experiments](#running-all-experiments) below. For details on experimental setup and variability considerations, check section [Experimental setup](#experimental-setup).

1. UZP GenEx: the code is contained in folder `uzp-genex`. In order to automatically launch an experiment:

    1. `cd spmv-executors`

    2. `./uzp_spmv.sh <group> <matrix> <datatype> <cache mode> <execution threads>`

        * `<group>` and `<matrix>` are a Group name and Matrix name from SuiteSparse.
        * `<datatype>`: either "_double_" or "_float_" for double and single precision, respectively.
        * `<cache mode>`: either "_hot_" or "_cold_". Under cold cache settings, the SpMV kernel is run only once. Under hot cache settings, the kernel is run 100 times without cleaning the cache between repetitions.
        * `<execution threads>`: either "_1th_", "_2th_", or "_8th_" for 1, 2, or 8 threads, respectively.

    3. See important notes below about [Experimental setup](#experimental-setup), including operational frequency and pinning of threads to cores in the execution machine.

    4. The script automatically downloads the required files from SuiteSparse to the `/tmp` folder of the execution machine, generates UZP files as needed, and launches the executor. By default, this script mines for patterns and tunes the UZP file as described in Sec. 4.3 in the paper. Refer to the subsections below about [customizing UZP construction](#customizing-the-pattern-mining-process) and [tuning](#customizing-the-uzp-tuning-process) for more information.

    5. The script outputs the obtained performance in GFLOPS. It can be modified to output performance counters instead, by modifying the `uzp_spmv.sh` script to compile with `OUTPUT_TYPE=PAPI`. In this case, a `papi_counters.list` must be created inside folder `uzp-genex`,capturing the counters to measure. See [PolyBench C/4.2.1](https://sourceforge.net/projects/polybench/) documentation for more details.

    6. For instance: 

        ```./uzp_spmv.sh QLi crashbasis double cold 1th```

        obtains a performance of approximately 6.8 GFLOPS on the Intel Core 12900K machine used for the tests when fixing the operational frequency at 3.2 GHz.

2. MKL executor: the code is contained in folder `spmv-mkl`. In order to automatically launch the experiment:

    1. `cd spmv-executors`

    2. `./mkl_spmv.sh <group> <matrix> <datatype> <cache mode> <execution threads>`

        * `<group>` and `<matrix>` are a Group name and Matrix name from SuiteSparse.
        * `<datatype>`: either "_double_" or "_float_" for double and single precision, respectively.
        * `<cache mode>`: either "_hot_" or "_cold_". Under cold cache settings, the SpMV kernel is run only once. Under hot cache settings, the kernel is run 100 times without cleaning the cache between repetitions.
        * `<execution threads>`: either "_1th_", "_2th_", or "_8th_" for 1, 2, or 8 threads, respectively.

    3. See important notes below about [Experimental setup](#experimental-setup), including operational frequency and pinning of threads to cores in the execution machine.

    4. The script automatically downloads the required files from SuiteSparse to the `/tmp` folder of the execution machine, and launches the executor. By default, this script uses the "_IEhint_" capabilities in MKL 2024.1. In order to disable them, comment the relevant lines in `spmv-mkl/spmv{d}_mkl.c` (calls to `mkl_sparse_set_mv_hint()` and `mkl_sparse_optimize()`).

    5. The script outputs the obtained performance in GFLOPS. It can be modified to output performance counters instead, by modifying the `spmv-mkl/Makefile` file to compile with `-DPOLYBENCH_PAPI`, instead of `-DPOLYBENCH_GFLOPS`. In this case, a `papi_counters.list` must be created inside folder spmv-mkl` capturing the counters to measure. See [PolyBench C/4.2.1](https://sourceforge.net/projects/polybench/) documentation for more details.

    6. For instance: 

        ```./mkl_spmv.sh QLi crashbasis double cold 1th``` 

        obtains a performance of approximately 4.8 GFLOPS on the Intel Core 12900K machine used for the tests when fixing the operational frequency at 3.2 GHz.

3. MACVETH executor: the helper files are contained in folder `spmv-macveth`. The source codes are downloaded from the [MACVETH artifact repository in GitHub](https://github.com/gabriel-rodriguez/DS-SpMV) by the script. In order to automatically launch the experiment:
    1. `cd spmv-executors`

    2. `./macveth_spmv.sh <group> <matrix> <datatype> <cache mode> <execution threads>`

        * `<group>` and `<matrix>` are a Group name and Matrix name from SuiteSparse.
        * `<datatype>`: only "_float_" is supported by MACVETH, using single-precision floating-point operations.
        * `<cache mode>`: either "_hot_" or "_cold_". Under cold cache settings, the SpMV kernel is run only once. Under hot cache settings, the kernel is run 100 times without cleaning the cache between repetitions.
        * `<execution threads>`: only "_1th_" is supported by the MACVETH artifact, using single-threaded execution.

    3. See important notes below about [Experimental setup](#experimental-setup), including operational frequency and pinning of threads to cores in the execution machine.

    4. The script automatically downloads the required files from SuiteSparse to the /tmp folder of the execution machine, downloads the data-specific source codes from the GitHub repository containing the artifact for MACVETH, compiles the source (this step may take a long time, depending on the input matrix and target machine), and launches the executor. 

    5. The script outputs the obtained performance in GFLOPS. It can be modified to output performance counters instead, by modifying the `spmv-macveth/Makefile` file to compile with `-DPOLYBENCH_PAPI`, instead of `-DPOLYBENCH_GFLOPS`. In this case, a `papi_counters.list` must be created inside folder spmv-macveth` capturing the counters to measure. See [PolyBench C/4.2.1](https://sourceforge.net/projects/polybench/) documentation for more details.

    6. For instance: 

        ```./macveth_spmv.sh QLi crashbasis float cold 1th```

        obtains a performance of approximately 4.0 GFLOPS on the Intel Core 12900K machine used for the tests when fixing the operational frequency at 3.2 GHz. The compilation phase takes approximately 5 minutes.

4. CSR5 executor: the relevant section of the [CSR5 artifact](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5) used for experimentation is contained in folder `spmv-csr5`. In order to automatically launch the experiment:
    1. `cd spmv-executors`

    2. `./csr5_spmv.sh <group> <matrix> <datatype> <cache mode> <execution threads>`

        * `<group>` and `<matrix>` are a Group name and Matrix name from SuiteSparse.
        * `<datatype>`: only "_double_" is supported by CSR5, using double-precision floating-point operations.
        * `<cache mode>`: either "_hot_" or "_cold_". Under cold cache settings, the SpMV kernel is run only once. Under hot cache settings, the kernel is run 100 times without cleaning the cache between repetitions.
        * `<execution threads>`: either "_1th_", "_2th_", or "_8th_" for 1, 2, or 8 threads, respectively.

    3. See important notes below about [Experimental setup](#experimental-setup), including operational frequency and pinning of threads to cores in the execution machine.

    4. The script automatically downloads the required files from SuiteSparse to the `/tmp` folder of the execution machine, and launches the executor. 

    5. The script outputs the obtained performance in GFLOPS. It can be modified to output performance counters instead, by modifying the `spmv-csr5/CSR5_avx2/Makefile` file to compile with `-DPOLYBENCH_PAPI`, instead of `-DPOLYBENCH_GFLOPS`. In this case, a `papi_counters.list` must be created inside folder `spmv-csr5/CSR5_avx2`, capturing the counters to measure. See [PolyBench C/4.2.1](https://sourceforge.net/projects/polybench/) documentation for more details.

    6. For instance: 

        ```./csr5_spmv.sh QLi crashbasis double cold 1th```

        obtains a performance of approximately 2.3 GFLOPS on the Intel Core 12900K machine used for the tests when fixing the operational frequency at 3.2 GHz.

5. Vanilla CSR executor: the code is contained in folder `spmv-csr`. In order to automatically launch the experiment:
    1. `cd spmv-executors`

    2. `./csr_spmv.sh <group> <matrix> <datatype> <cache mode> <execution threads>`

        * `<group>` and `<matrix>` are a Group name and Matrix name from SuiteSparse.
        * `<datatype>`: either "_double_" or "_float_" for double and single precision, respectively.
        * `<cache mode>`: either "_hot_" or "_cold_". Under cold cache settings, the SpMV kernel is run only once. Under hot cache settings, the kernel is run 100 times without cleaning the cache between repetitions.
        * `<execution threads>`: either "_1th_", "_2th_", or "_8th_" for 1, 2, or 8 threads, respectively.

    3. See important notes below about [Experimental setup](#experimental-setup), including operational frequency and pinning of threads to cores in the execution machine.

    4. The script automatically downloads the required files from SuiteSparse to the `/tmp` folder of the execution machine, and launches the executor.

    5. The script outputs the obtained performance in GFLOPS. It can be modified to output performance counters instead, by modifying the `spmv-csr/Makefile` file to compile with `-DPOLYBENCH_PAPI`, instead of `-DPOLYBENCH_GFLOPS`. In this case, a `papi_counters.list` must be created inside folder `spmv-csr`, capturing the counters to measure. See [PolyBench C/4.2.1](https://sourceforge.net/projects/polybench/) documentation for more details.

    6. For instance: 

        ```./csr_spmv.sh QLi crashbasis double cold 1th```

         obtains a performance of approximately 2.6 GFLOPS on the Intel Core 12900K machine used for the tests when fixing the operational frequency at 3.2 GHz.



### Running all experiments

The paper experiments on 229 matrices extracted from [SuiteSparse](https://sparse.tamu.edu). This artifact includes scripts to re-run the experiments in the paper as described below. The artifact also includes all the data used in the submitted version of the paper as a set of Pandas DataFrames. See also the section on the included [performance spreadsheets](#performance-spreadsheets) for information on how to obtain the original experimental data used in Sec. 5 of the paper.

The script `spmv-executors/run_all.py` allows to run all experiments of a particular type, e.g., to run all double cold single-threaded experiments for all relevant executor versions type:

```
$ cd spmv-executors
$ ./run_all.py double cold 1th
```

The script generates the output in a file called `spmv-executors/results_run_all_{datatype}_{cachemode}_{execution_threads}_{timestamp}.out`. Besides, for simplicity, the script also parses the output file and generates a CSV spreadsheet under the same file name but with the _.csv_ extension.

### Experimental setup

The experimentation that led to the results in Sec. 5 of the submitted paper was conducted on an Intel Core i9 12900K with 128 GiB of RAM. Using different processors may lead to different performance profiles, depending on their capabilities. Besides, there are some aspects of the setup which can lead to experimental variability, as described below:

**Operational frequency**: the use of DVFS may lead to experimental variability, particularly in bursts of short runs, that may affect the relative performance of different kernels depending on CPU temperature. In order to minimize this effect, during our experiments the operational frequency was fixed using the following command:

```cpupower frequency-set -d 3.2GHz -u 3.2GHz```

We use here the nominal base frequency of 3.2GHz for the Intel i9 12900K, which should correspond to a sustainable steady-state frequency that 	should not vary significantly due to thermal constraints.

_Reviewers are advised to fix the frequency of the CPU running the Singularity container, if possible, to minimize variability._

**Pinning of threads to cores**: the Intel i9 12900K CPU includes both P- and E-cores. In our tests, we have pinned threads to cores using the following command:

```
numactl -C10 <command> # For single-threaded executions
numactl -C10,12 <command> # For 2-threaded executions
numactl -C10,12,14,1,2,4,6 # For 8-threaded executions
```

These are logical P-cores in the Intel i9 12900K that are associated to different physical cores. The execution scripts in folder `/uzp-artifact/spmv-	executor/` employ this pinning strategy (hardcoded into each file). 

_Reviewers are advised to change this pinning strategy if it does not fit the processor used for tests, e.g., if there is no core number 14, or if some of the selected cores are E-cores or are logical cores associated to the same physical core._


# Customizing the pattern-mining process

As detailed in Sec. 4.1 in the submitted paper, the user must provide a set of polyhedral patterns that the Z-Polyhedrator tool will mine for. By default, the tool uses the patterns enumerated in file `/uzp-artifact/z_polyhedrator/data/patterns.txt`. In that file, each pattern is specified as a 3-tuple:

```(N,stride_y,stride_y)```

Where N is the number of points and the strides in _y_ and _x_ represent, respectively, the access stride in the rows and columns of a 2-dimensional matrix. The Z-Polyhedrator tool mines for these enumeration of patterns in the order that they are presented, e.g., it will try to find all possible instances of the first pattern before trying to find the second one. This behavior can be changed, as described in the tool manual.

If the reviewer wishes to experiment with different a different set of patterns, the steps to follow are:

1. Write the new patterns file, e.g., in `/uzp-artifact/z_polyhedrator/data/new_patterns.txt`.
2. Modify the script that runs Z-Polyhedrator on top of SuiteSparse matrices when running the SpMV kernels in `uzp-artifact/lib/scripts/z_polyhedrator_manager.py`. The relevant line is the last one, which calls the `manager.generate_uzp()` function specifying the patterns file as its first parameter.
3. Run the SpMV experiments as described in Section [Running the SpMV executors](#running-the-spmv-executors).

# Customizing the UZP tuning process

By default, the SpMV kernel executors are employing the tuner described in Sec. 4.3 of the submitted paper. The `/uzp-artifact/uzp-tuners/` folder contains an additional example tuner, that will tile a matrix using a given tile size for locality during the execution of the SpMV kernel. Note that this tuner, which typically obtains better performance than the standard one for large matrices, has not been used in the submitted paper, so it is experimental code and bugs might be present.

In order to use this tuner to generate a file, follow these steps:

```
$ cd /uzp-artifact/spmv-executors
$ ./uzp_spmv.sh Mittelmann rail4284 double cold 1th # Run once to download the required files from SuiteSparse and generate the basic UZP
$ cd /uzp-artifact/uzp-tuners
$ make spf_tiling_tuner
$ ./spf_tiling_tuner --tile 2048 /tmp/rail4284/rail4284.1d.uzp /tmp/rail4284/rail4284.tuned.uzp # Tune the UZP using 2048x2048 tiles
$ cd /uzp-artifact/spmv-executors
$ ./uzp_spmv.sh Mittelmann rail4284 double cold 1th # Repeat the execution, now the tiled UZP will be used automatically
```

In order to use this tuner by default for experiments, line 75 in file `/uzp-artifact/spmv-executors/uzp_spmv.sh` must be changed to invoke the tiling tuner.

# Performance spreadsheets

For reference and to simplify reviewing, we have included two Pandas DataFrames containing the results obtained during our experimentation in folder _/uzp-artifact/performance-spreadsheets/_. This folder contains two files:

* `pldi25_uzp_results_singlethread.pickle`: this file contains a Pandas DataFrame including 34 different columns with matrix statistics and performance counters, namely:
    * index: (app,version,variant,dtype,cachestate): matrix, kernel version, variant (e.g., tuner used for UZP, whether IE hint was used for MKL, etc.) data type and cache mode for the experiment.
    * _nnz_: number of nonzeros in the matrix.
    * _inc_: number of nonzeros incorporated inside z-polyhedral shapes in the UZP file. Mechanically, _nnz - inc_ computes the number of points *not* incorporated, but stored using a CSR/COO section.
    * _nrows_: number of rows of the sparse matrix.
    * _ncols_: number of columns of the sparse matrix.
    * _cycles_: execution cycles of the corresponding SpMV kernel (_UNHALTED\_REFERENCE\_CYCLES_)
    * _stalls_: number of cycles with no uops executed by thread (_UOPS\_EXECUTED:STALL\_CYCLES_)
    * _d1h_: Retired load uops with L1 cache hits as data source (_MEM\_LOAD\_UOPS\_RETIRED:L1\_HIT_)
    * _d1m_: Retired load uops which missed the L1D (_MEM\_LOAD\_UOPS\_RETIRED:L1\_MISS_)
    * _i1h_: Number of instruction fetch tag lookups that hit in the instruction cache (L1I). Counts at 64-byte cache-line granularity (_ICACHE\_64B:IFTAG\_HIT_)
    * _i1m_: Number of instruction fetch tag lookups that miss in the instruction cache (L1I). Counts at 64-byte cache-line granularity (_ICACHE\_64B:IFTAG\_MISS_)
    * _l2m_: All requests that miss the L2 cache (_L2\_RQSTS:MISS_)
    * _l2im_: L2 cache misses when fetching instructions (_L2\_RQSTS:CODE\_RD\_MISS_)
    * _l3m_: Core-originated cacheable demand requests missed LLC - architectural event (_LONGEST\_LAT\_CACHE:MISS_)
    * _brinst_: Branch instructions (_PAPI\_BR\_INS_)
    * _mispred_: Conditional branch instructions mispredicted (_PAPI\_BR\_MSP_)
    * _scalars_: Number of scalar single precision floating-point arithmetic instructions (multiply by 1 to get flops) (_FP\_ARITH:SCALAR\_SINGLE_)
    * _packed\_128s_: Number of scalar 128-bit packed single precision floating-point arithmetic instructions (multiply by 4 to get flops) (_FP\_ARITH:128B\_PACKED\_SINGLE_)
    * _packed\_256s_: Number of scalar 256-bit packed single precision floating-point arithmetic instructions (multiply by 8 to get flops) (_FP\_ARITH:256B\_PACKED\_SINGLE_)
    * _scalard_: Number of scalar double precision floating-point arithmetic instructions (multiply by 1 to get flops) (_FP\_ARITH:SCALAR\_DOUBLE_)
    * _packed\_128d_: Number of scalar 128-bit packed double precision floating-point arithmetic instructions (multiply by 2 to get flops) (_FP\_ARITH:256B\_PACKED\_DOUBLE_)
    * _packed\_256d_: Number of scalar 256-bit packed double precision floating-point arithmetic instructions (multiply by 4 to get flops) (_FP\_ARITH:256B\_PACKED\_DOUBLE_)
    * _insts_: Instructions completed (_PAPI\_TOT\_INS_)
    * _uops_: Number of uops executed per thread in each cycle (_UOPS\_EXECUTED:THREAD_)
    * _loads_: All load uops retired (_MEM\_INST\_RETIRED:ALL\_LOADS_)
    * _stores_: All store uops retired (_MEM\_INST\_RETIRED:ALL\_STORES_)
    * _dtlb\_lwalks_: Misses in all DTLB levels that cause page walks (_DTLB\_LOAD\_MISSES:MISS\_CAUSES\_A\_WALK_)
    * _dtlb\_swalks_: Misses in all DTLB levels that cause page walks (_DTLB\_STORE\_MISSES:MISS\_CAUSES\_A\_WALK_)
    * _itlb\_walks_: Misses in all DTLB levels that cause page walks (_ITLB\_MISSES:MISS\_CAUSES\_A\_WALK_)
    * _itlb\_wd_: Cycles when PMH is busy with page walks (_ITLB\_MISSES:WALK\_DURATION_)
    * _l2\_pfm_: Requests from the L1/L2/L3 hardware prefetchers or Load software prefetches that miss L2 cache (_L2\_RQSTS:PF\_MISS_)
    * _l2\_pfh_: Requests from the L1/L2/L3 hardware prefetchers or Load software prefetches that hit L2 cache (_L2\_RQSTS:PF\_HIT_)
    * _mite\_uops_: Number of uops delivered to Instruction Decode Queue (IDQ) from MITE path (_IDQ:MITE\_UOPS_)
    * _dsb\_uops_: Number of uops delivered to Instruction Decode Queue (IDQ) from Decode Stream Buffer (DSB) path (_IDQ:DSB\_UOPS_)
    * _gflops_: GFLOPS. This is a column computed as 2xNNZx3.2/CYCLES, i.e., 2xNNZ = theoretical number of flops for this matrix, divided by the execution cycles which gives FLOPS/cycle and multiplied by 3.2 which is the operational frequency in GHz.
* `pldi25_uzp_results_multithread.pickle`: similar to the DataFrame for single-threaded results, but contains an additional column in the index showing the number of threads used for execution. Note that performance counters were not captured for these executions, and the GFLOPS were computed using the PolyBench harness directly. All other columns described above are shown as NaNs.

Also note that some performance counters do not work in the Intel Core i9 12900K, for example, no data was registered regarding page walks.

## Regenerating plots

The _/uzp-artifact/performance-spreadsheets/_ folder also contains several scripts to regenerate the plots in Sec. 5 that depend on runtime results:

* `plot_fig6.py [ mkl | uzp | csr | csr5 ] <input.csv>`: receives a kernel version (mkl, uzp, csr, or csr5) and regenerates Fig. 6 in the paper, i.e., a scatterplot showing the double-precision cold-cache GFLOPS for the selected kernel version. By default, it generates the plot from the original results in the paper. If the optional parameter _<input.csv>_ is added, it employs that CSV as input. It must have been generated by the _run\_all.py_ command.
* `plot_fig7.py [ mkl | uzp | csr | macveth ] <input.csv>`: similar to the previous one, but generates the scatterplot for the single-precision cold-cache GFLOPS in Fig. 7 in the paper.
* `plot_fig10.py [ mkl | uzp | csr | csr5 | macveth ] [ mkl | uzp | csr | csr5 | macveth ] [ double | float ] [ hot | cold ] <input.csv>`: receives two kernel versions, a precision and a cache-mode configuration, and regenerates one of the subplots in Fig. 10 in the paper, i.e., a jointplot showing the GFLOPS of the first selected kernel version vs. the second. By default, it generates the plot from the original results in the paper. If the optional parameter _<input.csv>_ is added, it employs that CSV as input. It must have been generated by the _run\_all.py_ command.
* `plot_fig11.py [ mkl | uzp | csr5 ] [ double | float ] [ hot | cold ] <input1.csv> <input2.csv> <input3.csv>`: receives a kernel version, a precision and a cache-mode configuration, and regenerates one of the subplots in Fig. 11 in the paper, i.e., a scatterplot showing the performance scaling when employing parallel kernel versions. By default, it generates the plot from the original results in the paper. If the optional parameters _<input1.csv>_, _<input2.csv>_, and _<input3.csv>_ are passed, it employs those three CSVs as input. They must have been generates by the _run\_all.py_ command, and correspond to executions with the same selected kernel and 1, 2, and 8 threads.

