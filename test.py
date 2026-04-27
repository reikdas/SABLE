import glob
import os
import pathlib
import subprocess

import numpy
import pytest
import scipy
import scipy.io
import scipy.sparse

from src.codegen import (
    gen_single_threaded_spmm_naive_naive,
    gen_single_threaded_spmm_naive_spreg,
    gen_single_threaded_spmm_mixed_naive,
    gen_single_threaded_spmm_mkl_naive,
    gen_single_threaded_spmm_mkl_spreg,
    vbr_spmm_codegen,
    gen_single_threaded_spmv_naive_naive,
    gen_single_threaded_spmv_naive_uzp,
    gen_single_threaded_spmv_mixed_naive,
    gen_single_threaded_spmv_mkl_naive,
    gen_single_threaded_spmv_mkl_uzp,
    vbr_spmv_codegen,
)
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from src.consts import MKL_AVAILABLE as MKL_AVAILABLE
from src.consts import SPV8_FLAGS as SPV8_FLAGS

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)
from utils.convert_real_to_vbr import (convert_sparse_to_vbr,
                                       convert_sparse_to_vbrc,
                                       convert_sparse_to_vbrc_with_blocks,
                                       vbrc_matrix_gen,
                                       _write_vbrc_file)
from utils.fileio import read_vbr, read_vbrc, write_dense_matrix, write_dense_vector
from utils.utils import extract_mul_nums

def cmp_file(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        # Filter out lines that start with "Dense" and empty lines from both files
        lines1 = [line.strip() for line in f1 if not line.strip().startswith("Dense") and line.strip()]
        lines2 = [line.strip() for line in f2 if not line.strip().startswith("Dense") and line.strip()]
        
        # Compare the filtered lines
        for line1, line2 in zip(lines1, lines2):
            # Attempt to compare as floats if the lines contain numeric data
            try:
                # This will succeed if both lines are numeric
                if float(line1) != float(line2):
                    return False
            except ValueError:
                # If they aren't numeric, compare them as strings
                if line1 != line2:
                    return False
        
        # Check if both files have the same number of non-Dense lines
        if len(lines1) != len(lines2):
            return False
            
    return True

def test_read_vbr():
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(BASE_PATH, "tests", "example.vbr"))
    assert(numpy.array_equal(val,[4.0,1.0,2.0,5.0,1.0,2.0,-1.0,0.0,1.0,-1.0,6.0,2.0,-1.0,1.0,7.0,2.0,2.0,1.0,9.0,2.0,0.0,3.0,2.0,1.0,3.0,4.0,5.0,10.0,4.0,3.0,2.0,4.0,3.0,0.0,13.0,3.0,2.0,4.0,11.0,0.0,2.0,3.0,7.0,8.0,-2.0,4.0,3.0,25.0,8.0,3.0,12.0]))
    assert(numpy.array_equal(indx,[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(rpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(cpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))


def test_compression():
    dense = numpy.array([[ 4.,  2.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  1.],
                        [ 1.,  5.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., -1.],
                        [ 0.,  0.,  6.,  1.,  2.,  2.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  2.,  7.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0., -1.,  2.,  9.,  3.,  0.,  0.,  0.,  0.,  0.],
                        [ 2.,  1.,  3.,  4.,  5., 10.,  4.,  3.,  2.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  4., 13.,  4.,  2.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  3.,  3., 11.,  3.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  7.,  0.,  0.],
                        [ 8.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  3.],
                        [-2.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 12.]])
    sparse = scipy.sparse.csc_matrix(dense)
    rpntr = [0, 2, 5, 6, 9, 11]
    cpntr = [0, 2, 5, 6, 9, 11]
    val2, indx2, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc(sparse, rpntr, cpntr, "example2", "tests", op="spmv", density=80)
    assert(numpy.array_equal(val2,[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0]))
    assert(numpy.array_equal(indx2,[0, 4, 6, 15, 17, 20, 21, 24, 33, 37]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))
    assert(numpy.array_equal(ublocks,[2, 4, 9, 12]))
    assert(numpy.array_equal(indptr, [0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8]))
    assert(numpy.array_equal(indices,[10, 10, 5, 5, 5, 5, 10, 10]))
    assert(numpy.array_equal(csr_val,[1.0, -1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 12.0]))

def test_read_compression():
    test_compression()
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(BASE_PATH, "tests", "example2.vbrc"))
    assert(numpy.array_equal(val,[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0]))
    assert(numpy.array_equal(indx,[0, 4, 6, 15, 17, 20, 21, 24, 33, 37]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(rpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(cpntr,[0, 2, 5, 6, 9, 11]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))
    assert(numpy.array_equal(ublocks,[2, 4, 9, 12]))
    assert(numpy.array_equal(indptr, [0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8]))
    assert(numpy.array_equal(indices,[10, 10, 5, 5, 5, 5, 10, 10]))
    assert(numpy.array_equal(csr_val,[1.0, -1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 12.0]))

def test_partition():
    dense = numpy.array([[ 4.,  2.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  1.],
                                [ 1.,  5.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0., -1.],
                                [ 0.,  0.,  6.,  1.,  2.,  2.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  2.,  7.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0., -1.,  2.,  9.,  3.,  0.,  0.,  0.,  0.,  0.],
                                [ 2.,  1.,  3.,  4.,  5., 10.,  4.,  3.,  2.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  4., 13.,  4.,  2.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  3.,  3., 11.,  3.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  7.,  0.,  0.],
                                [ 8.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  3.],
                                [-2.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 12.]])
    sparse = scipy.sparse.csc_matrix(dense)
    rpntr = [0, 2, 5, 6, 9, 11]
    cpntr = [0, 2, 5, 6, 9, 11]
    val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc(sparse, rpntr, cpntr, "example2", "tests", op="spmv", density=0)
    assert(numpy.array_equal(val,[4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 0, 0.0, 1.0, -1.0, 6.0, 2.0, -1.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0, 2.0, 0.0, 3.0, 2.0, 1.0, 3.0, 4.0, 5.0, 10.0, 4.0, 3.0, 2.0, 4.0, 3.0, 0.0, 13.0, 3.0, 2.0, 4.0, 11.0, 0.0, 2.0, 3.0, 7.0, 8.0, -2.0, 4.0, 3.0, 0, 0, 3.0, 12.0]))
    assert(numpy.array_equal(indx,[0, 4, 6, 10, 19, 22, 24, 27, 28, 31, 34, 43, 47, 51]))
    assert(numpy.array_equal(bindx,[0, 2, 4, 1, 2, 0, 1, 2, 3, 2, 3, 0, 4]))
    assert(numpy.array_equal(bpntrb,[0, 3, 5, 9, 11]))
    assert(numpy.array_equal(bpntre,[3, 5, 9, 11, 13]))
    # Since density=0, all blocks should be dense, so ublocks, indptr, indices, csr_val should be empty
    assert(len(ublocks) == 0)
    assert(len(indptr) == 0)
    assert(len(indices) == 0)
    assert(len(csr_val) == 0)

def run_spmv(threads, mkl):
    write_dense_vector(1.0, 11)  # ensure dense vector exists (gitignored Generated_dense_tensors/)
    vbr_spmv_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests", mkl=mkl)
    if mkl:
        extra_flags = MKL_FLAGS
    else:
        spv8_obj = os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")
        if not os.path.isfile(spv8_obj):
            pytest.skip("SPV8 object not built (spv8-public/bin/spmv_spv8.o); skipping non-MKL spmv test")
        extra_flags = SPV8_FLAGS
    subprocess.check_call(["gcc", "-o", "example", "example.c"] + CFLAGS + extra_flags, cwd="tests")
    output = subprocess.check_output(["./example"], cwd="tests").decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon.txt"))

def run_spmv_unroll(threads, mkl):
    test_compression()
    vbr_spmv_codegen("example2", "tests", "tests", threads, mkl=mkl)
    if mkl:
        extra_flags = MKL_FLAGS
    else:
        spv8_obj = os.path.join(BASE_PATH, "spv8-public", "bin", "spmv_spv8.o")
        if not os.path.isfile(spv8_obj):
            pytest.skip("SPV8 object not built (spv8-public/bin/spmv_spv8.o); skipping non-MKL unroll test")
        extra_flags = SPV8_FLAGS
    subprocess.check_call(["gcc", "-o", "example2", "example2.c"] + CFLAGS + extra_flags, cwd="tests")
    output = subprocess.check_output(["./example2"], cwd="tests").decode("utf-8").split("\n")
    if "warning" in output[0].lower():
        output = output[2:]
    else:
        output = output[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    assert(cmp_file("tests/output.txt", "tests/output_spmv_canon_sparse.txt"))

def test_spmv():
    run_spmv(1, mkl=False)
#     run_spmv(2)
#     run_spmv(4)
#     run_spmv(8)
#     run_spmv(16)

def test_spmv_mkl():
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    run_spmv(1, mkl=True)

# def test_spmv_py():
#     run_spmv_py()

def test_spmv_unroll():
    run_spmv_unroll(1, mkl=False)
#     run_spmv_unroll(2)
#     run_spmv_unroll(4)
#     run_spmv_unroll(8)
#     run_spmv_unroll(16)

def test_spmv_unroll_mkl():
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    run_spmv_unroll(1, mkl=True)

def run_spmm(threads):
    """Test SpMM codegen."""
    write_dense_matrix(1.0, 11, 512)
    vbr_spmm_codegen(filename="example", dir_name="tests", threads=threads, vbr_dir="tests", bench=5)
    subprocess.check_call(["gcc", "-o", "example_spmm", "example.c"] + CFLAGS, cwd="tests")
    output = subprocess.check_output(["./example_spmm"], cwd="tests").decode("utf-8").split("\n")
    # Skip timing lines and check output
    output_lines = [line for line in output if line and not line.startswith("Sparse:") and not line.startswith("Dense:") and not line.startswith("Dense Block")]
    with open(os.path.join("tests", "output_spmm.txt"), "w") as f:
        f.write("\n".join(output_lines))
    # For now, just check that it runs without error
    assert(len(output_lines) > 0)

def test_spmm():
    run_spmm(1)

def run_spmm_unroll(threads):
    """Test SpMM codegen with sparse blocks."""
    test_compression()
    write_dense_matrix(1.0, 11, 512)
    vbr_spmm_codegen(filename="example2", dir_name="tests", threads=threads, vbr_dir="tests", bench=5)
    subprocess.check_call(["gcc", "-o", "example2_spmm", "example2.c"] + CFLAGS, cwd="tests")
    output = subprocess.check_output(["./example2_spmm"], cwd="tests").decode("utf-8").split("\n")
    # Skip timing lines
    output_lines = [line for line in output if line and not line.startswith("Sparse:") and not line.startswith("Dense:") and not line.startswith("Dense Block")]
    with open(os.path.join("tests", "output_spmm_unroll.txt"), "w") as f:
        f.write("\n".join(output_lines))
    # For now, just check that it runs without error
    assert(len(output_lines) > 0)

def test_spmm_unroll():
    run_spmm_unroll(1)

def _run_spmm_dense_backend_test(gen_func, filename: str, link_mkl: bool):
    """Generate a split SpMM fixture, run it, and compare against scipy."""
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape

    dense_blocks = [(0, 3, 0, 3)]
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, dense_blocks)
    assert len(val) > 0, "Expected split matrix with a dense block"

    vbr_dir = os.path.join("tests")
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    write_dense_matrix(1.0, cols, 512)

    dir_name = os.path.join("tests")
    gen_func(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1
    )

    compile_flags = list(CFLAGS) + (list(MKL_FLAGS) if link_mkl else [])
    subprocess.check_call(
        ["gcc", "-o", filename, f"{filename}.c"] + compile_flags,
        cwd="tests"
    )

    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")

    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            float(line)
            result_lines.append(line)
        except ValueError:
            pass

    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    y_expected = A.dot(numpy.ones((cols, 512)))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmm_naive_naive():
    """Test SpMM with handwritten dense dispatch + naive sparse kernel."""
    _run_spmm_dense_backend_test(
        gen_single_threaded_spmm_naive_naive,
        filename="example3_spmm_naive_naive",
        link_mkl=False,
    )


def test_spmm_mixed_naive():
    """Test SpMM with mixed dense dispatch + naive sparse kernel."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    _run_spmm_dense_backend_test(
        gen_single_threaded_spmm_mixed_naive,
        filename="example3_spmm_mixed_naive",
        link_mkl=True,
    )


def test_spmm_mkl_naive():
    """Test SpMM with always-MKL dense dispatch + naive sparse kernel."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    _run_spmm_dense_backend_test(
        gen_single_threaded_spmm_mkl_naive,
        filename="example3_spmm_mkl_naive",
        link_mkl=True,
    )


def _check_avx512_support():
    """Check if AVX512 is supported on this CPU."""
    result = subprocess.run(
        ["grep", "-c", "avx512", "/proc/cpuinfo"],
        capture_output=True, text=True
    )
    return result.returncode == 0 and int(result.stdout.strip()) > 0


def _get_spreg_source_files():
    """Get all source files needed for sparse-register-tiling compilation."""
    spreg_base = os.path.join(BASE_PATH, "sparse-register-tiling", "spmm_nano_kernels")
    
    # Core source files from src/
    core_src_files = [
        os.path.join(spreg_base, "src", "cake_block_dims.cpp"),
        os.path.join(spreg_base, "src", "ExecutorFactory.cpp"),
        os.path.join(spreg_base, "src", "kernel_mapping.cpp"),
        os.path.join(spreg_base, "src", "mapping_io.cpp"),
        os.path.join(spreg_base, "src", "packing.cpp"),
        os.path.join(spreg_base, "src", "utils", "misc.cpp"),
        os.path.join(spreg_base, "src", "spmm_spreg_wrapper.cpp"),
    ]
    
    # Generated source files (top-level)
    generated_dir = os.path.join(spreg_base, "generated")
    generated_files = glob.glob(os.path.join(generated_dir, "*.cpp"))
    
    # Generated AVX512 factory files
    avx512_factory_files = glob.glob(os.path.join(generated_dir, "AVX512", "factories", "*", "*.cpp"))
    
    return core_src_files + generated_files + avx512_factory_files


def _get_spreg_include_dirs():
    """Get all include directories needed for sparse-register-tiling."""
    spreg_base = os.path.join(BASE_PATH, "sparse-register-tiling", "spmm_nano_kernels")
    return [
        os.path.join(spreg_base, "include"),
        os.path.join(spreg_base, "third_party", "version2", "vectorclass"),
        os.path.join(spreg_base, "third_party", "rte"),
        os.path.join(spreg_base, "generated"),
        os.path.join(spreg_base, "generated", "AVX512", "include"),
    ]


@pytest.mark.skipif(
    not _check_avx512_support(),
    reason="AVX512 not supported on this CPU"
)
@pytest.mark.skip(reason="Compilation too slow for regular testing - use benchmark script instead")
def test_spmm_naive_spreg():
    """Test gen_single_threaded_spmm_naive_spreg against scipy SpMM."""
    # Load matrix from MTX file
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)
    
    # Convert to CSC format
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape
    
    # Convert to fully sparse VBRC format (no dense blocks)
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, [])
    
    # Verify it's fully sparse (no dense blocks)
    assert len(val) == 0, "Expected fully sparse matrix (val should be empty)"
    
    # Write VBRC file (required by generated code)
    vbr_dir = os.path.join("tests")
    filename = "example3_spmm_spreg"
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    
    # Write dense matrix file (required by generated code) - 512 columns for SpMM
    write_dense_matrix(1.0, cols, 512)
    
    # Generate code using naive_spreg kernel (generate directly in tests/ directory)
    dir_name = os.path.join("tests")
    gen_single_threaded_spmm_naive_spreg(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1
    )
    
    # Get source files and include directories (same as benchmark script)
    spreg_src_files = _get_spreg_source_files()
    include_dirs = _get_spreg_include_dirs()
    
    # Build include flags
    include_flags = []
    for inc_dir in include_dirs:
        include_flags.extend(["-I", inc_dir])
    
    # Compile with g++ to handle C++ code
    # Use -x c++ to treat the .c file as C++ for linking purposes
    # CRITICAL: -DENABLE_AVX512 is required for factory registration
    compile_cmd = [
        "g++", "-x", "c++", "-o", filename, f"{filename}.c"
    ] + spreg_src_files + [
        "-std=c++17", "-O3", "-march=native", "-fopenmp", "-lstdc++fs", "-lpthread",
        "-DRTE_CACHE_LINE_SIZE=64",
        "-DENABLE_AVX512",  # Critical for factory registration
        "-mavx512f",        # Enable AVX512 instructions
    ] + include_flags
    
    subprocess.check_call(compile_cmd, cwd="tests")
    
    # Run the executable and capture output
    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")
    
    # Skip timing lines and extract result matrix
    # The output format is: timing lines, then blank line, then y values one per line
    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False  # Blank line marks end of timing, start of results
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            # Try to parse as float - if successful, it's a result value
            float(line)
            result_lines.append(line)
        except ValueError:
            pass
    
    # Convert result to numpy array and reshape to (rows, 512)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)
    
    # Compute expected result using scipy
    # X is a matrix of ones with shape (cols, 512)
    X = numpy.ones((cols, 512))
    Y_expected = A.dot(X)
    
    # Compare results (allow small numerical differences)
    numpy.testing.assert_allclose(y_generated, Y_expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not _check_avx512_support(),
    reason="AVX512 not supported on this CPU"
)
@pytest.mark.skip(reason="Compilation too slow for regular testing - use benchmark script instead")
def test_spmm_mkl_spreg():
    """Test gen_single_threaded_spmm_mkl_spreg against scipy SpMM.

    Tests MKL cblas_dgemm for dense blocks + sparse-register-tiling for sparse part.
    """
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available")
    # Load matrix from MTX file
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)

    # Convert to CSC format
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape

    # Convert to fully sparse VBRC format (no dense blocks)
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, [])

    # Verify it's fully sparse (no dense blocks)
    assert len(val) == 0, "Expected fully sparse matrix (val should be empty)"

    # Write VBRC file (required by generated code)
    vbr_dir = os.path.join("tests")
    filename = "example3_spmm_mkl_spreg"
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)

    # Write dense matrix file (required by generated code) - 512 columns for SpMM
    write_dense_matrix(1.0, cols, 512)

    # Generate code using mkl_spreg kernel (generate directly in tests/ directory)
    dir_name = os.path.join("tests")
    gen_single_threaded_spmm_mkl_spreg(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1
    )

    # Get source files and include directories (same as benchmark script)
    spreg_src_files = _get_spreg_source_files()
    include_dirs = _get_spreg_include_dirs()

    # Build include flags
    include_flags = []
    for inc_dir in include_dirs:
        include_flags.extend(["-I", inc_dir])

    # Compile with g++ to handle C++ code
    # Use -x c++ to treat the .c file as C++ for linking purposes
    # CRITICAL: -DENABLE_AVX512 is required for factory registration
    compile_cmd = [
        "g++", "-x", "c++", "-o", filename, f"{filename}.c"
    ] + spreg_src_files + [
        "-std=c++17", "-O3", "-march=native", "-fopenmp", "-lstdc++fs", "-lpthread",
        "-DRTE_CACHE_LINE_SIZE=64",
        "-DENABLE_AVX512",  # Critical for factory registration
        "-mavx512f",        # Enable AVX512 instructions
    ] + include_flags + MKL_FLAGS

    subprocess.check_call(compile_cmd, cwd="tests")

    # Run the executable and capture output
    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")

    # Skip timing lines and extract result matrix
    # The output format is: timing lines, then blank line, then y values one per line
    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False  # Blank line marks end of timing, start of results
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            # Try to parse as float - if successful, it's a result value
            float(line)
            result_lines.append(line)
        except ValueError:
            pass

    # Convert result to numpy array and reshape to (rows, 512)
    y_generated = numpy.array([float(x) for x in result_lines]).reshape(rows, 512)

    # Compute expected result using scipy
    # X is a matrix of ones with shape (cols, 512)
    X = numpy.ones((cols, 512))
    Y_expected = A.dot(X)

    # Compare results (allow small numerical differences)
    numpy.testing.assert_allclose(y_generated, Y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_naive():
    """Test gen_single_threaded_spmv_naive_naive against scipy SpMV."""
    # Load matrix from MTX file
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)
    
    # Convert to CSC format
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape
    
    # Convert to fully sparse VBRC format (no dense blocks)
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, [])
    
    # Verify it's fully sparse (no dense blocks)
    assert len(val) == 0, "Expected fully sparse matrix (val should be empty)"
    
    # Write VBRC file (required by generated code)
    vbr_dir = os.path.join("tests")
    filename = "example3"
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    
    # Write dense vector file (required by generated code)
    write_dense_vector(1.0, cols)
    
    # Generate code using naive kernel (generate directly in tests/ directory)
    dir_name = os.path.join("tests")
    gen_single_threaded_spmv_naive_naive(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1
    )
    
    # Compile the generated code
    subprocess.check_call(
        ["gcc", "-o", filename, f"{filename}.c"] + CFLAGS,
        cwd="tests"
    )
    
    # Run the executable and capture output
    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")
    
    # Skip timing lines and extract result vector
    # The output format is: timing lines, then blank line, then y values one per line
    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False  # Blank line marks end of timing, start of results
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            # Try to parse as float - if successful, it's a result value
            float(line)
            result_lines.append(line)
        except ValueError:
            pass
    
    # Convert result to numpy array
    y_generated = numpy.array([float(x) for x in result_lines])
    
    # Compute expected result using scipy
    x = numpy.ones(cols)  # Use all-ones vector (same as generated code)
    y_expected = A.dot(x)
    
    # Compare results (allow small numerical differences)
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def _run_spmv_dense_backend_test(gen_func, filename: str, link_mkl: bool):
    """Shared body for SpMV dense-backend correctness tests.

    Generates code via *gen_func* (a SpMV codegen function with naive sparse
    dispatch), compiles it (linking MKL if *link_mkl*), runs the binary,
    and compares the output against scipy's SpMV result.
    """
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape

    dense_blocks = [(0, 3, 0, 3)]
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, dense_blocks)
    assert len(val) > 0, "Expected split matrix with a dense block"

    vbr_dir = os.path.join("tests")
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    write_dense_vector(1.0, cols)

    dir_name = os.path.join("tests")
    gen_func(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1,
    )

    compile_flags = list(CFLAGS) + (list(MKL_FLAGS) if link_mkl else [])
    subprocess.check_call(
        ["gcc", "-o", filename, f"{filename}.c"] + compile_flags,
        cwd="tests",
    )
    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")

    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            float(line)
            result_lines.append(line)
        except ValueError:
            pass

    y_generated = numpy.array([float(x) for x in result_lines])
    y_expected = A.dot(numpy.ones(cols))
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-10, atol=1e-10)


def test_spmv_naive_naive():
    """Test SpMV with handwritten dense dispatch + naive sparse kernel."""
    _run_spmv_dense_backend_test(
        gen_single_threaded_spmv_naive_naive,
        filename="example3_spmv_naive_naive",
        link_mkl=False,
    )


def test_spmv_mixed_naive():
    """Test SpMV with mixed dense dispatch (cblas_dgemv + handwritten kernels) + naive sparse kernel."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available; skipping mixed dense dispatch test")
    _run_spmv_dense_backend_test(
        gen_single_threaded_spmv_mixed_naive,
        filename="example3_spmv_mixed_naive",
        link_mkl=True,
    )


def test_spmv_mkl_naive():
    """Test SpMV with always-MKL dense dispatch (cblas_dgemv for every block) + naive sparse kernel."""
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available; skipping always-MKL dense dispatch test")
    _run_spmv_dense_backend_test(
        gen_single_threaded_spmv_mkl_naive,
        filename="example3_spmv_mkl_naive",
        link_mkl=True,
    )


def _uzp_available() -> bool:
    """Check whether the UZP toolchain is functional.

    Verifies that uzp_prepare.sh exists and is executable, and that the
    z_polyhedrator binary has been built (Rust + cargo required).
    """
    uzp_prepare = os.path.join(BASE_PATH, "uzp_prepare.sh")
    if not os.path.isfile(uzp_prepare) or not os.access(uzp_prepare, os.X_OK):
        return False
    # z_polyhedrator must be pre-built (requires Rust toolchain)
    z_poly = os.path.join(
        BASE_PATH, "uzp-artifact", "z_polyhedrator", "target", "release", "z_polyhedrator"
    )
    return os.path.isfile(z_poly)


def test_spmv_uzp_sparse_dispatch():
    """Test UZP sparse-dispatch backend against scipy SpMV.

    This verifies that:
    - SABLE-generated code can dispatch the sparse CSR remainder to UZP
    - UZP preparation (z_polyhedrator + spf_aggregator) runs via uzp_prepare.sh
    - Only the UZP kernel execution is timed
    """
    if not _uzp_available():
        pytest.skip("UZP toolchain not available; skipping UZP dispatch test")

    # Load matrix from MTX file
    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape

    # Convert to fully sparse VBRC format (no dense blocks)
    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, [])
    assert len(val) == 0, "Expected fully sparse matrix (val should be empty)"
    assert len(ublocks) > 0, "Expected sparse remainder to exist (ublocks should be non-empty)"

    # Write VBRC file (required by generated code)
    vbr_dir = os.path.join("tests")
    filename = "example3_uzp"
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)

    # Generate .mtx from the sparse CSR part (offline step)
    nrows = rpntr[-1]
    ncols = cpntr[-1]
    sparse_mat = scipy.sparse.csr_array((csr_val, indices, indptr), shape=(nrows, ncols))
    sparse_mtx_path = os.path.join(BASE_PATH, "tests", f"{filename}.mtx")
    scipy.io.mmwrite(sparse_mtx_path, sparse_mat)

    # Write dense vector file (required by generated code)
    write_dense_vector(1.0, cols)

    # Generate UZP dispatch code (generate directly in tests/ directory)
    dir_name = os.path.join("tests")
    gen_single_threaded_spmv_naive_uzp(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1,
        sparse_mtx_path=sparse_mtx_path,
    )

    # Compile the generated code, linking in UZP executor sources
    uzp_genex_dir = os.path.join(BASE_PATH, "uzp-artifact", "spmv-executors", "uzp-genex")
    uzp_sources = [
        os.path.join(uzp_genex_dir, "polybench.c"),
        os.path.join(uzp_genex_dir, "spf_structure.c"),
        os.path.join(uzp_genex_dir, "spf_executors.c"),
        os.path.join(uzp_genex_dir, "spf_executors_uninc.c"),
    ]
    subprocess.check_call(
        ["gcc", "-o", filename, f"{filename}.c"] + uzp_sources + CFLAGS + [f"-I{uzp_genex_dir}", "-DGEN_EXECUTOR_SPMV_ORIGINAL", "-lm"],
        cwd="tests"
    )

    # Run the executable and capture output
    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")

    # Skip timing lines and extract result vector
    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            float(line)
            result_lines.append(line)
        except ValueError:
            pass

    y_generated = numpy.array([float(x) for x in result_lines])
    x = numpy.ones(cols)
    y_expected = A.dot(x)
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-8, atol=1e-8)


def test_spmv_uzp_sparse_dispatch_mkl():
    """Test UZP sparse-dispatch + MKL dense blocks backend against scipy SpMV."""
    if not _uzp_available():
        pytest.skip("UZP toolchain not available; skipping UZP dispatch MKL test")
    if not MKL_AVAILABLE:
        pytest.skip("MKL not available; skipping MKL dense test")

    mtx_path = os.path.join(BASE_PATH, "tests", "example3.mtx")
    mtx = scipy.io.mmread(mtx_path)
    A = scipy.sparse.csc_matrix(mtx, copy=False)
    rows, cols = A.shape

    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = \
        convert_sparse_to_vbrc_with_blocks(A, [])
    assert len(val) == 0, "Expected fully sparse matrix (val should be empty)"
    assert len(ublocks) > 0, "Expected sparse remainder to exist (ublocks should be non-empty)"

    vbr_dir = os.path.join("tests")
    filename = "example3_uzp_mkl"
    _write_vbrc_file(filename, vbr_dir, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val)
    write_dense_vector(1.0, cols)

    # Generate .mtx from the sparse CSR part (offline step)
    nrows = rpntr[-1]
    ncols = cpntr[-1]
    sparse_mat = scipy.sparse.csr_array((csr_val, indices, indptr), shape=(nrows, ncols))
    sparse_mtx_path = os.path.join(BASE_PATH, "tests", f"{filename}.mtx")
    scipy.io.mmwrite(sparse_mtx_path, sparse_mat)

    dir_name = os.path.join("tests")
    gen_single_threaded_spmv_mkl_uzp(
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks,
        indptr, indices, csr_val, dir_name, filename, vbr_dir, bench=1,
        sparse_mtx_path=sparse_mtx_path,
    )

    uzp_genex_dir = os.path.join(BASE_PATH, "uzp-artifact", "spmv-executors", "uzp-genex")
    uzp_sources = [
        os.path.join(uzp_genex_dir, "polybench.c"),
        os.path.join(uzp_genex_dir, "spf_structure.c"),
        os.path.join(uzp_genex_dir, "spf_executors.c"),
        os.path.join(uzp_genex_dir, "spf_executors_uninc.c"),
    ]
    subprocess.check_call(
        ["gcc", "-o", filename, f"{filename}.c"] + uzp_sources + CFLAGS + MKL_FLAGS + [f"-I{uzp_genex_dir}", "-DGEN_EXECUTOR_SPMV_ORIGINAL", "-lm"],
        cwd="tests"
    )

    output = subprocess.check_output([f"./{filename}"], cwd="tests").decode("utf-8").split("\n")
    result_lines = []
    skip_timing = True
    for line in output:
        line = line.strip()
        if not line:
            skip_timing = False
            continue
        if skip_timing:
            continue
        if line.startswith("Sparse:") or line.startswith("Dense:") or line.startswith("Dense Block"):
            continue
        try:
            float(line)
            result_lines.append(line)
        except ValueError:
            pass

    y_generated = numpy.array([float(x) for x in result_lines])
    x = numpy.ones(cols)
    y_expected = A.dot(x)
    numpy.testing.assert_allclose(y_generated, y_expected, rtol=1e-8, atol=1e-8)

# def test_vbrc_matrix_gen():
#     """Test vbrc_matrix_gen function by generating a small matrix and verifying it can be read back."""
#     # Generate a small 6x6 matrix with 2x2 blocks
#     # 2 row splits, 2 col splits = 4 blocks total
#     # 2 dense blocks, 50% zeros in dense blocks, 1 sparse block
#     filename = vbrc_matrix_gen(
#         m=6, n=6, 
#         partitioning="uniform", 
#         row_split=2, col_split=2, 
#         num_dense=2, perc_dense_zeros=50, 
#         num_sparse=1, 
#         dense_blocks_only=False, 
#         vbr_dir="tests", 
#         density=80  # Use density threshold instead of ML model
#     )
    
#     # Read back the generated file
#     vbrc_path = os.path.join("tests", f"{filename}.vbrc")
#     val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(vbrc_path)
    
#     # Verify basic structure
#     assert len(rpntr) == 3  # 2 splits + 1
#     assert len(cpntr) == 3  # 2 splits + 1
#     assert rpntr[0] == 0 and rpntr[-1] == 6
#     assert cpntr[0] == 0 and cpntr[-1] == 6
    
#     # Verify block structure
#     assert len(bpntrb) == 2  # 2 row splits
#     assert len(bpntre) == 2  # 2 row splits
    
#     # Verify that we have some data
#     assert len(val) > 0 or len(csr_val) > 0
    
#     # Verify that indx starts with 0
#     assert indx[0] == 0
    
#     # Verify that bindx contains valid column indices
#     assert all(0 <= idx < 2 for idx in bindx)  # 2 column splits
    
#     # Verify that ublocks, indptr, indices, csr_val are consistent
#     if len(ublocks) > 0:
#         assert len(indptr) > 0
#         assert len(indices) > 0
#         assert len(csr_val) > 0
#         # indptr should have length equal to number of rows + 1
#         assert len(indptr) == 7  # 6 rows + 1
    
#     # Clean up the generated file
#     if os.path.exists(vbrc_path):
#         os.remove(vbrc_path)
#     # Also clean up the directory if it's empty
#     if os.path.exists("tests") and not os.listdir("tests"):
#         os.rmdir("tests")
