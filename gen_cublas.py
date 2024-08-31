import scipy.io
import pathlib
import os
from argparse import ArgumentParser

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

def gen_spmm_cublas_files(dense_blocks_only):
    if dense_blocks_only:
        mm_dir = "Generated_MMarket"
        dir_name = "Generated_SpMM_cuBLAS"
        vbr_dir = "Generated_VBR"
    else:
        mm_dir = "Generated_MMarket_Sparse"
        dir_name = "Generated_SpMM_cuBLAS_Sparse"
        vbr_dir = "Generated_VBR_Sparse"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    mtx_files = os.listdir(os.path.join(BASE_PATH, mm_dir))
    for filename in mtx_files:
        gen_spmm_cublas_file(filename[:-4], dir_name, f'{BASE_PATH}/{vbr_dir}/', mm_dir, False)

def gen_spmm_cublas_file(filename, dir_name, vbr_dir, mm_dir, testing):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(BASE_PATH, dir_name, filename + ".cu"), "w") as f:
        if testing:
            mtx = scipy.io.mmread(f'{dir_name}/{filename}.mtx')
        else:
            mtx = scipy.io.mmread(f'{BASE_PATH}/{mm_dir}/{filename}.mtx')
        mtx = mtx.todense(order='C')
        content = """#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <assert.h>
#include <sys/time.h>

#define CUDA_CHECK(func)                                                       \\
{                                                                              \\
    cudaError_t status = (func);                                               \\
    if (status != cudaSuccess) {                                               \\
        printf("CUDA API failed at line %d with error: %s (%d)",             \\
               __LINE__, cudaGetErrorString(status), status);                  \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

#define CUBLAS_CHECK(func)                                                       \\
{                                                                              \\
    cublasStatus_t status = (func);                                               \\
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \\
        printf("CUDA API failed at line %d with error: (%d)",             \\
               __LINE__, status);                  \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

int main(void) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
"""
        content += f"\tconst int m = {mtx.shape[0]};\n"
        content += f"\tconst int n = 512;\n"
        content += f"\tconst int k = {mtx.shape[1]};\n"
        content += "\tconst int lda = k;\n"
        content += "\tconst int ldb = n;\n"
        content += "\tconst int ldc = m;\n"
        content += f"\tconst int A_size = {mtx.shape[0] * mtx.shape[1]};\n"
        content += f"\tconst int B_size = {512 * mtx.shape[1]};\n"
        content += f"\tconst int C_size = {mtx.shape[0] * 512};\n"
        val_file = os.path.join(dir_name, filename + ".cublas")
        with open(val_file, "w") as f2:
            f2.write("val=[")
            for i, element in enumerate(mtx.flat):
                if i != 0:
                    f2.write(",")
                f2.write(f"{element}")
            f2.write("]\n")
        content += f"\tfloat* A = (float*)malloc({mtx.shape[0] * mtx.shape[1]} * sizeof(float));\n"
        content += f"\tFILE *file1 = fopen(\"{os.path.abspath(val_file)}\", \"r\");\n"
        content += "\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n"
        content += "\tchar c;\n"
        content += '''\tint val_size = 0;
    assert(fscanf(file1, "val=[%f", &A[val_size]) == 1.0);
    val_size++;
    while (1) {
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {
            assert(fscanf(file1, "%f", &A[val_size]) == 1.0);
            val_size++;
        } else if (c == ']') {
            break;
        } else {
            assert(0);
        }
    }
    if (fscanf(file1, "%c", &c));
    assert(c=='\\n');
    '''
        content += f"\tfloat* B = (float*)malloc({512 * mtx.shape[1]} * sizeof(float));\n"
        content += f'\tFILE *file2 = fopen("{BASE_PATH}/generated_matrix_{mtx.shape[0]}x512.matrix", "r");\n'
        content += '\tif (file2 == NULL) { printf("Error opening file2"); return 1; }'
        content += '''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &B[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(mtx.shape[0])
        content += f"\t float* C = (float*)calloc({mtx.shape[0] * 512}, sizeof(float));\n"
        content += '''const float alpha = 1.0f;
    const float beta = 0.0f;
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * A_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * B_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * C_size));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(float) * A_size, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeof(float) * B_size, cudaMemcpyHostToDevice,
                               stream));
    struct timeval t1;
	gettimeofday(&t1, NULL);
	long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
    cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    cudaDeviceSynchronize();
    struct timeval t2;
	gettimeofday(&t2, NULL);
	long t2s = t2.tv_sec * 1000000L + t2.tv_usec;
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeof(float) * C_size, cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    '''    
        content += f'printf("{filename} = %lu\\n", t2s-t1s);\n'
        content += '''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < n; j++) {{
            printf("%f\\n", C[j*m+i]);
        }}
    }}\n'''.format(mtx.shape[0])
        content += '''\tCUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}'''
        f.write(content)

if __name__ == "__main__":
    parser = ArgumentParser(description='Gen cuSparse')
    parser.add_argument("--dense-blocks-only", action="store_true", default=False, required=False)
    args = parser.parse_args()
    dense_blocks_only = args.dense_blocks_only
    if dense_blocks_only:
        gen_spmm_cublas_files(dense_blocks_only=True)
    else:
        gen_spmm_cublas_files(dense_blocks_only=False)
