import scipy.io
import pathlib
import os
import scipy.sparse

from src.fileio import read_vbr

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

def gen_spmv_cusparse_files():
    dir_name = "Generated_SpMV_cuSparse"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    mtx_files = os.listdir(os.path.join(BASE_PATH, "Generated_MMarket"))
    for filename in mtx_files:
        gen_spmv_cusparse_file(filename[:-4], dir_name, f'{BASE_PATH}/Generated_VBR/', False)

def gen_spmm_cusparse_files():
    dir_name = "Generated_SpMM_cuSparse"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    mtx_files = os.listdir(os.path.join(BASE_PATH, "Generated_MMarket"))
    for filename in mtx_files:
        gen_spmm_cusparse_file(filename[:-4], dir_name, f'{BASE_PATH}/Generated_VBR/', False)

def gen_spmv_cusparse_file(filename, dir_name, vbr_dir, testing):
    with open(os.path.join(BASE_PATH, dir_name, filename + ".c"), "w") as f:
        if testing:
            mtx = scipy.io.mmread(f'{dir_name}/{filename}.mtx')
        else:
            mtx = scipy.io.mmread(f'{BASE_PATH}/Generated_MMarket/{filename}.mtx')
        vbr_path = f"{vbr_dir}/{filename}.vbr"
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
        csr = scipy.sparse.csr_matrix(mtx)
        content = """#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <assert.h>
#include <sys/time.h>

#define CHECK_CUDA(func)                                                       \\
{                                                                              \\
    cudaError_t status = (func);                                               \\
    if (status != cudaSuccess) {                                               \\
        printf("CUDA API failed at line %d with error: %s (%d)",             \\
               __LINE__, cudaGetErrorString(status), status);                  \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

#define CHECK_CUSPARSE(func)                                                   \\
{                                                                              \\
    cusparseStatus_t status = (func);                                          \\
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \\
        printf("CUSPARSE API failed at line %d with error: %s (%d)",         \\
               __LINE__, cusparseGetErrorString(status), status);              \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

int main(void) {
"""
        content += f"\tconst int A_num_rows      = {csr.shape[0]};\n"
        content += f"\tconst int A_num_cols      = {csr.shape[1]};\n"
        content += f"\tconst int A_nnz           = {csr.nnz};\n"
        content += f"\tint* hA_csrOffsets = malloc({len(csr.indptr)} * sizeof(int));\n"
        content += f"\tint* hA_columns = malloc({len(csr.indices)} * sizeof(int));\n"
        for count, elem in enumerate(csr.indptr):
            content += f"\thA_csrOffsets[{count}] = {elem};\n"
        for count, elem in enumerate(csr.indices):
            content += f"\thA_columns[{count}] = {elem};\n"
        content += f"\tfloat*    hA_values       = (float*)malloc({csr.nnz} * sizeof(float));\n"
        content += f"\tfloat* hX = (float*)malloc({rpntr[-1]} * sizeof(float));\n"
        val_file = os.path.join(dir_name, filename + ".cusparse")
        with open(val_file, "w") as f2:
            f2.write("val=[")
            for i, value in enumerate(csr.data):
                if i != 0:
                    f2.write(",")
                f2.write(f"{value}")
            f2.write("]\n")
        content += f"\tFILE *file1 = fopen(\"{os.path.abspath(val_file)}\", \"r\");\n"
        content += "\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n"
        content += '\tint x_size = 0, val_size = 0;\n'
        content += "\tchar c;\n"
        content += '''\tassert(fscanf(file1, "val=[%f", &hA_values[val_size]) == 1.0);
    val_size++;
    while (1) {
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {
            assert(fscanf(file1, "%f", &hA_values[val_size]) == 1.0);
            val_size++;
        } else if (c == ']') {
            break;
        } else {
            assert(0);
        }
    }
    if(fscanf(file1, "%c", &c));
    assert(c=='\\n');
    fclose(file1);\n'''
        content += f'\tFILE *file2 = fopen("{BASE_PATH}/generated_vector_{rpntr[-1]}.vector", "r");\n'
        content += '\tif (file2 == NULL) { printf("Error opening file2"); return 1; }'
        content += '''
    while (x_size < {0} && fscanf(file2, "%f,", &hX[x_size]) == 1) {{
        x_size++;
    }}
    fclose(file2);\n'''.format(csr.shape[1])
        content += f"\t float* hY = calloc({csr.shape[0]}, sizeof(float));\n"
        content += '''float     alpha           = 1.0f;
    float     beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    
    struct timeval t1;
	gettimeofday(&t1, NULL);
	long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
    
    struct timeval t2;
	gettimeofday(&t2, NULL);
	long t2s = t2.tv_sec * 1000000L + t2.tv_usec;'''

        content += f'printf("{filename}= %lu\\n", t2s-t1s);\n'
    
        content +='''// destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    for (int i = 0; i < A_num_rows; i++) {
        printf("%f\\n", hY[i]);
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);
    free(hX);
    free(hY);
    return EXIT_SUCCESS;
}
'''
        f.write(content)

def gen_spmm_cusparse_file(filename, dir_name, vbr_dir, testing):
    with open(os.path.join(BASE_PATH, dir_name, filename + ".c"), "w") as f:
        if testing:
            mtx = scipy.io.mmread(f'{dir_name}/{filename}.mtx')
        else:
            mtx = scipy.io.mmread(f'{BASE_PATH}/Generated_MMarket/{filename}.mtx')
        vbr_path = f"{vbr_dir}/{filename}.vbr"
        val, indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(vbr_path)
        csr = scipy.sparse.csr_matrix(mtx)
        content = """#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <assert.h>
#include <sys/time.h>

#define CHECK_CUDA(func)                                                       \\
{                                                                              \\
    cudaError_t status = (func);                                               \\
    if (status != cudaSuccess) {                                               \\
        printf("CUDA API failed at line %d with error: %s (%d)",             \\
               __LINE__, cudaGetErrorString(status), status);                  \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

#define CHECK_CUSPARSE(func)                                                   \\
{                                                                              \\
    cusparseStatus_t status = (func);                                          \\
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \\
        printf("CUSPARSE API failed at line %d with error: %s (%d)",         \\
               __LINE__, cusparseGetErrorString(status), status);              \\
        return EXIT_FAILURE;                                                   \\
    }                                                                          \\
}

int main(void) {
"""
        content += f"\tconst int A_num_rows      = {csr.shape[0]};\n"
        content += f"\tconst int A_num_cols      = {csr.shape[1]};\n"
        content += f"\tconst int A_nnz           = {csr.nnz};\n"
        content += f"\tint B_num_rows = A_num_cols;\n"
        content += f"\tint B_num_cols = 512;\n"
        content += """int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
"""
        content += f"\tint* hA_csrOffsets = malloc({len(csr.indptr)} * sizeof(int));\n"
        content += f"\tint* hA_columns = malloc({len(csr.indices)} * sizeof(int));\n"
        for count, elem in enumerate(csr.indptr):
            content += f"\thA_csrOffsets[{count}] = {elem};\n"
        for count, elem in enumerate(csr.indices):
            content += f"\thA_columns[{count}] = {elem};\n"
        content += f"\tfloat*    hA_values       = (float*)malloc({csr.nnz} * sizeof(float));\n"
        val_file = os.path.join(dir_name, filename + ".cusparse")
        with open(val_file, "w") as f2:
            f2.write("val=[")
            for i, value in enumerate(csr.data):
                if i != 0:
                    f2.write(",")
                f2.write(f"{value}")
            f2.write("]\n")
        content += f"\tFILE *file1 = fopen(\"{os.path.abspath(val_file)}\", \"r\");\n"
        content += "\tif (file1 == NULL) { printf(\"Error opening file1\"); return 1; }\n"
        content += '\tint x_size = 0, val_size = 0;\n'
        content += "\tchar c;\n"
        content += '''\tassert(fscanf(file1, "val=[%f", &hA_values[val_size]) == 1.0);
    val_size++;
    while (1) {
        assert(fscanf(file1, "%c", &c) == 1);
        if (c == ',') {
            assert(fscanf(file1, "%f", &hA_values[val_size]) == 1.0);
            val_size++;
        } else if (c == ']') {
            break;
        } else {
            assert(0);
        }
    }
    if(fscanf(file1, "%c", &c));
    assert(c=='\\n');
    fclose(file1);\n'''
        content += f"\tfloat* hB = (float*)malloc({512 * cpntr[-1]} * sizeof(float));\n"
        content += f'\tFILE *file2 = fopen("{BASE_PATH}/generated_matrix_{rpntr[-1]}x512.matrix", "r");\n'
        content += '\tif (file2 == NULL) { printf("Error opening file2"); return 1; }'
        content += '''\tfor(int i = 0; i < {0}; i++) {{
        for(int j = 0; j < 512; j++) {{
            assert(fscanf(file2, "%f,", &hB[i*512+j]) == 1);
        }}
    }}
    fclose(file2);\n'''.format(rpntr[-1])
        content += f"\t float* hC = calloc({csr.shape[0] * 512}, sizeof(float));\n"
        content += '''float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    struct timeval t1;
	gettimeofday(&t1, NULL);
	long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    struct timeval t2;
	gettimeofday(&t2, NULL);
	long t2s = t2.tv_sec * 1000000L + t2.tv_usec;\n'''
        content += f'printf("{filename}= %lu\\n", t2s-t1s);\n'
        content +='''// destroy matrix descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    for (int i = 0; i < A_num_rows; i++) {
        for (int j=0; j<B_num_cols; j++)
            printf("%f\\n", hC[i+j*ldb]);
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);
    free(hB);
    free(hC);
    return EXIT_SUCCESS;
}
'''
        f.write(content)

if __name__ == "__main__":
    gen_spmv_cusparse_files()
    gen_spmm_cusparse_files()
