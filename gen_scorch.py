import os
import pathlib

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

def gen_spmm_scorch_files(dense_blocks_only):
    if dense_blocks_only:
        mm_dir = "Generated_MMarket"
        dir_name = "Generated_SpMM_scorch"
    else:
        raise NotImplementedError("Sparse blocks not implemented yet")
        mm_dir = "Generated_MMarket_Sparse"
        dir_name = "Generated_SpMM_scorch_Sparse"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    mtx_files = os.listdir(os.path.join(BASE_PATH, mm_dir))
    for filename in mtx_files:
        gen_spmm_scorch_file(filename[:-4], dir_name, mm_dir, False)

def gen_spmm_scorch_file(filename, dir_name, mm_dir, testing):
    with open(os.path.join(BASE_PATH, dir_name, filename + ".py"), "w") as f:
        if testing:
            mtx_filename = f'{dir_name}/{filename}.mtx'
        else:
            mtx_filename = f'{BASE_PATH}/{mm_dir}/{filename}.mtx'
        content = f"""import numpy as np
import scorch as torch
import scipy
import time

mtx = scipy.io.mmread('{mtx_filename}')
coo = mtx.tocoo()
num_rows = coo.shape[0]
num_cols = coo.shape[1]
indices = np.array([coo.row, coo.col])
indices = torch.LongTensor(indices).to("cuda")
values = torch.FloatTensor(coo.data).to("cuda")
stensor = torch.sparse_coo_tensor(indices, values, coo.shape, device="cuda")
with open(f"/local/scratch/a/das160/SABLE/generated_matrix_{{num_cols}}x512.matrix", 'r') as file:
    content = file.readline().strip()
values_str_list = content.split(',')
values_float_list = [float(value) for value in values_str_list]
z = torch.tensor(values_float_list, device="cuda").reshape(num_cols, 512)
torch.cuda.synchronize()
start = time.perf_counter()
result = torch.matmul(stensor, z)
torch.cuda.synchronize()
end = time.perf_counter()
print('{filename} = ', (end - start)*1_000_000)
for i in range(num_rows):
    for j in range(512):
        print(result[i, j].item())
"""
        f.write(content)
        f.flush()
