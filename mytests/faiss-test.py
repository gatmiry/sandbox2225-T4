import numpy as np

#### create index for the dataset to do knn search in
d = 64
nb = 1000000
nq = 100000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')
import faiss
index_flat = faiss.IndexFlatL2(d)
res = faiss.StandardGpuResources()
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index_flat.add(xb)
print(gpu_index_flat.ntotal)

## searching dataset
k = 4
D, I = gpu_index_flat.search(xq, k)
print(I[:5])