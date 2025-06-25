#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <torch/torch.h>


constexpr int next_power_of_2(int n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}


__global__ void get_local_indices(
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    const int* __restrict__ sparse_indptr,
    int* __restrict__ sparse_indices,
    const int TopK
){

    int batch_idx = blockIdx.x;
    const int* kv_indices_start = kv_indices + kv_indptr[batch_idx];
    int* sparse_indices_start = sparse_indices + sparse_indptr[batch_idx];
    int active_kv_blocks = sparse_indptr[batch_idx + 1] - sparse_indptr[batch_idx];
    int total_kv_blocks = kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
    int tid = threadIdx.x;
    if(total_kv_blocks <= TopK){
        if(tid < active_kv_blocks){
            sparse_indices_start[tid] = kv_indices_start[tid];
        }
    } else {
        if(tid < TopK - 1){
            sparse_indices_start[tid + 1] = kv_indices_start[total_kv_blocks - TopK + tid + 1];
        }
        if(tid == 0){
            sparse_indices_start[0] = kv_indices_start[0];
        }
    }
    

}
void build_local_indices(
    at::Tensor q,
    at::Tensor landmarks,
    at::Tensor score_buffer,
    at::Tensor kv_indptr,
    at::Tensor kv_indices,
    at::Tensor sparse_indptr,
    at::Tensor sparse_indices,
    const int TopK
){

    int BATCH_SIZE = kv_indptr.size(0) - 1;

    get_local_indices<<<BATCH_SIZE, next_power_of_2(TopK)>>>(
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        sparse_indptr.data_ptr<int>(),
        sparse_indices.data_ptr<int>(),
        TopK
    );

}

// PyBind wrapper
PYBIND11_MODULE(block_sparse_attention, m) {
    m.def("build_local_indices", &build_local_indices, "Streaming LLM block implementation");
}