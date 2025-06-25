#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <torch/torch.h>

#define NUM_SMS 142
#define HX 128
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

__global__ void get_block_scores(
    const __nv_bfloat162* __restrict__ q,
    const __nv_bfloat162* __restrict__ lmk,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    const int* __restrict__ workload_indptr,
    const int* __restrict__ req_table,
    __nv_bfloat16* __restrict__ scores_buffer,
    int SEQ_LEN,
    int LMK_PAGE_SIZE
) {
    int pid = blockIdx.x;
    int tid = threadIdx.x;
    int batch_idx = req_table[pid];

    int batch_start = kv_indptr[batch_idx];
    int batch_end = kv_indptr[batch_idx + 1];

    int batch_offset = pid - workload_indptr[batch_idx];
    bool is_first_page = (batch_offset == 0);
    bool is_last_page = (pid == (workload_indptr[batch_idx + 1] - 1));

    int page_len = min(LMK_PAGE_SIZE, batch_end - batch_start - batch_offset * LMK_PAGE_SIZE);

    const int HHEAD_DIM = int(HX/2);
    const __nv_bfloat162* q_ptr = q + batch_idx * HHEAD_DIM;
    const int* indices = kv_indices + batch_start + batch_offset * LMK_PAGE_SIZE;
    __nv_bfloat16* out = scores_buffer + batch_idx * SEQ_LEN + batch_offset * LMK_PAGE_SIZE;

    // Shared memory for q
    extern __shared__ __nv_bfloat162 q_shared[HHEAD_DIM];

    if (tid < HHEAD_DIM) {
        q_shared[tid] = q_ptr[tid];
    }

     __syncthreads();

    float acc = 0.0f;
    if (tid < page_len){
        int index = indices[tid];
        const __nv_bfloat162* lmk_ptr = lmk + HHEAD_DIM * index;
        for (int i = 0; i < HHEAD_DIM; ++i){
            __nv_bfloat162 val = __hmul2(lmk_ptr[i], q_shared[i]);
            acc += __bfloat162float(val.x);
            acc += __bfloat162float(val.y);
        }
        out[tid] = __float2bfloat16(acc);
    }
    if (is_first_page && tid == 0){
        out[0] = __float2bfloat16(-INFINITY);
    }
    if (is_last_page && tid == 0){
        out[page_len - 1] = __float2bfloat16(-INFINITY);
    }

}


__global__ void copy_block_topk_indices(
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    const int* __restrict__ sparse_indptr,
    int* __restrict__ sparse_indices,
    const int64_t* __restrict__ topk_indices,
    const int topk_indices_stride,
    const int TopK
){

    int batch_idx = blockIdx.x;
    const int* kv_indices_start = kv_indices + kv_indptr[batch_idx];
    int* sparse_indices_start = sparse_indices + sparse_indptr[batch_idx];
    const int64_t* topk_indices_start = topk_indices + batch_idx * topk_indices_stride;
    int active_kv_blocks = sparse_indptr[batch_idx + 1] - sparse_indptr[batch_idx];
    int total_kv_blocks = kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
    int tid = threadIdx.x;
    if(total_kv_blocks <= TopK){
        if(tid < active_kv_blocks){
            sparse_indices_start[tid] = kv_indices_start[tid];
        }
    } else {
        if(tid < TopK - 2){
            sparse_indices_start[tid + 1] = kv_indices_start[topk_indices_start[tid]];
        }
        if(tid == 0){
            sparse_indices_start[0] = kv_indices_start[0];
            sparse_indices_start[TopK - 1] = kv_indices_start[total_kv_blocks - 1];
        }
    }
}


void build_block_topk_indices(
    at::Tensor q,
    at::Tensor landmarks,
    at::Tensor score_buffer,
    at::Tensor kv_indptr,
    at::Tensor kv_indices,
    at::Tensor sparse_indptr,
    at::Tensor sparse_indices,
    at::Tensor workload_inptr,
    at::Tensor req_table,
    int LMK_PAGE_SIZE,
    int TopK
){
    const int num_threads = max(int(HX/2), LMK_PAGE_SIZE);
    const int num_blocks = req_table.size(0);
    get_block_scores<<<num_blocks, num_threads>>>(
        reinterpret_cast<__nv_bfloat162*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162*>(landmarks.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        workload_inptr.data_ptr<int>(),
        req_table.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(score_buffer.data_ptr<at::BFloat16>()),
        int(score_buffer.size(1)),
        LMK_PAGE_SIZE
    );

    assert(TopK >= 3);
    auto result = torch::topk(score_buffer, TopK - 2, /*dim=*/-1, true,  false);
    torch::Tensor values = std::get<0>(result);
    torch::Tensor indices = std::get<1>(result);

    const int batch_size = kv_indptr.size(0) - 1;
    copy_block_topk_indices<<<batch_size, next_power_of_2(TopK)>>>(
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        sparse_indptr.data_ptr<int>(),
        sparse_indices.data_ptr<int>(),
        indices.data_ptr<int64_t>(),
        indices.size(1),
        TopK
    );

}

// PyBind wrapper
PYBIND11_MODULE(block_sparse_attention, m) {
    m.def("build_local_indices", &build_local_indices, "Streaming LLM block implementation");
    m.def("build_block_topk_indices", &build_block_topk_indices, "block topk implementation");
}