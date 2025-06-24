import triton
import triton.language as tl


@triton.jit
def sa_create_flashinfer_kv_indices_paged_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr, # [bsz]
    paged_seq_lens_ptr, # [bsz * NUM_KV_HEADS]
    kv_indptr, # [bsz * NUM_KV_HEADS + 1]
    kv_indices_ptr, # [num_total_kv_pages * NUM_KV_HEADS + 1]
    req_to_token_ptr_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr
):  
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)
    req_id = pid // NUM_KV_HEAD
    head_id = pid % NUM_KV_HEAD

    req_pool_index = tl.load(req_pool_indices_ptr + req_id)
    kv_indices_offset = tl.load(kv_indptr + pid)
    
    paged_kv_len = tl.load(paged_seq_lens_ptr + pid)
    
    num_loop = tl.cdiv(paged_kv_len, BLOCK_SIZE)
    
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = (tl.arange(0, BLOCK_SIZE).to(tl.int64) \
        + i * BLOCK_SIZE)
        mask = offset < paged_kv_len
        
        
        token_origin = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset * PAGE_SIZE,
            mask=mask,
        )
        
        token_trans = (token_origin // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_origin %  PAGE_SIZE
        
        page_id = token_trans // PAGE_SIZE
        
        tl.store(kv_indices_ptr + kv_indices_offset + offset, page_id, mask=mask)
        

@triton.jit
def sa_create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr, # [bsz]
    page_kernel_lens_ptr, # [bsz * NUM_KV_HEADS]
    kv_indptr, # [bsz * NUM_KV_HEADS + 1]
    kv_start_idx, # Always None
    kv_indices_ptr, # [num_total_kv_tokens * NUM_KV_HEADS + 1]
    req_to_token_ptr_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr
):  
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)
    req_id = pid // NUM_KV_HEAD
    head_id = pid % NUM_KV_HEAD
    
    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + req_id)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        
        data_trans = (data // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + data %  PAGE_SIZE
        
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data_trans, mask=mask)


@triton.jit
def q_transpose_triton(
    src,
    dst,
    indptr,
    NUM_KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr
):  
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)
    group = tl.arange(0, GROUP)
    start = tl.load(indptr + req_id).to(tl.int32)
    end = tl.load(indptr + req_id + 1).to(tl.int32)
    num_loop = end - start
    
    for i in range(num_loop):
        src_tensor = tl.load(src + \
        start * NUM_KV_HEAD * GROUP * HEAD_DIM + i * NUM_KV_HEAD * GROUP * HEAD_DIM + \
        head_id * GROUP * HEAD_DIM  + group[:,None] * HEAD_DIM + dim[None,:])
        
        dst_ptr = dst + start * NUM_KV_HEAD * GROUP * HEAD_DIM \
            + head_id * num_loop* GROUP * HEAD_DIM + i * GROUP * HEAD_DIM + group[:,None] * HEAD_DIM + dim[None,:]
        
        tl.store(dst_ptr, src_tensor)
        

@triton.jit
def o_transpose_triton(
    src,
    dst,
    score_src,
    score_dst,
    indptr,
    NUM_KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr
):  
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)
    group = tl.arange(0, GROUP)
    start = tl.load(indptr + req_id).to(tl.int32)
    end = tl.load(indptr + req_id + 1).to(tl.int32)
    num_loop = end - start
    
    for i in range(num_loop):
        
        src_tensor = tl.load(src + start * NUM_KV_HEAD * GROUP * HEAD_DIM \
            + head_id * num_loop* GROUP * HEAD_DIM + i * GROUP * HEAD_DIM + group[:,None] * HEAD_DIM + dim[None,:])
        
        dst_ptr = dst + start * NUM_KV_HEAD * GROUP * HEAD_DIM + i * NUM_KV_HEAD * GROUP * HEAD_DIM + \
        head_id * GROUP * HEAD_DIM  + group[:,None] * HEAD_DIM + dim[None,:]
        
        tl.store(dst_ptr, src_tensor)
        
        score_src_tensor = tl.load(score_src + start * NUM_KV_HEAD * GROUP \
            + head_id * num_loop* GROUP + i * GROUP  + group)
        
        score_dst_ptr = score_dst + start * NUM_KV_HEAD * GROUP + i * NUM_KV_HEAD * GROUP+ \
        head_id * GROUP  + group
        
        tl.store(score_dst_ptr, score_src_tensor)
        
        
        
    