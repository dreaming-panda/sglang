import torch
import block_sparse_attention
import pytest

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)

@pytest.mark.parametrize("lmk_page_size", [4, 8, 16])
@pytest.mark.parametrize("num_reqs", [1, 8, 16])
@pytest.mark.parametrize("avg_kv_size", [4, 16, 64])
@pytest.mark.parametrize("topk", [4, 8])
def test_block_topk(
    lmk_page_size: int,
    num_reqs: int,
    avg_kv_size: int,
    topk: int
):
    device = "cuda"
    head_dim = 128
    total_kv = num_reqs * avg_kv_size
    kv_req_idx = torch.randint(low=0, high=num_reqs, 
            size=(total_kv,), device=device, dtype=torch.int32)
    kv_indptr = torch.zeros(size=(num_reqs + 1,), device=device, dtype=torch.int32)
    kv_indices = torch.zeros(size=(total_kv,), device=device, dtype=torch.int32)
    for i in range(num_reqs):
        indices = torch.where(kv_req_idx == i)[0]
        kv_indptr[i+1] = kv_indptr[i] + len(indices)
        kv_indices[kv_indptr[i]: kv_indptr[i+1]].copy_(indices.to(torch.int32))
    
    query = torch.randn(size=(num_reqs, head_dim), device=device, dtype=torch.bfloat16)
    lmk = torch.randn(size=(total_kv, head_dim), device=device, dtype=torch.bfloat16)
    
    workload_indptr = torch.zeros(size=(num_reqs + 1,), device=device, dtype=torch.int32)
    workload_per_req = kv_indptr[1:] - kv_indptr[:-1]
    sparse_workload_per_req = torch.clamp(workload_per_req, max=topk)
    sparse_kv_indptr = torch.zeros(size=(num_reqs + 1,), device=device, dtype=torch.int32)
    sparse_kv_indptr[1:].copy_(sparse_workload_per_req.cumsum(dim=0))
    sparse_kv_indices = torch.zeros(size=(sparse_kv_indptr[-1],), device=device, dtype=torch.int32)
    
    
    paged_workload_per_req = (workload_per_req + lmk_page_size - 1) // lmk_page_size
    workload_indptr[1:].copy_(paged_workload_per_req.cumsum(dim=0))
    
    req_table = torch.zeros(size=(workload_indptr[-1],), device=device, dtype=torch.int32)
    for i in range(num_reqs):
        req_table[workload_indptr[i]: workload_indptr[i+1]] = i
    
    score = torch.full(size=(num_reqs, max(total_kv, topk)), fill_value=-torch.inf, device=device, dtype=torch.bfloat16)
    score_ref = torch.full(size=(num_reqs, max(total_kv, topk)), fill_value=-torch.inf, device=device, dtype=torch.bfloat16)
    
    for i in range(num_reqs):
        q = query[i]
        indices = kv_indices[kv_indptr[i]: kv_indptr[i+1]]
        k = lmk[indices]
        o = torch.mv(k, q)
        score_ref[i][0: kv_indptr[i+1]-kv_indptr[i]].copy_(o)
    
    
    block_sparse_attention.build_block_topk_indices(
        query,
        lmk,
        score,
        kv_indptr,
        kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices,
        workload_indptr,
        req_table,
        lmk_page_size,
        topk
    )
    
    for i in range(num_reqs):
        l = kv_indptr[i+1] - kv_indptr[i]
        assert torch.allclose(score[i][1:l-1], score_ref[i][1:l-1], rtol=0.05, atol=0.1)
        assert score[i][0].isinf()
        assert score[i][l-1].isinf()
        score_ref[i][0] = -torch.inf
        score_ref[i][l-1] = -torch.inf
        if l > topk:
            ref_selected_indices = kv_indices[kv_indptr[i] + score_ref[i].topk(topk-2).indices]
            
            ref_selected_indices = ref_selected_indices.tolist()
            #ref_selected_indices.append(kv_indices[kv_indptr[i]].item())
            #ref_selected_indices.append(kv_indices[kv_indptr[i+1]-1].item())
        
        elif l <= topk:
            ref_selected_indices = kv_indices[kv_indptr[i]: kv_indptr[i+1]].tolist()
        
        selected_indices = set(sparse_kv_indices[sparse_kv_indptr[i]: sparse_kv_indptr[i+1]].tolist())
        if l >= 1:
            assert kv_indices[kv_indptr[i]].item() in selected_indices
            assert kv_indices[kv_indptr[i+1] - 1].item() in selected_indices
            assert ref_selected_indices[0] in selected_indices
        # assert (ref_selected_indices[1]) in selected_indices
test_block_topk(4, 16, 4, 4)