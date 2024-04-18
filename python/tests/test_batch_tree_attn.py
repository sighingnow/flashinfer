"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Tuple

import numpy
import torch

import flashinfer

def generate_requests(
    context_lens: List[int],
    query_lens: List[List[int]],
    num_kv_heads,
    num_qo_heads,
    head_dim,
    dtype=torch.float16,
    device='cuda',
) -> Tuple[List[torch.Tensor],
           List[torch.Tensor], List[torch.Tensor],
           List[torch.Tensor], List[torch.Tensor]]:
    k_caches, v_caches, queries, keys, values = [], [], [], [], []
    for context_len, query_lens_ in zip(context_lens, query_lens):
        k_cache = torch.randn((context_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        v_cache = torch.randn((context_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        k_caches.append([k_cache] * len(query_lens_))
        v_caches.append([v_cache] * len(query_lens_))
        queries.append([
            torch.full((query_len, num_qo_heads, head_dim), 0.5, dtype=dtype, device=device)
            for query_len in query_lens_
        ])
        keys.append([
            torch.full((query_len, num_kv_heads, head_dim), 0.2, dtype=dtype, device=device)
            for query_len in query_lens_
        ])
        values.append([
            torch.full((query_len, num_kv_heads, head_dim), 0.1, dtype=dtype, device=device)
            for query_len in query_lens_
        ])
    return queries, k_caches, v_caches, keys, values

def allocate_table_blocks(
    context_lens: List[int],
    query_lens: List[int],
    page_size: int,
    compressed: bool = False,
) -> List[List[int]]:
    start_table, block_table = 0, []
    for context_len, query_lens_ in zip(context_lens, query_lens):
        if compressed:
            num_required_blocks = (context_len + sum(query_lens_) + page_size - 1) // page_size
            block_table.append(list(range(start_table, start_table + num_required_blocks)))
            start_table += num_required_blocks
        else:
            for query_len in query_lens_:
                num_required_blocks = (context_len + query_len + page_size - 1) // page_size
                block_table.append(list(range(start_table, start_table + num_required_blocks)))
                start_table += num_required_blocks
    return block_table

def compute_batch_prefill(
    context_lens: List[int],
    query_lens: List[List[int]],
    k_caches: List[List[torch.Tensor]],
    v_caches: List[List[torch.Tensor]],
    queries: List[List[torch.Tensor]],
    keys: List[List[torch.Tensor]],
    values: List[List[torch.Tensor]],
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_data: torch.Tensor,  # [num_pages, 2, page_size, num_kv_heads, head_dim]
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    dtype=torch.float16,
    device='cuda',
):
    query = torch.cat([q for qs in queries for q in qs], dim=0)
    block_tables = allocate_table_blocks(context_lens, query_lens, kv_data.size(2))

    # store kv cache
    flat_context_lens = [
        context_len for context_len, query_lens_ in zip(context_lens, query_lens) for query_len in query_lens_
    ]
    flat_query_lens = [
        query_len for query_lens_ in query_lens for query_len in query_lens_
    ]
    block_table_index = 0
    for context_len, query_lens_, k_caches_, v_caches_, ks, vs in zip(
        context_lens, query_lens, k_caches, v_caches, keys, values,
    ):
        for query_len, k_cache, v_cache, k, v in zip(query_lens_, k_caches_, v_caches_, ks, vs):
            block_table = block_tables[block_table_index]
            block_table_index += 1
            slots = [
                block_table[p // kv_data.size(2)] * kv_data.size(2) + p % kv_data.size(2)
                for p in range(context_len + query_len)
            ]
            blocks = [slot // kv_data.size(2) for slot in slots]
            block_indices = [slot % kv_data.size(2) for slot in slots]
            kv_data[blocks, 0, block_indices, :, :] = torch.cat([k_cache, k], dim=0)
            kv_data[blocks, 1, block_indices, :, :] = torch.cat([v_cache, v], dim=0)

    flat_context_lens = torch.tensor(flat_context_lens,
                                     device=device, dtype=torch.int32)
    flat_query_lens = torch.tensor(flat_query_lens,
                                   device=device, dtype=torch.int32)
    qo_indptr = torch.zeros((len(flat_query_lens) + 1,),
                            device=device, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(flat_query_lens, dim=0)

    paged_kv_indices = torch.tensor([block for blocks in block_tables for block in blocks],
                                    device=device, dtype=torch.int32)
    paged_kv_indptr = torch.zeros((len(block_tables) + 1,),
                                  device=device, dtype=torch.int32)
    paged_kv_indptr[1:] = torch.cumsum(torch.tensor([len(blocks) for blocks in block_tables],
                                                    device=device, dtype=torch.int32), dim=0)
    paged_last_page_len = (flat_context_lens + flat_query_lens - 1) % kv_data.size(2) + 1

    # print(f'{qo_indptr=}')
    # print(f'{paged_kv_indptr=}')
    # print(f'{paged_kv_indices=}')
    # print(f'{paged_last_page_len=}')

    wrapper.begin_forward(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    o = wrapper.forward(
        q=query,
        paged_kv_data=kv_data,
        causal=True,
        pos_encoding_mode="NONE",
        allow_fp16_qk_reduction=True,
        sm_scale=1.0,
    )
    wrapper.end_forward()

    kv_layout = "NHD"
    for i in range(len(flat_context_lens)):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = query[qo_indptr[i] : qo_indptr[i + 1]]
        ki = torch.cat(
            [
                kv_data[paged_kv_indptr[i] : paged_kv_indptr[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data[paged_kv_indptr[i + 1] - 1, 0, :, : paged_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data[paged_kv_indptr[i + 1] - 1, 0, : paged_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                kv_data[paged_kv_indptr[i] : paged_kv_indptr[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data[paged_kv_indptr[i + 1] - 1, 1, :, : paged_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data[paged_kv_indptr[i + 1] - 1, 1, : paged_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi, ki, vi, causal=True, pos_encoding_mode="NONE",
            allow_fp16_qk_reduction=True, sm_scale=1.0,
        )
        o_i_np = o[qo_indptr[i] : qo_indptr[i + 1]].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)

    return o


def compute_batch_prefill_compressed(
    context_lens: List[int],
    query_lens: List[List[int]],
    k_caches: List[List[torch.Tensor]],
    v_caches: List[List[torch.Tensor]],
    queries: List[List[torch.Tensor]],
    keys: List[List[torch.Tensor]],
    values: List[List[torch.Tensor]],
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_data: torch.Tensor,  # [num_pages, 2, page_size, num_kv_heads, head_dim]
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    dtype=torch.float16,
    device='cuda',
):
    query = torch.cat([q for qs in queries for q in qs], dim=0)
    block_tables = allocate_table_blocks(context_lens, query_lens, kv_data.size(2), compressed=True)

    # store kv cache
    flat_context_lens = context_lens
    flat_query_lens = [sum(query_lens_) for query_lens_ in query_lens]

    # for compressed attention, the sub query_lens in each query should be same
    assert len(set(tuple(query_lens_) for query_lens_ in query_lens)) == 1

    total_query_len = sum(query_lens[0])
    mask = [[0 for _ in range(total_query_len)] for _ in range(total_query_len)]
    acc = 0
    for query_len in query_lens[0]:
        for i in range(query_len):
            for j in range(query_len):
                if i >= j:
                    mask[acc + i][acc + j] = 1
        acc += query_len
    mask = torch.tensor(mask, dtype=torch.int8, device=device)
    print(f'{mask=}')

    for context_len, query_lens_, k_caches_, v_caches_, ks, vs, block_table in zip(
        context_lens, query_lens, k_caches, v_caches, keys, values, block_tables,
    ):
        slots = [
            block_table[p // kv_data.size(2)] * kv_data.size(2) + p % kv_data.size(2)
            for p in range(context_len + sum(query_lens_))
        ]
        blocks = [slot // kv_data.size(2) for slot in slots]
        block_indices = [slot % kv_data.size(2) for slot in slots]
        for k_cache, v_cache in zip(k_caches_, v_caches_):
            torch.testing.assert_close(k_caches_[0], k_cache)
            torch.testing.assert_close(v_caches_[0], v_cache)
        kv_data[blocks, 0, block_indices, :, :] = torch.cat([k_caches_[0]] + ks, dim=0)
        kv_data[blocks, 1, block_indices, :, :] = torch.cat([v_caches_[0]] + vs, dim=0)

    flat_context_lens = torch.tensor(flat_context_lens,
                                     device=device, dtype=torch.int32)
    flat_query_lens = torch.tensor(flat_query_lens,
                                   device=device, dtype=torch.int32)
    qo_indptr = torch.zeros((len(flat_query_lens) + 1,),
                            device=device, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(flat_query_lens, dim=0)

    paged_kv_indices = torch.tensor([block for blocks in block_tables for block in blocks],
                                    device=device, dtype=torch.int32)
    paged_kv_indptr = torch.zeros((len(block_tables) + 1,),
                                  device=device, dtype=torch.int32)
    paged_kv_indptr[1:] = torch.cumsum(torch.tensor([len(blocks) for blocks in block_tables],
                                                    device=device, dtype=torch.int32), dim=0)
    paged_last_page_len = (flat_context_lens + flat_query_lens - 1) % kv_data.size(2) + 1

    # print(f'{qo_indptr=}')
    # print(f'{paged_kv_indptr=}')
    # print(f'{paged_kv_indices=}')
    # print(f'{paged_last_page_len=}')

    wrapper.begin_forward(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    o = wrapper.forward(
        q=query,
        paged_kv_data=kv_data,
        causal=True,
        pos_encoding_mode="NONE",
        allow_fp16_qk_reduction=True,
        sm_scale=1.0,
        mask=mask,
    )
    wrapper.end_forward()
    return o


def test_batch_prefill_with_paged_kv_cache(
    context_lens,
    query_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
):
    total_num_pages = 4096
    assert kv_layout == "NHD"
    kv_data = torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim,
                          dtype=torch.float16, device='cuda')
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device='cuda')
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )

    queries, k_caches, v_caches, keys, values = generate_requests(
        context_lens, query_lens, num_kv_heads, num_qo_heads, head_dim
    )
    torch.cuda.synchronize()
    o1 = compute_batch_prefill(
        context_lens, query_lens, k_caches, v_caches, queries, keys, values,
        num_kv_heads, num_qo_heads, head_dim, kv_data, wrapper
    )
    torch.cuda.synchronize()
    o2 = compute_batch_prefill_compressed(
        context_lens, query_lens, k_caches, v_caches, queries, keys, values,
        num_kv_heads, num_qo_heads, head_dim, kv_data, wrapper
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(o1, o2)


if __name__ == "__main__":
    context_lens = [
        114,
        2,
        1144,
        45,
        2048,
        # 3,
        # 1,
        # 4,
    ]
    query_lens = [
        [3, 2, 4, 7],
        [3, 2, 4, 7],
        [3, 2, 4, 7],
        [3, 2, 4, 7],
        [3, 2, 4, 7],
        # [4],
        # [1, 2],
        # [2, 2],
    ]

    test_batch_prefill_with_paged_kv_cache(
        context_lens, query_lens, 16, 1, 1, 128, True, "NHD", "NONE"
    )
