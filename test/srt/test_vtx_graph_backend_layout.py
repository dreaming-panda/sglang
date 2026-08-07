"""
Unit tests for VTXGraphAttnBackend KV cache layout.

These tests validate that:
  1. VTXGraphCachePool's physical layout matches what _vtx_to_std_layout assumes.
  2. A round-trip (set_kv_buffer → read via std layout at token positions) returns
     the correct (token, head) data.

Why this exists:
  VTXGraphCachePool stores one page per (token, kv_head) pair, while FlashInfer
  indexes a paged cache by token and expects all heads of a token in one page.
  Reinterpreting one layout as the other type-checks and produces the right
  shape, but silently returns the wrong (token, head) rows -- during
  speculative decoding it showed up only as a collapsed acceptance length
  (~0.01), not as an error.

These tests lock the invariant in place so regressions are caught without
running end-to-end spec_dec benchmarks.
"""

import torch

try:
    import pytest
    HAS_PYTEST = True
    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="VTX kernels require CUDA"
    )
except ImportError:
    HAS_PYTEST = False
    # Minimal shim so the file can be run directly without pytest installed.
    class _PytestShim:
        class raises:
            def __init__(self, exc, match=None):
                self.exc = exc
                self.match = match
            def __enter__(self):
                return self
            def __exit__(self, etype, evalue, tb):
                if etype is None:
                    raise AssertionError(f"Expected {self.exc.__name__}, none raised")
                if not issubclass(etype, self.exc):
                    return False
                if self.match is not None and self.match not in str(evalue):
                    raise AssertionError(
                        f"Exception message {evalue!r} does not match {self.match!r}"
                    )
                return True
    pytest = _PytestShim()


def _build_vtx_cache(num_tokens: int, num_kv_heads: int, head_dim: int,
                    guard_pages: int = 1):
    """Simulate the VTXGraphCachePool storage layout.

    Shape: (num_tokens * num_kv_heads + guard_pages, page_size=1, head_dim)
    Position formula (page_size=1):  flat_idx = token_pos * num_kv_heads + head_id
    """
    num_pages = num_tokens * num_kv_heads + guard_pages
    return torch.zeros(num_pages, 1, head_dim, dtype=torch.bfloat16, device="cuda")


def test_vtx_layout_formula_roundtrip():
    """Fill each (token, head) slot with a unique value and read it back via
    the VTX layout formula. Confirms _vtx_to_std_layout's underlying assumption.
    """
    num_tokens, num_kv_heads, head_dim = 5, 8, 64
    cache = _build_vtx_cache(num_tokens, num_kv_heads, head_dim)

    for t in range(num_tokens):
        for h in range(num_kv_heads):
            flat = t * num_kv_heads + h
            cache[flat, 0, :] = float(t * 16 + h)

    # Read back: cache[t * num_kv_heads + h] should hold t*100 + h
    for t in range(num_tokens):
        for h in range(num_kv_heads):
            flat = t * num_kv_heads + h
            assert cache[flat, 0, 0].item() == float(t * 16 + h), \
                f"VTX layout broken at (t={t}, h={h})"


def test_vtx_to_std_layout_matches_flashinfer_expectation():
    """After _vtx_to_std_layout, cache[token_idx, 0, head_idx, :] should contain
    the data written for that (token, head). This is what FlashInfer expects
    when indexed by kv_indices = [token positions].
    """
    from sglang.srt.layers.attention.vtx_graph_backend import _vtx_to_std_layout

    num_tokens, num_kv_heads, head_dim = 7, 8, 64
    cache = _build_vtx_cache(num_tokens, num_kv_heads, head_dim, guard_pages=3)

    # Write unique marker per (t, h)
    for t in range(num_tokens):
        for h in range(num_kv_heads):
            flat = t * num_kv_heads + h
            cache[flat, 0, :] = float(t * 16 + h)

    std = _vtx_to_std_layout(cache, num_kv_heads, page_size=1, head_dim=head_dim)

    # Expected shape: (num_usable_tokens, 1, num_kv_heads, head_dim)
    assert std.dim() == 4
    assert std.shape[1] == 1
    assert std.shape[2] == num_kv_heads
    assert std.shape[3] == head_dim
    assert std.shape[0] >= num_tokens, (
        f"std layout dropped usable tokens: got {std.shape[0]}, expected >= {num_tokens}"
    )

    for t in range(num_tokens):
        for h in range(num_kv_heads):
            val = std[t, 0, h, 0].item()
            assert val == float(t * 16 + h), \
                f"std layout mismatch at (t={t}, h={h}): got {val}, expected {t*100+h}"


def test_vtx_to_std_layout_drops_guard_page():
    """_vtx_to_std_layout must drop trailing guard pages so shape[0] is
    divisible by num_kv_heads; otherwise view() would fail.
    """
    from sglang.srt.layers.attention.vtx_graph_backend import _vtx_to_std_layout

    num_tokens, num_kv_heads, head_dim = 4, 8, 32
    # Create a cache with an odd trailing fragment (3 extra pages, < num_kv_heads)
    cache = _build_vtx_cache(num_tokens, num_kv_heads, head_dim, guard_pages=3)
    assert cache.shape[0] % num_kv_heads != 0  # sanity: there IS a fragment

    std = _vtx_to_std_layout(cache, num_kv_heads, page_size=1, head_dim=head_dim)
    assert std.shape[0] * num_kv_heads == (cache.shape[0] // num_kv_heads) * num_kv_heads


def test_vtx_to_std_layout_requires_page_size_1():
    """Current implementation only supports page_size=1. Assert that the
    contract is enforced loudly (rather than silently producing wrong data).
    """
    from sglang.srt.layers.attention.vtx_graph_backend import _vtx_to_std_layout

    cache = torch.zeros(16, 2, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="page_size=1"):
        _vtx_to_std_layout(cache, num_kv_heads=8, page_size=2, head_dim=64)
