import math
import torch
import triton
import triton.language as tl
from fwht.kernel._fwht_triton import fwht as fwht_fn

import argparse

# ============================================================
# Triton scatter: u (N,K) -> out (N,D) with out[:, idx[k]] = u[:, k]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64},  num_warps=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_K": 512}, num_warps=8),
    ],
    key=["K"],
)
@triton.jit
def scatter_cols_kernel(
    u_ptr,          # (N, K)
    idx_ptr,        # (K,) int32
    out_ptr,        # (N, D)
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    stride_un: tl.constexpr,   # u.stride(0) in elements
    stride_on: tl.constexpr,   # out.stride(0) in elements
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < K

    cols = tl.load(idx_ptr + k, mask=k_mask, other=0).to(tl.int32)  # in [0, D)
    vals = tl.load(u_ptr + pid_n * stride_un + k, mask=k_mask, other=0.0)

    out_ptrs = out_ptr + pid_n * stride_on + cols
    tl.store(out_ptrs, vals, mask=k_mask)


def scatter_cols_triton(u_2d: torch.Tensor, idx_i32: torch.Tensor, D: int) -> torch.Tensor:
    """
    u_2d: (N, K) contiguous CUDA tensor
    idx_i32: (K,) contiguous CUDA int32 tensor with unique entries in [0, D)
    returns out: (N, D) with out[:, idx[k]] = u[:, k]
    """
    assert u_2d.is_cuda and idx_i32.is_cuda
    assert u_2d.is_contiguous(), "u_2d must be contiguous"
    assert idx_i32.dtype == torch.int32 and idx_i32.is_contiguous()
    N, K = u_2d.shape

    out = u_2d.new_zeros((N, D))
    grid = (N, triton.cdiv(K, 256))
    scatter_cols_kernel[grid](
        u_2d, idx_i32, out,
        N=N, D=D, K=K,
        stride_un=u_2d.stride(0),
        stride_on=out.stride(0),
    )
    return out


# ============================================================
# Up direction: x = (1/sqrt(D)) * H_D * u_full, u_full[:, idx] = u
# ============================================================
def up_project_fwht_triton_scatter(
    u_btk: torch.Tensor,
    idx: torch.Tensor,
    D: int,
    normalized: bool = True,
) -> torch.Tensor:
    assert u_btk.is_cuda and idx.is_cuda
    assert u_btk.shape[-1] == idx.numel()
    assert D & (D - 1) == 0, "D must be power of 2"
    B, T, K = u_btk.shape
    N = B * T

    u2 = u_btk.contiguous().view(N, K)
    idx_i32 = idx.to(torch.int32).contiguous()

    u_full = scatter_cols_triton(u2, idx_i32, D=D)  # (N, D)

    scale = 1.0 / math.sqrt(D) if normalized else 1.0
    x_full = fwht(u_full, scale=scale, inplace=True)  # (N, D)
    return x_full.view(B, T, D)


# ============================================================
# Baseline #2: build u_full with torch.index_add (K -> D), then FWHT
# ============================================================
def up_project_fwht_index_add(
    u_btk: torch.Tensor,
    idx: torch.Tensor,
    D: int,
    normalized: bool = True,
) -> torch.Tensor:
    assert u_btk.is_cuda and idx.is_cuda
    B, T, K = u_btk.shape
    N = B * T
    u2 = u_btk.contiguous().view(N, K)

    u_full = torch.zeros((N, D), device=u_btk.device, dtype=u_btk.dtype)
    u_full.index_add_(dim=1, index=idx.to(torch.int64), source=u2)

    scale = 1.0 / math.sqrt(D) if normalized else 1.0
    x_full = fwht(u_full, scale=scale, inplace=True)
    return x_full.view(B, T, D)


# ============================================================
# Baseline #3: Regular dense matrix multiplication u @ W
# ============================================================
@torch.no_grad()
def dense_mm_baseline(
    u_btk: torch.Tensor,   # (B, T, K)
    W: torch.Tensor,       # (K, D)
) -> torch.Tensor:
    """
    Regular dense MM baseline: x = u @ W.
    This is NOT expected to match the FWHT-based operator (different linear map),
    but is useful for performance comparison / sanity on shapes.
    """
    B, T, K = u_btk.shape
    K2, D = W.shape
    assert K == K2
    u2 = u_btk.contiguous().view(-1, K)
    x2 = (u2.to(torch.float32) @ W.to(torch.float32)).to(u_btk.dtype)  # fp32 accumulate
    return x2.view(B, T, D)


def benchmark(fn, iters=200, warmup=50):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="Run timing benchmarks")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(args.seed)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = "cuda"

    B, T, K, D = args.B, args.T, args.K, args.D
    assert D & (D - 1) == 0, "D must be power-of-2 for FWHT"
    assert K <= D, "K must be <= D for index set into [0,D)"

    # User-chosen fixed set S: for test we sample unique indices.
    # Replace idx with your fixed S.
    idx = torch.randperm(D, device=device)[:K].contiguous()

    u = torch.randn((B, T, K), device=device, dtype=dtype)

    # 1) Triton scatter + FWHT (normalized)
    x_triton = up_project_fwht_triton_scatter(u, idx, D=D, normalized=True)

    # 2) torch.index_add + FWHT (normalized) - should match (same operator)
    x_index_add = up_project_fwht_index_add(u, idx, D=D, normalized=True)

    diff = (x_triton - x_index_add).abs()
    print("[check] Triton-scatter+FWHT vs index_add+FWHT (should match):")
    print(f"  max_abs  = {diff.max().item():.3e}")
    print(f"  mean_abs = {diff.mean().item():.3e}")
    print("  shape:", tuple(x_triton.shape))

    # 3) Regular dense MM baseline (different operator; only shape check)
    W = torch.randn((K, D), device=device, dtype=dtype)
    x_dense = dense_mm_baseline(u, W)
    print("[dense mm] u @ W shape:", tuple(x_dense.shape))

    if args.bench:
        ms_triton = benchmark(lambda: up_project_fwht_triton_scatter(u, idx, D=D, normalized=True),
                             iters=args.iters)
        ms_indexadd = benchmark(lambda: up_project_fwht_index_add(u, idx, D=D, normalized=True),
                               iters=args.iters)
        ms_dense = benchmark(lambda: dense_mm_baseline(u, W), iters=args.iters)

        N = B * T
        print("\n[timing] (avg over iters)")
        print(f"  Triton scatter + FWHT : {ms_triton:.4f} ms   (N={N}, K={K}, D={D})")
        print(f"  index_add + FWHT      : {ms_indexadd:.4f} ms")
        print(f"  dense mm (u@W)        : {ms_dense:.4f} ms")


if __name__ == "__main__":
    main()

