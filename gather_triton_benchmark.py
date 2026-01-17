# gather_triton_benchmark.py
import math
import time
import argparse

import torch

# ---- Triton imports (requires CUDA build + triton installed) ----
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_K": 512}, num_warps=8),
    ],
    key=["K"],
)
@triton.jit
def gather_cols_kernel(
    x_ptr,          # *fp16/bf16/fp32, shape (N, D) row-major
    idx_ptr,        # *int32, shape (K,)
    y_ptr,          # *same as x, shape (N, K) row-major
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    stride_xn: tl.constexpr,  # x stride for row (in elements)
    stride_yn: tl.constexpr,  # y stride for row (in elements)
    SCALE: tl.constexpr,      # compile-time scalar scale (or set to 1.0)
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)  # row index in [0, N)
    pid_k = tl.program_id(axis=1)  # block over K

    # offsets in K dimension
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < K

    cols = tl.load(idx_ptr + k, mask=k_mask, other=0).to(tl.int32)  # arbitrary indices in [0, D)
    x_ptrs = x_ptr + pid_n * stride_xn + cols
    vals = tl.load(x_ptrs, mask=k_mask, other=0.0)

    y_ptrs = y_ptr + pid_n * stride_yn + k
    tl.store(y_ptrs, vals * SCALE, mask=k_mask)


def gather_triton(x_2d: torch.Tensor, idx_i32: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    x_2d: (N, D) contiguous CUDA tensor
    idx_i32: (K,) contiguous CUDA int32 tensor
    returns y: (N, K)
    """
    assert x_2d.is_cuda and idx_i32.is_cuda
    assert x_2d.is_contiguous(), "make x contiguous before gather"
    assert idx_i32.dtype == torch.int32 and idx_i32.is_contiguous()
    N, D = x_2d.shape
    K = idx_i32.numel()
    y = torch.empty((N, K), device=x_2d.device, dtype=x_2d.dtype)

    grid = (N, triton.cdiv(K, 256))  # BLOCK_K autotuned anyway
    # SCALE is constexpr here; if you want runtime scale, pass it as a tensor or tl.multiple_of tricks.
    gather_cols_kernel[grid](
        x_2d, idx_i32, y,
        N=N, D=D, K=K,
        stride_xn=x_2d.stride(0),
        stride_yn=y.stride(0),
        SCALE=scale,
    )
    return y


def torch_gather(x_2d: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # Two common baselines:
    # 1) index_select along last dim:
    return torch.index_select(x_2d, dim=1, index=idx)
    # 2) advanced indexing (often similar):
    # return x_2d[:, idx]


@torch.no_grad()
def benchmark(fn, iters=200, warmup=50):
    # CUDA timing
    torch.cuda.synchronize()
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    B, T, D, K = args.B, args.T, args.D, args.K
    N = B * T

    # ---- Create (B,T,D) then flatten to (N,D) ----
    x = torch.randn((B, T, D), device="cuda", dtype=dtype)
    x2 = x.contiguous().view(-1, D)

    # user-chosen fixed set S (arbitrary). We'll sample unique indices for test.
    # Replace this with your actual fixed S.
    idx = torch.randperm(D, device="cuda")[:K].contiguous()
    idx_i32 = idx.to(torch.int32)

    # ---- Correctness check ----
    y_torch = torch_gather(x2, idx)
    y_triton = gather_triton(x2, idx_i32, scale=1.0)

    max_abs = (y_torch - y_triton).abs().max().item()
    mean_abs = (y_torch - y_triton).abs().mean().item()
    print(f"[check] max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

    # ---- Benchmark ----
    ms_torch = benchmark(lambda: torch_gather(x2, idx), iters=args.iters)
    ms_triton = benchmark(lambda: gather_triton(x2, idx_i32, scale=1.0), iters=args.iters)

    # Bytes moved: read N*K elements and write N*K elements (ignoring idx load + cache effects)
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype != torch.float32 else 4
    bytes_rw = 2 * N * K * bytes_per_elem
    gbps_torch = (bytes_rw / (ms_torch * 1e-3)) / 1e9
    gbps_triton = (bytes_rw / (ms_triton * 1e-3)) / 1e9

    print(f"[torch]  {ms_torch:.4f} ms  (~{gbps_torch:.1f} GB/s effective RW)")
    print(f"[triton] {ms_triton:.4f} ms  (~{gbps_triton:.1f} GB/s effective RW)")
    print(f"[speedup] {ms_torch / ms_triton:.2f}x")

    # Optional: check output shape
    y = y_triton.view(B, T, K)
    print("output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()