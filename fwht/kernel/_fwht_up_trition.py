import triton
import triton.language as tl
import math
import torch

from ._build_hadamard import build_H

def build_pos2k(idx: torch.Tensor, D: int) -> torch.Tensor:
    """
    idx: (K,) int32/int64 CUDA tensor, unique entries in [0, D)
    returns pos2k: (D,) int32 CUDA tensor with -1 default
    """
    assert idx.is_cuda
    idx_i64 = idx.to(torch.int64)
    K = idx.numel()
    pos2k = torch.full((D,), -1, device=idx.device, dtype=torch.int32)
    pos2k[idx_i64] = torch.arange(K, device=idx.device, dtype=torch.int32)
    return pos2k

@triton.jit
def fwht_256_2step_kernel_fp32acc(
    a: tl.tensor,
    base: tl.tensor,
    A_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr
):
    batch_size: tl.constexpr = A_SIZE // (BASE_SIZE ** 2)
    ar = tl.reshape(a, (batch_size, BASE_SIZE, BASE_SIZE))
    br = base[None, :, :]  # (1,S,S) broadcast in dot may not work as batched; keep as-is if your triton allows

    # NOTE: if your Triton doesn't support batched tl.dot, you'll need a 2D reshape rewrite.
    left = tl.dot(br.broadcast_to(batch_size, BASE_SIZE, BASE_SIZE), ar, out_dtype=tl.float32).to(a.dtype)
    out  = tl.dot(left, br.broadcast_to(batch_size, BASE_SIZE, BASE_SIZE), out_dtype=tl.float32).to(a.dtype)
    return tl.reshape(out, (A_SIZE,))


@triton.autotune(
    configs=[triton.Config(kwargs={}, num_warps=4)],
    key=["WORK_SIZE"],
)
@triton.jit
def fwht_256_kernel_up_fused(
    u_ptr,          # (N, K) input
    pos2k_ptr,      # (D,) int32, maps position->k or -1
    out_ptr,        # (N, WORK_SIZE) output
    scale,          # scalar
    K: tl.constexpr,
    D: tl.constexpr,         # e.g., 4096
    WORK_SIZE: tl.constexpr, # power-of-2 >= D; for your case D==WORK_SIZE==4096
    BASE_SIZE: tl.constexpr, # 16
    POWER_OF_2: tl.constexpr,
):
    tl.static_assert(WORK_SIZE >= 16)
    tl.static_assert(WORK_SIZE <= (2 ** 3) * (16 ** 3))
    tl.static_assert(WORK_SIZE % BASE_SIZE == 0)

    pid = tl.program_id(axis=0)  # token row id in [0, N)

    pos = tl.arange(0, WORK_SIZE)
    in_range = pos < D  # only positions < D are valid for pos2k

    # k = pos2k[pos] (or -1)
    k = tl.load(pos2k_ptr + pos, mask=in_range, other=-1).to(tl.int32)

    # safe pointer (avoid negative pointer arithmetic)
    k_safe = tl.maximum(k, 0)
    u_ptrs = u_ptr + pid * K + k_safe
    vals = tl.load(u_ptrs, mask=(k >= 0) & in_range, other=0.0)

    a = vals  # (WORK_SIZE,)

    # ---- Hadamard base (assumed +/-1) ----
    base = build_H(BASE_SIZE, a.dtype)

    BASE_SIZE_SQUARED: tl.constexpr = BASE_SIZE ** 2
    BASE_SIZE_CUBED: tl.constexpr = BASE_SIZE ** 3

    if BASE_SIZE_SQUARED <= WORK_SIZE:
        tl.static_assert(WORK_SIZE % BASE_SIZE_SQUARED == 0)
        a = fwht_256_2step_kernel_fp32acc(a, base, WORK_SIZE, BASE_SIZE)

    if BASE_SIZE_CUBED <= WORK_SIZE:
        tl.static_assert(WORK_SIZE % BASE_SIZE_CUBED == 0)
        BATCH_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE_CUBED
        mat = tl.reshape(a, (BATCH_SIZE, BASE_SIZE, BASE_SIZE_SQUARED))
        # fp32 acc then cast back
        mat = tl.dot(
            base[None, :, :].broadcast_to(BATCH_SIZE, BASE_SIZE, BASE_SIZE),
            mat,
            out_dtype=tl.float32
        ).to(a.dtype)
        a = tl.reshape(mat, (WORK_SIZE,))

    if WORK_SIZE < BASE_SIZE_SQUARED:
        INNER_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE
        ar = tl.reshape(a, (INNER_SIZE, BASE_SIZE))
        ar = tl.sum(ar[:, :, None] * base[None, :, :], axis=1)
        a = tl.reshape(ar, (WORK_SIZE,))

    if POWER_OF_2 > 1:
        H = build_H(POWER_OF_2, a.dtype)
        mat = tl.reshape(a, (POWER_OF_2, WORK_SIZE // POWER_OF_2))
        mat = tl.sum(H[:, :, None] * mat[None, :, :], axis=1)
        a = tl.reshape(mat, (WORK_SIZE,))

    # store full WORK_SIZE output
    out_ptrs = out_ptr + pid * WORK_SIZE + pos
    tl.store(out_ptrs, a * scale)

def fwht_up(u_btk: torch.Tensor, idx: torch.Tensor, D: int = 4096, normalized: bool = True):
    """
    u_btk: (B,T,K)
    idx:   (K,) fixed user-chosen indices in [0,D), unique
    returns x: (B,T,D) where x = (H_D / sqrt(D)) * u_full and u_full[:, idx] = u
    """
    assert u_btk.is_cuda and idx.is_cuda
    assert u_btk.shape[-1] == idx.numel()
    assert D & (D - 1) == 0

    B, T, K = u_btk.shape
    N = B * T

    u2 = u_btk.contiguous().view(N, K)

    # WORK_SIZE logic (keep same constraint as your fwht())
    work_size = D  # if you always use D=4096; otherwise next power-of-2 >= D
    assert work_size == D, "This fused kernel assumes WORK_SIZE == D for now (easy to generalize)."

    power_of_16 = 4096 if work_size >= 4096 else (256 if work_size >= 256 else 16)
    power_of_2 = work_size // power_of_16
    assert power_of_2 in (1, 2, 4, 8)

    pos2k = build_pos2k(idx, D=D)  # int32 CUDA

    out = torch.empty((N, work_size), device=u2.device, dtype=u2.dtype)

    scale = (1.0 / math.sqrt(D)) if normalized else 1.0

    grid = (N,)
    fwht_256_kernel_up_fused[grid](
        u2,
        pos2k,
        out,
        scale,
        K=K,
        D=D,
        WORK_SIZE=work_size,
        BASE_SIZE=16,
        POWER_OF_2=power_of_2,
    )
    return out.view(B, T, D)