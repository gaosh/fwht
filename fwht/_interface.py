import torch
import math
from torch import nn

from fwht.kernel._fwht_triton import fwht
from fwht.kernel._fwht_up_trition import fwht_up

class HadamardTransformAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, inplace):
        ctx._scale = scale
        ctx._inplace = inplace
        return fwht(input, scale, inplace)
    
    @staticmethod
    def backward(ctx, grad_output):
        return fwht(grad_output, ctx._scale, ctx._inplace), None, None
    
def fast_hadamard_transform(input, scale=1.0, inplace=False):
    return HadamardTransformAutograd.apply(input, scale, inplace)

class FastHadamardTransform(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, a, scale=1.0):
        return fast_hadamard_transform(a, scale)


class UpProjectHadamardTransformAutograd(torch.autograd.Function):
    """
    Autograd wrapper for the up-projection:
      x = A u,  where A = H_D[:, S] / sqrt(D) (if normalized)

    Forward uses your customized fused-up kernel: fwht_up(u_btk, idx, D, normalized).
    Backward computes grad_u = A^T grad_output = (H_D[S,:]/sqrt(D)) grad_output,
    i.e., FWHT over last dim D, then gather idx.
    """
    @staticmethod
    def forward(ctx, u: torch.Tensor, idx: torch.Tensor, D: int, normalized: bool):
        assert u.is_cuda and idx.is_cuda
        assert u.shape[-1] == idx.numel(), f"u last dim {u.shape[-1]} != idx size {idx.numel()}"
        assert D & (D - 1) == 0, "D must be power-of-2"
        ctx.D = int(D)
        ctx.normalized = bool(normalized)
        ctx.save_for_backward(idx)

        # Forward: fused-up
        return fwht_up(u, idx, D=D, normalized=normalized)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (idx,) = ctx.saved_tensors
        D = ctx.D
        normalized = ctx.normalized

        # grad_output: (..., D)
        assert grad_output.shape[-1] == D
        orig = grad_output.shape[:-1]
        g2 = grad_output.contiguous().view(-1, D)

        # Compute z = (H g) / sqrt(D) if normalized else (H g)
        scale = (1.0 / math.sqrt(D)) if normalized else 1.0
        z = fwht(g2, scale=scale, inplace=True)  # (N, D)

        # Gather idx -> (N, K)
        # NOTE: for speed, replace this with your Triton gather kernel later.
        gu2 = torch.index_select(z, dim=1, index=idx.to(torch.int64))

        grad_u = gu2.view(*orig, idx.numel())
        return grad_u, None, None, None


def up_project_fast_hadamard_transform(u: torch.Tensor, idx: torch.Tensor, D: int = 4096, normalized: bool = True):
    """
    Functional API.
    """
    return UpProjectHadamardTransformAutograd.apply(u, idx, D, normalized)


class UpProjectFastHadamardTransform(nn.Module):
    """
    nn.Module wrapper.

    Stores idx as a buffer so it moves with .to(device) and is checkpointed.
    """
    def __init__(self, idx: torch.Tensor, D: int = 4096, normalized: bool = True):
        super().__init__()
        assert idx.dim() == 1, "idx must be 1D"
        self.D = int(D)
        self.normalized = bool(normalized)
        self.register_buffer("idx", idx.to(torch.int32).contiguous())

    def forward(self, u: torch.Tensor):
        return up_project_fast_hadamard_transform(u, self.idx, self.D, self.normalized)
