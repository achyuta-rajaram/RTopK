import torch
import rtopk_cuda


class fastRTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int, max_iter: int = 10000, precision: float = 0.0):
        vals, idx = rtopk_cuda.rtopk(x, k=k, max_iter=max_iter, precision=precision)
        idx = idx.to(torch.long)
        ctx.save_for_backward(idx)
        ctx.input_shape = x.shape
        return vals, idx

    @staticmethod
    def backward(ctx, grad_vals, grad_idx):  # type: ignore[override]
        del grad_idx
        (idx,) = ctx.saved_tensors
        N, D = ctx.input_shape

        if grad_vals is None:
            return torch.zeros((N, D), device=idx.device, dtype=torch.float32), None, None, None

        grad_in = torch.zeros((N, D), device=grad_vals.device, dtype=grad_vals.dtype)
        grad_in.scatter_add_(1, idx.long(), grad_vals)
        return grad_in, None, None, None


def fast_rtopk(x: torch.Tensor, k: int, max_iter: int = 10000, precision: float = 0.0):
    return fastRTopK.apply(x, k, max_iter, precision)


