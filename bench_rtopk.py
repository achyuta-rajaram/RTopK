import os
import math
import time
import torch

try:
    import triton
    from triton import testing as triton_testing
except Exception as e:
    raise RuntimeError("Please install Triton to use triton.testing.do_bench: pip install triton") from e

from fast_rtopk import fast_rtopk


def check_correctness(x: torch.Tensor, k: int, *, atol: float = 0.0, rtol: float = 0.0) -> bool:
    vals_ext, idx_ext = fast_rtopk(x, k)
    vals_ref, idx_ref = torch.topk(x, k, dim=1, largest=True, sorted=False)

    # Compare sets of indices per row (order-insensitive)
    idx_ext_sorted, _ = torch.sort(idx_ext, dim=1)
    idx_ref_sorted, _ = torch.sort(idx_ref, dim=1)
    same_indices = torch.equal(idx_ext_sorted, idx_ref_sorted)

    # Optional: check values gathered by indices (order-insensitive)
    vals_ext_sorted, _ = torch.sort(vals_ext, dim=1, descending=True)
    vals_ref_sorted, _ = torch.sort(vals_ref, dim=1, descending=True)
    same_values = torch.allclose(vals_ext_sorted, vals_ref_sorted, atol=atol, rtol=rtol)

    return bool(same_indices and same_values)


def bench_once(fn):
    return triton_testing.do_bench(fn)


def check_backward(N: int, D: int, k: int) -> bool:
    x1 = torch.rand((N, D), device="cuda", dtype=torch.float32, requires_grad=True)
    v1, i1 = fast_rtopk(x1, k)
    loss1 = v1.sum()
    loss1.backward()
    g1 = x1.grad.detach().clone()

    x2 = x1.detach().clone().requires_grad_(True)
    v2, i2 = torch.topk(x2, k, dim=1, largest=True, sorted=False)
    loss2 = v2.sum()
    loss2.backward()
    g2 = x2.grad.detach().clone()

    return torch.equal(g1, g2)


def main():
    torch.manual_seed(1234)
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")

    N = 65536*64
    D = 256
    ks = [8, 16, 32, 64, 128]

    x = torch.rand((N, D), device=device, dtype=torch.float32)

    print(f"Input: N={N}, D={D}; testing ks={ks}")

    for k in ks:
        print(f"\n== k={k} ==")

        # Warmup
        fast_rtopk(x, k)
        torch.topk(x, k, dim=1, largest=True, sorted=False)
        torch.cuda.synchronize()

        # Correctness
        ok = check_correctness(x, k)
        print(f"correctness: {'OK' if ok else 'MISMATCH'}")

        # Speed (ms)
        t_ext_ms = bench_once(lambda: fast_rtopk(x, k))
        t_ref_ms = bench_once(lambda: torch.topk(x, k, dim=1, largest=True, sorted=False))

        speedup = t_ref_ms / t_ext_ms if t_ext_ms > 0 else float('inf')
        print(f"rtopk_cuda.rtopk: {t_ext_ms:.3f} ms")
        print(f"torch.topk     : {t_ref_ms:.3f} ms")
        print(f"speedup        : {speedup:.2f}x")

        # Backward check on a smaller problem to avoid OOM
        N_grad = min(4096, N)
        ok_bwd = check_backward(N_grad, D, ks[-1])
        print(f"\nbackward (k={k}) correctness: {'OK' if ok_bwd else 'MISMATCH'}")


if __name__ == "__main__":
    main()


