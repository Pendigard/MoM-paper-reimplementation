import triton
import triton.language as tl
import torch
import time

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * N
    offsets = block_start + tl.arange(0, N)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Input tensors must have the same shape"
    out = torch.empty_like(x)
    N = x.numel()
    grid = (triton.cdiv(N, 1024),)
    add_kernel[grid](x.flatten(), y.flatten(), out.flatten(), N)
    return out

if __name__ == "__main__":
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    start = time.time()
    c = triton_add(a, b)
    end = time.time()
    print(f"Triton addition took {end - start:.6f} seconds")
    print(c)

    start = time.time()
    c = triton_add(a, b)
    end = time.time()
    print(f"Triton addition took {end - start:.6f} seconds after warm-up")


    start = time.time()
    d = a + b
    end = time.time()
    print(f"PyTorch addition took {end - start:.6f} seconds")


    start = time.time()
    d = a + b
    end = time.time()
    print(f"PyTorch addition took {end - start:.6f} seconds after warm-up")
