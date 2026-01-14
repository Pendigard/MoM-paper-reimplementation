
import torch
from typing import Tuple

import src.module.mom_varlen as mom_hw_eff
import src.module.naive_mom as naive_mom

from src.utils.torch_utils import _get_device, _set_seed, assert_allclose, max_abs_diff, make_inputs
from src.utils.benchmark_utils import cuda_time_ms, cpu_time_ms

def make_modules(
    input_dim: int,
    hidden_dim: int,
    num_memories: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Creates:
      - module_triton: mom_hw_eff.MoM
      - module_ref   : mom_hw_eff.MoM (same weights as module_triton)
      - module_naive : naive_mom.MoM (same weights loaded from module_triton)
    """
    module_triton = mom_hw_eff.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k
    ).to(device=device, dtype=dtype)

    module_ref = mom_hw_eff.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k
    ).to(device=device, dtype=dtype)
    module_ref.load_state_dict(module_triton.state_dict())

    module_naive = naive_mom.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k
    ).to(device=device, dtype=dtype)
    module_naive.load_state_dict(module_triton.state_dict())

    module_triton.eval()
    module_ref.eval()
    module_naive.eval()

    return module_triton, module_ref, module_naive


@torch.no_grad()
def test_triton_vs_torch_varlen(
    device: torch.device,
    seq_len: int = 5,
    batch_size: int = 2,
    input_dim: int = 4,
    hidden_dim: int = 8,
    num_memories: int = 3,
    k: int = 2,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
):
    """
    Compares mom_hw_eff.linear_attn_varlen_triton vs mom_hw_eff.linear_attn_varlen
    inside the SAME MoM module (same weights).
    """
    module_triton, module_ref, _ = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    out_triton = module_triton(X, M0, varlen_update=mom_hw_eff.linear_attn_varlen_triton)
    out_ref = module_ref(X, M0, varlen_update=mom_hw_eff.linear_attn_varlen)

    assert_allclose(out_triton, out_ref, atol=atol, rtol=0.0, name="triton_vs_torch_varlen")

    print(f"[OK] Triton varlen vs Torch varlen | max_abs_diff={max_abs_diff(out_triton, out_ref):.6e}")


@torch.no_grad()
def test_naive_vs_varlen(
    device: torch.device,
    seq_len: int = 5,
    batch_size: int = 2,
    input_dim: int = 4,
    hidden_dim: int = 8,
    num_memories: int = 3,
    k: int = 2,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
):
    """
    Compares naive_mom.MoM (naive update) vs mom_hw_eff.MoM (varlen path using Triton).
    """
    _, module_ref, module_naive = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    out_varlen = module_ref(X, M0, varlen_update=mom_hw_eff.linear_attn_varlen_triton)
    out_naive = module_naive(X, M0, update_function=naive_mom.linear_attn)

    assert_allclose(out_varlen, out_naive, atol=atol, rtol=0.0, name="naive_vs_varlen")

    print(f"[OK] Naive vs Varlen(Triton) | max_abs_diff={max_abs_diff(out_varlen, out_naive):.6e}")


@torch.no_grad()
def benchmark_varlen_triton_vs_naive(
    device: torch.device,
    seq_len: int = 128,
    batch_size: int = 8,
    input_dim: int = 256,
    hidden_dim: int = 64,
    num_memories: int = 8,
    k: int = 2,
    dtype: torch.dtype = torch.float16,
    warmup: int = 100,
    iters: int = 1000,
):
    """
    Benchmarks:
      - mom_hw_eff.MoM with Triton varlen update
      - naive_mom.MoM with naive update
    using fixed inputs (no allocations in loop).
    """
    if device.type == "cuda" and dtype == torch.float32:
        # ok, but fp16/bf16 is usually more realistic
        pass

    module_triton, module_ref, module_naive = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    def run_triton():
        return module_triton(X, M0, varlen_update=mom_hw_eff.linear_attn_varlen_triton)

    def run_naive():
        return module_naive(X, M0, update_function=naive_mom.linear_attn)

    def run_ref():
        return module_ref(X, M0, varlen_update=mom_hw_eff.linear_attn_varlen)

    

    if device.type == "cuda":
        triton_ms = cuda_time_ms(run_triton, warmup=warmup, iters=iters)
        ref_ms = cuda_time_ms(run_ref, warmup=warmup, iters=iters)
        naive_ms = cuda_time_ms(run_naive, warmup=warmup, iters=iters)
    else:
        triton_ms = cpu_time_ms(run_triton, warmup=max(5, warmup // 10), iters=max(50, iters // 10))
        ref_ms = cpu_time_ms(run_ref, warmup=max(5, warmup // 10), iters=max(50, iters // 10))
        naive_ms = cpu_time_ms(run_naive, warmup=max(5, warmup // 10), iters=max(50, iters // 10))

    speedup = naive_ms / triton_ms if triton_ms > 0 else float("inf")

    print("\n=== Benchmark ===")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Shapes: X=({seq_len},{batch_size},{input_dim}), hidden_dim={hidden_dim}, M={num_memories}, k={k}")
    print(f"Triton varlen: {triton_ms:.4f} ms/iter")
    print(f"Naive update : {naive_ms:.4f} ms/iter")
    print(f"Speedup     : {speedup:.2f}x")
    print("--- Reference Torch varlen implementation ---")
    print(f"Torch varlen : {ref_ms:.4f} ms/iter")

    return triton_ms, naive_ms, speedup

if __name__ == "__main__":
    device = _get_device("cuda")
    _set_seed(0)

    # Numeric tests (small)
    print("=== Tests numériques ===")
    test_triton_vs_torch_varlen(device=device, dtype=torch.float32, atol=1e-5)
    test_naive_vs_varlen(device=device, dtype=torch.float32, atol=1e-5)

    # Benchmark (bigger, more realistic)
    benchmark_varlen_triton_vs_naive(
        device=device,
        seq_len=128,
        batch_size=8,
        input_dim=256,
        hidden_dim=64,
        num_memories=8,
        k=2,
        dtype=torch.float16,   # typical for speed
        warmup=10,
        iters=100,
    )