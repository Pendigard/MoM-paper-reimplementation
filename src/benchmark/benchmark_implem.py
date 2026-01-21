
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
    @brief Crée trois modules MoM avec les mêmes poids: une version triton, une version torch varlen et une version naive.
    @param input_dim: dimension d'entrée
    @param hidden_dim: dimension cachée
    @param num_memories: nombre de mémoires
    @param k: nombre de têtes
    @param device: torch.device
    @param dtype: type de données
    @return: tuple des trois modules (triton, torch varlen, naive)
    """
    module_triton = mom_hw_eff.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=mom_hw_eff.LinearAttentionVarlenModule(use_triton=True)
    ).to(device=device, dtype=dtype)

    module_ref = mom_hw_eff.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=mom_hw_eff.LinearAttentionVarlenModule(use_triton=False)
    ).to(device=device, dtype=dtype)
    module_ref.load_state_dict(module_triton.state_dict())

    module_naive = naive_mom.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=naive_mom.LinearAttention()
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
    @brief Comparaison numérique de l'implementation varlen triton et l'implementation varlen torch dans des modules MoM
    avec les mêmes poids.
    @param device: torch.device
    @param seq_len: longueur de la séquence
    @param batch_size: taille du batch
    @param input_dim: dimension d'entrée
    @param hidden_dim: dimension cachée
    @param num_memories: nombre de mémoires
    @param k: nombre de têtes
    @param dtype: type de données
    @param atol: tolérance absolue pour la comparaison
    """
    module_triton, module_ref, _ = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    out_triton = module_triton(X, M0)
    out_ref = module_ref(X, M0)

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
    @brief Comparaison numérique de l'implementation varlen et l'implementation naive dans des modules MoM
    avec les mêmes poids.
    @param device: torch.device
    @param seq_len: longueur de la séquence
    @param batch_size: taille du batch
    @param input_dim: dimension d'entrée
    @param hidden_dim: dimension cachée
    @param num_memories: nombre de mémoires
    @param k: nombre de têtes
    @param dtype: type de données
    @param atol: tolérance absolue pour la comparaison
    """
    _, module_ref, module_naive = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    out_varlen = module_ref(X, M0)
    out_naive = module_naive(X, M0)

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
    @brief Benchmark de l'implementation varlen triton vs l'implementation naive dans des modules MoM avec les mêmes poids.
    @param device: torch.device
    @param seq_len: longueur de la séquence
    @param batch_size: taille du batch
    @param input_dim: dimension d'entrée
    @param hidden_dim: dimension cachée
    @param num_memories: nombre de mémoires
    @param k: nombre de têtes
    @param dtype: type de données
    @param warmup: nombre d'itérations de warmup
    @param iters: nombre d'itérations de benchmark
    @return: tuple des temps moyens (triton_ms, naive_ms) et du speedup
    """
    if device.type == "cuda" and dtype == torch.float32:
        # ok, but fp16/bf16 is usually more realistic
        pass

    module_triton, module_ref, module_naive = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    def run_triton():
        return module_triton(X, M0)

    def run_naive():
        return module_naive(X, M0)

    def run_ref():
        return module_ref(X, M0)

    

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
    print(f"Torch varlen : {ref_ms:.4f} ms/iter\n")

    return triton_ms, naive_ms, speedup

if __name__ == "__main__":
    device = _get_device("cuda")
    _set_seed(0)

    print("=== Tests numériques ===")
    test_triton_vs_torch_varlen(device=device, dtype=torch.float32, atol=1e-5)
    test_naive_vs_varlen(device=device, dtype=torch.float32, atol=1e-5)

    benchmark_varlen_triton_vs_naive(
        device=device,
        seq_len=128,
        batch_size=8,
        input_dim=256,
        hidden_dim=64,
        num_memories=8,
        k=2,
        dtype=torch.float16,
        warmup=10,
        iters=100,
    )