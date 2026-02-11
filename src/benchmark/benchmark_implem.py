
import torch
from typing import Tuple

import src.module.mom_varlen.mom_fast as mom_fast
import src.module.mom_varlen.mom_mem_eff as mom_mem_eff
import src.module.mom_varlen.old.mom_varlen as mom_varlen_old
import src.module.mom_varlen.update_module as umv
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
    module_fast = mom_fast.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=umv.LinearAttentionVarlenModule, update_module_args=(True,)
    ).to(device=device, dtype=dtype)

    module_old = mom_varlen_old.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=mom_varlen_old.LinearAttentionVarlenModule, update_module_args=(True,)
    ).to(device=device, dtype=dtype)
    module_old.load_state_dict(module_fast.state_dict())

    module_old_ref = mom_varlen_old.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=mom_varlen_old.LinearAttentionVarlenModule, update_module_args=(False,)
    ).to(device=device, dtype=dtype)
    module_old_ref.load_state_dict(module_fast.state_dict())

    module_mem_eff = mom_mem_eff.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=mom_mem_eff.LinearAttentionVarlenModuleMem, update_module_args=(True,)
    ).to(device=device, dtype=dtype)
    module_mem_eff.load_state_dict(module_fast.state_dict())

    module_naive = naive_mom.MoMRef(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=naive_mom.LinearAttention
    ).to(device=device, dtype=dtype)
    module_naive.load_state_dict(module_fast.state_dict())

    module_ref = mom_fast.MoM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, update_module=umv.LinearAttentionVarlenModule, update_module_args=(False,)
    ).to(device=device, dtype=dtype)
    module_ref.load_state_dict(module_fast.state_dict())

    module_fast.eval()
    module_mem_eff.eval()
    module_naive.eval()
    module_ref.eval()

    return module_fast, module_mem_eff, module_naive, module_ref, module_old, module_old_ref


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
def test_mom_implem(
    mom_module_1: torch.nn.Module,
    mom_module_2: torch.nn.Module,
    name: str,
    device: torch.device,
    seq_len: int = 5,
    batch_size: int = 2,
    input_dim: int = 4,
    hidden_dim: int = 8,
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
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    out_varlen, _, _ = mom_module_1(X, M0)
    out_naive, _, _  = mom_module_2(X, M0)

    allclose = torch.allclose(out_varlen, out_naive, atol=atol, rtol=0.0)

    if not allclose:
        print(f"[FAIL] {name} | max_abs_diff={max_abs_diff(out_varlen, out_naive):.6e}")
    else:
        print(f"[OK] {name} | max_abs_diff={max_abs_diff(out_varlen, out_naive):.6e}")


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

    module_fast, module_mem_eff, module_naive, _, _, _ = make_modules(
        input_dim=input_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k, device=device, dtype=dtype
    )
    X, M0 = make_inputs(seq_len, batch_size, input_dim, hidden_dim, device, dtype=dtype)

    def run_fast():
        with torch.no_grad():
            module_fast(X, M0)
    def run_naive():
        with torch.no_grad():
            module_naive(X, M0)
    def run_mem_eff():
        with torch.no_grad():
            module_mem_eff(X, M0)

    

    if device.type == "cuda":
        fast_ms = cuda_time_ms(run_fast, warmup=warmup, iters=iters)
        naive_ms = cuda_time_ms(run_naive, warmup=warmup, iters=iters)
        mem_eff_ms = cuda_time_ms(run_mem_eff, warmup=warmup, iters=iters)
    else:
        raise NotImplementedError("Benchmarking is only implemented for CUDA devices")


    print("\n=== Benchmark ===")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Shapes: X=({seq_len},{batch_size},{input_dim}), hidden_dim={hidden_dim}, M={num_memories}, k={k}")
    print(f"Triton fast varlen: {fast_ms:.4f} ms/iter")
    print(f"Naive update : {naive_ms:.4f} ms/iter")
    print(f"Memory efficient varlen : {mem_eff_ms:.4f} ms/iter")

    return fast_ms, naive_ms, mem_eff_ms

def compare_modules(module_1, module_2, X, M0, name, atol=1e-5):
    out_1, _, _ = module_1(X, M0)
    out_2, _, _ = module_2(X, M0)

    allclose = torch.allclose(out_1, out_2, atol=atol, rtol=0.0)
    if not allclose:
        print(f"[FAIL] {name} | max_abs_diff={max_abs_diff(out_1, out_2):.6e}")
    else:
        print(f"[OK] {name} | max_abs_diff={max_abs_diff(out_1, out_2):.6e}")

if __name__ == "__main__":
    device = _get_device("cuda")
    _set_seed(0)

    module_fast, module_mem_eff, module_naive, module_ref, module_old, module_old_ref = make_modules(
        input_dim=64, hidden_dim=64, num_memories=8, k=2, device=device, dtype=torch.float32
    )

    print("=== Tests numériques ===")
    test_mom_implem(module_naive, module_old, name="naive vs old", device=device, seq_len=128, batch_size=8, input_dim=64, hidden_dim=64, dtype=torch.float32, atol=1e-4)
    test_mom_implem(module_naive, module_fast, name="naive vs fast", device=device, seq_len=128, batch_size=8, input_dim=64, hidden_dim=64, dtype=torch.float32, atol=1e-4)
    test_mom_implem(module_naive, module_ref, name="naive vs ref", device=device, seq_len=128, batch_size=8, input_dim=64, hidden_dim=64, dtype=torch.float32, atol=1e-4)
    test_mom_implem(module_mem_eff, module_naive, name="naive vs mem_eff", device=device, seq_len=128, batch_size=8, input_dim=64, hidden_dim=64, dtype=torch.float32, atol=1e-4)
    test_mom_implem(module_old_ref, module_naive, name="old_ref vs naive", device=device, seq_len=128, batch_size=8, input_dim=64, hidden_dim=64, dtype=torch.float32, atol=1e-4)
    test_mom_implem(module_old_ref, module_old, name="old_ref vs old", device=device, seq_len=128, batch_size=8, input_dim=64, hidden_dim=64, dtype=torch.float32, atol=1e-4)

    benchmark_varlen_triton_vs_naive(
        device=device,
        seq_len=128,
        batch_size=8,
        input_dim=256,
        hidden_dim=256,
        num_memories=8,
        k=2,
        dtype=torch.float16,
        warmup=10,
        iters=100,
    )