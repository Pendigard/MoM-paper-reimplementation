from typing import Callable, Dict, List, Tuple
import torch
import matplotlib.pyplot as plt

from src.module.mom_llm import MoMLLM
from src.module.llm import TransformerLLM
import src.module.mom_varlen.update_module as umv
import src.module.mom_varlen.mom_fast as mom_varlen
import src.module.naive_mom as naive_mom
import torch.nn as nn
from src.utils.benchmark_utils import cuda_time_and_memory
import numpy as np


CONFIG = {
    "vocab_size": 32000,
    "dim": 256,
    "num_layers": 4,
    "num_memories": 4,
    "top_k": 2,
    "batch_size": 2,
    "num_heads": 8,
    "warmup": 100,
    "iters": 100,
    "context_lengths": [1024, 2048, 4096, 8192],
    "update_module": umv.LinearAttentionVarlenModule(use_triton=True, no_grad=True) # umv.LinearAttentionVarlenModule(use_triton=True, no_grad=True), # 
}

def causal_mask_triu_bool(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


def plot_series(
    xs: List[int],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    outpath: str,
    color_map: str = "Purples"
) -> None:

    plt.figure()

    # Génère des couleurs violettes bien espacées
    cmap = plt.cm.get_cmap(color_map)
    colors = cmap(np.linspace(0.3, 0.9, len(series)))  # évite trop clair/foncé

    for (name, ys), color in zip(series.items(), colors):
        plt.plot(
            xs,
            ys,
            marker="o",
            label=name,
            color=color,
            linewidth=2
        )

    plt.xlabel("Context Length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath)
    plt.show()


@torch.no_grad()
def benchmark_model(
    name: str,
    forward_fn_factory: Callable[[torch.Tensor, torch.Tensor], Callable[[], torch.Tensor]],
    vocab_size: int,
    batch_size: int,
    context_lengths: List[int],
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[List[float], List[float]]:
    times_ms: List[float] = []
    mem_allocs: List[float] = []
    

    for T in context_lengths:
        begin = torch.cuda.memory_allocated(device=device) / (1024**2)
        input_ids = torch.randint(0, vocab_size, (batch_size, T), device=device)
        end = torch.cuda.memory_allocated(device=device) / (1024**2)
        print(f"Input IDs allocation for T={T}: {end - begin:.2f} MB")
        mask = causal_mask_triu_bool(T, device)

        fn = forward_fn_factory(input_ids, mask)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t_ms, mem = cuda_time_and_memory(fn, warmup=warmup, iters=iters)
        times_ms.append(t_ms)
        mem_allocs.append(mem / (1024**2))

        print(f"T={T:5d} | {name:12s} | {t_ms:8.2f} ms | {mem / (1024**2):8.2f} MB")

        del input_ids, mask, fn
        torch.cuda.synchronize()

    return times_ms, mem_allocs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    begin_mem = torch.cuda.memory_allocated(device=device) / (1024**2)

    transformerLLM = TransformerLLM(
        vocab_size=CONFIG["vocab_size"],
        hidden_dim=CONFIG["dim"] * CONFIG["num_memories"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
    ).to(device).eval()

    momLLM = MoMLLM(
        vocab_size=CONFIG["vocab_size"],
        hidden_dim=CONFIG["dim"],
        num_memories=CONFIG["num_memories"],
        k=CONFIG["top_k"],
        num_layers=CONFIG["num_layers"],
        mom_implem=mom_varlen.MoM,
        layer_norm=nn.LayerNorm,
        update_module=CONFIG["update_module"],
    ).to(device).eval()

    naive_momLLM = MoMLLM(
        vocab_size=CONFIG["vocab_size"],
        hidden_dim=CONFIG["dim"],
        num_memories=CONFIG["num_memories"],
        k=CONFIG["top_k"],
        num_layers=CONFIG["num_layers"],
        mom_implem=naive_mom.MoM,
        layer_norm=nn.LayerNorm,
        update_module=naive_mom.LinearAttention(),
    ).to(device).eval()

    end_mem = torch.cuda.memory_allocated(device=device) / (1024**2)
    print(f"Model allocation: {end_mem - begin_mem:.2f} MB")

    context_lengths = CONFIG["context_lengths"]

    def transformer_factory(input_ids: torch.Tensor, mask: torch.Tensor):
        def _fn():
            return transformerLLM(input_ids, causal_mask=mask, materialize_output=False)
        return _fn

    def mom_factory(input_ids: torch.Tensor, _mask: torch.Tensor):
        def _fn():
            return momLLM(input_ids, materialize_output=False)[0]
        return _fn

    def naive_mom_factory(input_ids: torch.Tensor, _mask: torch.Tensor):
        def _fn():
            return naive_momLLM(input_ids)[0]
        return _fn

    # times_naive_mom, mem_naive_mom = benchmark_model(
    #     "NaiveMoMLLM",
    #     naive_mom_factory,
    #     CONFIG["vocab_size"],
    #     CONFIG["batch_size"],
    #     context_lengths,
    #     device,
    #     CONFIG["warmup"],
    #     CONFIG["iters"],
    # )

    times_mom, mem_mom = benchmark_model(
        "MoMLLM",
        mom_factory,
        CONFIG["vocab_size"],
        CONFIG["batch_size"],
        context_lengths,
        device,
        CONFIG["warmup"],
        CONFIG["iters"],
    )
    
    times_transformer, mem_transformer = benchmark_model(
        "TransformerLLM",
        transformer_factory,
        CONFIG["vocab_size"],
        CONFIG["batch_size"],
        context_lengths,
        device,
        CONFIG["warmup"],
        CONFIG["iters"],
    )

    plot_series(
        context_lengths,
        {"TransformerLLM": times_transformer, "MoMLLM": times_mom},
        "Forward Pass Time vs Context Length",
        "Time per Forward Pass (ms)",
        "forward_pass_time_comparison.png",
    )

    plot_series(
        context_lengths,
        {"TransformerLLM": mem_transformer, "MoMLLM": mem_mom},
        "Peak Memory Usage vs Context Length",
        "Peak Memory Usage (MB)",
        "peak_memory_usage_comparison.png",
    )