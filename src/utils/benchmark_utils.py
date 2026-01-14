import time
from typing import Callable
import torch

def cuda_time_ms(fn: Callable[[], torch.Tensor], warmup: int = 50, iters: int = 1000) -> float:
    """
    @brief Chronomètre une fonction sur GPU en millisecondes.
    @param fn: Fonction à chronométrer. Doit retourner un tenseur torch.Tensor.
    @param warmup: Nombre d'itérations de warm-up avant le chronométrage.
    @param iters: Nombre d'itérations à chronométrer.
    @return: Temps moyen par itération en millisecondes.
    """
    assert torch.cuda.is_available(), "CUDA n'est pas disponible pour le timing GPU."

    # Warm-up
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def cpu_time_ms(fn: Callable[[], torch.Tensor], warmup: int = 10, iters: int = 200) -> float:
    """
    @brief Chronomètre une fonction sur CPU en millisecondes.
    @param fn: Fonction à chronométrer. Doit retourner un tenseur torch.Tensor.
    @param warmup: Nombre d'itérations de warm-up avant le chronométrage.
    @param iters: Nombre d'itérations à chronométrer.
    @return: Temps moyen par itération en millisecondes.
    """
    for _ in range(warmup):
        _ = fn()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters