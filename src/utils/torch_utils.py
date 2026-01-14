import torch
from typing import Optional, Tuple

def _get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    @brief Calcule la différence absolue maximale entre deux tenseurs.
    @param a: Premier tenseur.
    @param b: Deuxième tenseur.
    @return: La différence absolue maximale.
    """
    return (a - b).abs().max().item()


@torch.no_grad()
def assert_allclose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5, rtol: float = 0.0, name: str = "") -> None:
    """
    @brief Vérifie que deux tenseurs sont proches l'un de l'autre.
    @param a: Premier tenseur.
    @param b: Deuxième tenseur.
    @param atol: Tolérance absolue.
    @param rtol: Tolérance relative.
    @param name: Nom du test (utilisé dans le message d'erreur).
    @raise AssertionError: Si les tenseurs ne sont pas proches.
    """
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = max_abs_diff(a, b)
        raise AssertionError(f"[{name}] Output différents! max_abs_diff={diff:.6e}, atol={atol}, rtol={rtol}")

def make_inputs(
    seq_len: int,
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    @brief Crée des tenseurs d'entrée X et M0 pour les tests de MoM.
    @param seq_len: Longueur de la séquence.
    @param batch_size: Taille du batch.
    @param input_dim: Dimension de l'entrée.
    @param hidden_dim: Dimension cachée des mémoires.
    @param device: Appareil sur lequel créer les tenseurs.
    @param dtype: Type de données des tenseurs.
    @return: Un tuple (X, M0) où X est de forme (seq_len, batch_size, input_dim) et M0 de forme (hidden_dim, hidden_dim).
    """
    X = torch.randn(seq_len, batch_size, input_dim, device=device, dtype=dtype)
    M0 = torch.zeros(hidden_dim, hidden_dim, device=device, dtype=dtype)
    return X, M0


