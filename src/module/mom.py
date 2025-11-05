import torch
import torch.nn as nn


class LinearAttention(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, M_t : torch.Tensor, k : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
        """
        @brief Met à jour la mémoire M_t avec la clé k et la valeur v en utilisant une attention linéaire.
        @param M_t: Mémoire actuelle de forme (batch_size, hidden_dim, hidden_dim)
        @param k: Clé de forme (batch_size, hidden_dim)
        @param v: Valeur de forme (batch_size, hidden_dim)
        @return: Mémoire mise à jour de forme (batch_size, hidden_dim, hidden_dim)
        """
        # unsqueeze(-1) met k en vecteur colonne
        # unsqueeze(-2) met v en vecteur ligne
        return M_t + k.unsqueeze(-1) @ v.unsqueeze(-2)


class Memory(nn.Module):

    def __init__(self, input_dim : int, hidden_dim : int, update_module : nn.Module = LinearAttention(), *args, **kwargs):
        """
        @brief Module de mémoire qui utilise une clé et une valeur pour mettre à jour la mémoire.
        @param input_dim: Dimension de l'entrée x.
        @param hidden_dim: Dimension de la mémoire M_t.
        @param update_module: Module de mise à jour de la mémoire.
        """
        super().__init__(*args, **kwargs)

        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

        self.update_module = update_module

    def forward(self, M_t : torch.Tensor, x : torch.Tensor) -> torch.Tensor:
        """
        @brief Met à jour la mémoire M_t avec l'entrée x.
        @param M_t: Mémoire actuelle de forme (batch_size, hidden_dim, hidden_dim)
        @param x: Entrée de forme (batch_size, input_dim)
        @return: Mémoire mise à jour de forme (batch_size, hidden_dim, hidden_dim)
        """
        k = self.W_k(x)
        v = self.W_v(x)
        return self.update_module(M_t, k, v)
    


if __name__ == "__main__":
    batch_size = 32
    module = Memory(10, 10)
    X = torch.randn(32, 10)
    M_t = torch.zeros(32, 10, 10)
    print(X.shape)
    print(module(M_t, X).shape)
