import torch
import torch.nn as nn
from src.module.mom_varlen import MoM




class MoMPipeline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, embedding_dim: int, num_memories: int, k: int, output_activation: nn.Module = nn.Identity(), *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.mom = MoM(input_dim=embedding_dim, hidden_dim=hidden_dim, num_memories=num_memories, k=k)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        M_0 = torch.zeros(x.shape[1], self.mom.num_memories + 1, self.mom.hidden_dim, self.mom.hidden_dim, device=x.device)
        mom_out, _ = self.mom(x_emb, M_0)
        out = self.output_layer(mom_out)
        out = self.output_activation(out)
        return out