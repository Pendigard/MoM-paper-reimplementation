import torch
import torch.nn as nn
import src.module.mom_varlen as mom_varlen
import src.module.naive_mom as naive_mom
from typing import Callable

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: nn.Module = nn.ReLU(), *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MoMLLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_memories: int, k: int, num_layers: int, mom_implem = naive_mom.MoM, layer_norm = nn.LayerNorm, update_module : nn.Module = None, *args, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_memories = num_memories
        self.k = k
        self.num_layers = num_layers
        self.mom_implem = mom_implem

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        
        self.layers = nn.ModuleList([
            mom_implem(
                input_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                num_memories=num_memories, 
                k=k,
                update_module=update_module or naive_mom.LinearAttention()
            ) for _ in range(num_layers)
        ])

        self.MLPs = nn.ModuleList([
            MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 4,
                output_dim=hidden_dim,
                activation=nn.GELU()
            ) for _ in range(num_layers)
        ])
        
        self.norms_1 = nn.ModuleList([
            layer_norm(hidden_dim) for _ in range(num_layers)
        ])
        self.norms_2 = nn.ModuleList([
            layer_norm(hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)
        # self.output_layer.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(0, 1)
        
        M = torch.zeros(
            self.hidden_dim,
            self.hidden_dim,
            device=x.device
        )

        for i, layer in enumerate(self.layers):
            x = x + layer(self.norms_1[i](x), M.clone())
            x = x + self.MLPs[i](self.norms_2[i](x).transpose(0, 1)).transpose(0, 1)
        x = x.transpose(0, 1)

        logits = self.output_layer(x)
        return logits
    

if __name__ == "__main__":
    model = MoMLLM(
        vocab_size=200,
        hidden_dim=64,
        num_memories=4,
        k=2,
        num_layers=2,
        mom_implem=naive_mom.MoM,
        layer_norm=nn.LayerNorm,
        update_module=naive_mom.LinearAttention()
    )

    input_ids = torch.randint(0, 200, (16, 128))  # (batch_size, seq_len)
    logits = model(input_ids)
    print(logits.shape)  # Expected output: (16, 128, 200)

    model_varlen = MoMLLM(
        vocab_size=200,
        hidden_dim=64,
        num_memories=4,
        k=2,
        num_layers=2,
        mom_implem=mom_varlen.MoM,
        layer_norm=nn.LayerNorm,
        update_module=mom_varlen.LinearAttentionVarlen()
    )
    model_varlen.load_state_dict(model.state_dict())

    logits_triton = model_varlen(input_ids)
    print(logits_triton.shape)  # Expected output: (16, 128, 200)
    max_abs_diff = torch.max(torch.abs(logits - logits_triton)).item()
    print(f"Max absolute difference between logits: {max_abs_diff}")
    # assert torch.allclose(logits, logits_triton, atol=1e-4), "Les logits des deux modèles ne correspondent pas!"

    model_paper = MoMLLM(
        vocab_size=32000,
        hidden_dim=1024,
        num_memories=4,
        k=2,
        num_layers=24,
        mom_implem=mom_varlen.MoM,
        layer_norm=nn.LayerNorm,
        update_module=mom_varlen.LinearAttentionVarlen()
    )

    print(f"Model size: {sum(p.numel() for p in model_paper.layers.parameters())/1e6:.2f}M parameters")

