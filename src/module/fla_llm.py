
import torch
import torch.nn as nn
from fla.layers import GatedLinearAttention
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


class FLALLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, layer : nn.Module, layer_norm = nn.LayerNorm, *args, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer = layer

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        
        self.layers = nn.ModuleList([
            layer(
                hidden_size=hidden_dim
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, materialize_output: bool = True) -> torch.Tensor:
        x = self.embedding(input_ids)
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            x_norm = self.norms_1[i](x)
            layer_out, _, _ = layer(x_norm)
            x = x + layer_out
            x = x + self.MLPs[i](self.norms_2[i](x))

        if materialize_output:
            outputs = self.output_layer(x)
            return outputs, total_aux_loss
        else:
            outputs = self.output_layer(x[:, -1, :])
            return outputs, total_aux_loss
 
if __name__ == "__main__":
    vocab_size = 100
    hidden_dim = 128
    num_layers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FLALLM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        layer=GatedLinearAttention
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (2, 10)).to(device)  # (batch_size, seq_length)
    outputs, aux_loss = model(input_ids)
    print("Outputs shape:", outputs.shape)  # Expected: (2, 10, vocab_size)
    print("Auxiliary loss:", aux_loss)