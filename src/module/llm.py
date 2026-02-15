import torch
import torch.nn as nn


class AttentionCore(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return y


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, layer_norm=nn.LayerNorm):
        super().__init__()
        self.norm1 = layer_norm(hidden_dim)
        self.norm2 = layer_norm(hidden_dim)
        self.core = AttentionCore(hidden_dim, num_heads, dropout=dropout)
        self.mlp = MLP(hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.core(self.norm1(x), attn_mask=attn_mask))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        layer_norm=nn.LayerNorm,
        output_size=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size if output_size is not None else vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout=dropout, layer_norm=layer_norm)
             for _ in range(num_layers)]
        )

        self.final_norm = layer_norm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, self.output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        causal_mask: torch.Tensor = None,
        materialize_output: bool = True
    ) -> torch.Tensor:

        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Taille de séquence dépasse la max_seq_len"

        tok = self.embedding(input_ids)

        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos = self.pos_embedding(pos_ids)

        x = tok + pos

        x = x.transpose(0, 1)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.final_norm(x)

        x = x.transpose(0, 1)

        if materialize_output:
            return self.output_layer(x), 0.0
        else:
            return self.output_layer(x[:, -1, :]), 0.0