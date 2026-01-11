import torch
import torch.nn as nn
import torch.nn.functional as F

class HGRNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.input_proj = nn.Linear(dim, dim, bias=False)
        self.forget_proj = nn.Linear(dim, dim, bias=True)
        self.input_gate_proj = nn.Linear(dim, dim, bias=True)
        self.output_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, h_prev):
        candidate = self.input_proj(x)
        candidate = F.silu(candidate)

        f = torch.sigmoid(self.forget_proj(x))
        i = torch.sigmoid(self.input_gate_proj(x))

        h_new = f * h_prev + i * candidate

        output = self.output_proj(h_new)

        return output, h_new

class HGRN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config["dim"]

        self.embedding = nn.Embedding(config["vocab_size"], config["dim"])

        self.layers = nn.ModuleList([
            HGRNLayer(config["dim"]) for _ in range(config["num_layers"])
        ])

        self.norm = nn.LayerNorm(config["dim"])
        self.head = nn.Linear(config["dim"], config["vocab_size"], bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        batch_size, seq_len, _ = x.shape

        h_states = [
            torch.zeros(batch_size, self.dim, device=x.device)
            for _ in range(self.config["num_layers"])
        ]

        outputs = []
        x = x.transpose(0, 1)

        for t in range(seq_len):
            courant = x[t]
            for i, layer in enumerate(self.layers):
                courant, h_states[i] = layer(courant, h_states[i])
            outputs.append(courant)

        out = torch.stack(outputs, dim=0).transpose(0, 1)
        out = self.norm(out)
        logits = self.head(out)

        return logits, torch.tensor(0.0, device=input_ids.device)




