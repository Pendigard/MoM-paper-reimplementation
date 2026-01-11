import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.module.mom import MoM
from src.module.retnet import RetNetModule
from src.module.hgrn import HGRN

class MoMWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = MoM(
            input_dim=config["vocab_size"],  
            hidden_dim=config["dim"],       
            num_memories=config["num_memories"],
            k=config["top_k"],              
        )
       
        self.embedding = nn.Embedding(config["vocab_size"], config["dim"])

        self.model.input_dim = config["dim"]
        self.model.W_k = nn.Linear(config["dim"], config["dim"] * (config["num_memories"] + 1))
        self.model.W_v = nn.Linear(config["dim"], config["dim"] * (config["num_memories"] + 1))
        self.model.W_g = nn.Linear(config["dim"], config["num_memories"])
        self.model.W_q = nn.Linear(config["dim"], config["dim"])


    def forward(self, x):
        x_emb = self.embedding(x)
        
        x_emb = x_emb.transpose(0, 1) # (Seq, Batch, Dim)
        
        batch_size = x_emb.shape[1]
        device = x_emb.device
        
       
        num_mems = self.model.num_memories
        h_dim = self.model.hidden_dim
        
        M_0 = torch.zeros(
            batch_size, 
            num_mems + 1, 
            h_dim, 
            h_dim, 
            device=device
        )
        
        logits, _ = self.model(x_emb, M_0)
        
        return logits.transpose(0, 1)

class BaselineTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['dim'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['dim'],
            nhead=2,
            dim_feedforward=config['dim'] * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        self.head = nn.Linear(config['dim'], config['vocab_size'])

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.head(x)
        return x

class NaiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        return self.out_proj(out)

class NaiveTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = nn.ModuleList([
            NaiveAttention(config["dim"]) for _ in range(config["num_layers"])
        ])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x) 
        return x


def benchmark(model, seq_len, batch_size=1, device='cuda'):
    times = []
    memories = []
    model.to(device)
    model.eval()

    print(f"benchmark du modele {model.__class__.__name__} ")

    for seq in seq_len:
        x = torch.randint(0, 1000, (batch_size, seq)).to(device)
        with torch.no_grad():
            _ = model(x)  # Warm-up

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        end_event.record()
        torch.cuda.synchronize()

        temps_ecoule = start_event.elapsed_time(end_event) / 10  # moyenne sur 10 runs
        memoire_max = torch.cuda.max_memory_allocated() / (1024 ** 2)  # en mebaoctets

        times.append(temps_ecoule)
        memories.append(memoire_max)

        print(f"Seq Len: {seq}, temps: {temps_ecoule:.2f} ms, memoire: {memoire_max:.2f} mo")

    return times, memories

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CONFIG = {
        'vocab_size': 1000,
        'dim': 128,
        'num_layers': 4,
        "num_memories": 4,
        "top_k":2,
        "dropout":0.0
    }

    seq_len = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    models = {
        'Transformer Pytorch': BaselineTransformer(CONFIG),
        'Transformer Naif': NaiveTransformer(CONFIG),
        'RetNet': RetNetModule(CONFIG),
        'HGRN': HGRN(CONFIG),
        'MoM': MoMWrapper(CONFIG)
    }
    results = {}

    for name, model in models.items():
        try:
            t, m = benchmark(model, seq_len, device=device)
            results[name] = {'time': t, 'memories': m}
        except RuntimeError as e:
            print("aie pas assez de mémoire pour le modèle ", name)
            filled = len(results.get(name, {}).get('time', []))
            remaining = len(seq_len)-filled
            results[name] = {
                "time": t + [None]*remaining,
                "memories": m + [None]*remaining
            }

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, data in results.items():
        valid_indice =  [i for i, v in enumerate(data['time']) if v is not None]
        valid_x = [seq_len[i] for i in valid_indice]
        valid_y = [data['time'][i] for i in valid_indice]

        plt.plot(valid_x, valid_y, marker='o', label=name)
    plt.title("Temps d'inférence vs Longueur de séquence")

    plt.xlabel("Longueur de séquence")
    plt.ylabel("Temps d'inférence (ms)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        valid_indice =  [i for i, v in enumerate(data['memories']) if v is not None]
        valid_x = [seq_len[i] for i in valid_indice]
        valid_y = [data['memories'][i] for i in valid_indice]

        plt.plot(valid_x, valid_y, marker='o', label=name)
    plt.title("Utilisation de la mémoire vs Longueur de séquence")
    plt.xlabel("Longueur de séquence")
    plt.ylabel("Utilisation de la mémoire (Mo)")
    plt.legend()
    plt.grid(True)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    save_path = os.path.join(root_dir, 'fig', 'linearity_benchmark.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Graphiques sauvegardés dans {save_path}")

if __name__ == "__main__":
    run_benchmark()
    