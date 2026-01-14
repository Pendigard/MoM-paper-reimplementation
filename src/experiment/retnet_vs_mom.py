import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.module.naive_mom import MoM
from src.module.retnet import RetNetModule
from src.module.hgrn import HGRN
from src.experiment.generate_recall_data import generate_recall_data


CONFIG = {
"vocab_size": 200,
    "dim": 64,
    "num_layers": 2,
    "num_memories": 4,
    "top_k": 2,           
    "seq_len": 128,
    "num_examples": 1000,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "steps": 3000,
    "dropout": 0.0
}

class MoMLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.embedding = nn.Embedding(config["vocab_size"], config["dim"])
        
        self.layers = nn.ModuleList([
            MoM(
                input_dim=config["dim"], 
                hidden_dim=config["dim"], 
                num_memories=config["num_memories"], 
                k=config["top_k"]
            ) for _ in range(config["num_layers"])
        ])
        
        self.norm = nn.LayerNorm(config["dim"])
        self.head = nn.Linear(config["dim"], config["vocab_size"], bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(0, 1)
        
        batch_size = x.shape[1]

        M = torch.zeros(
            self.config["dim"], 
            self.config["dim"], 
            device=x.device
        )

        for layer in self.layers:
            x = layer(x, M.clone())
            
        x = x.transpose(0, 1)
        
        x = self.norm(x)
        logits = self.head(x)
        return logits

def train_model(model, name, config):
    print(f"--- Entraînement de {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    data_loader = generate_recall_data(
        num_examples=config["num_examples"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        batch_size=config["batch_size"]
    )

    model.train()
    losses = []
    accuracy_list = []

    iter_loader = iter(data_loader)

    for step in tqdm(range(config["steps"])):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(data_loader)
            batch = next(iter_loader)
        
        x, y, *_ = batch
        x, y = x.to(device), y.to(device)

        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        logits = model(x)
        
        loss = criterion(logits[:, -1, :], y[:, -1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        preds = torch.argmax(logits[:, -1, :], dim=-1)
        accuracy = (preds == y[:, -1]).float().mean().item()
        accuracy_list.append(accuracy)

    return losses, accuracy_list

def run(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retnet = RetNetModule(CONFIG).to(device)
    losses_retnet, accuracy_retnet = train_model(retnet, "RetNet", CONFIG)

    hgrn = HGRN(CONFIG).to(device)
    losses_hgrn, accuracy_hgrn = train_model(hgrn, "HGRN", CONFIG)

    mom = MoMLLM(CONFIG).to(device)
    losses_mom, accuracy_mom = train_model(mom, "MoM", CONFIG)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_retnet, label='RetNet (Fixed Decay)')
    plt.plot(losses_hgrn, label='HGRN (Gated Decay)') 
    plt.plot(losses_mom, label='MoM (Multi-Memory)')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_retnet, label='RetNet')
    plt.plot(accuracy_hgrn, label='HGRN')         
    plt.plot(accuracy_mom, label='MoM')
    plt.title('Training Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('MoM-paper-reimplementation/fig/retnet_vs_mom.png')


if __name__ == "__main__":
    run()

            