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

from src.module.naive_mom import MoM, LinearAttention, GLAAttention, GDeltaAttention
from src.module.retnet import RetNetModule
from src.module.hgrn import HGRN
from src.experiment.generate_recall_data import generate_recall_data
from src.module.mom_llm import MoMLLM


CONFIG = {
"vocab_size": 200,
"dim": 64,
"num_layers": 2,
"num_memories": 4,
"k": 2,        
"seq_len": 128,
"num_examples": 1000,
"batch_size": 16,
"learning_rate": 1e-3,
"steps": 500,
"hidden_dim": 64,
"dropout": 0.0
}

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

        outputs = model(x)

        if isinstance(outputs, tuple):
            logits = outputs[0]
            aux_loss = outputs[1]
        else:
            logits = outputs
            aux_loss = 0.0
        
        loss_base = criterion(logits[:, -1, :], y[:, -1])

        loss = loss_base + 0.01 * aux_loss
        
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

    mom_gdelta = MoMLLM(
        vocab_size=CONFIG["vocab_size"],
        hidden_dim=CONFIG["hidden_dim"],
        num_memories=CONFIG["num_memories"],
        k=CONFIG["k"],
        num_layers=CONFIG["num_layers"],
        update_module=GDeltaAttention(CONFIG["dim"], CONFIG["hidden_dim"], CONFIG["num_memories"]),

    ).to(device)
    losses_mom_gdelta, accuracy_mom_gdelta = train_model(mom_gdelta, "MoM_GDelta", CONFIG)

    retnet = RetNetModule(CONFIG).to(device)
    losses_retnet, accuracy_retnet = train_model(retnet, "RetNet", CONFIG)

    hgrn = HGRN(CONFIG).to(device)
    losses_hgrn, accuracy_hgrn = train_model(hgrn, "HGRN", CONFIG)

    mom_linear = MoMLLM(
        vocab_size=CONFIG["vocab_size"],
        hidden_dim=CONFIG["hidden_dim"],
        num_memories=CONFIG["num_memories"],
        k=CONFIG["k"],
        num_layers=CONFIG["num_layers"],
    ).to(device)
    losses_mom_linear, accuracy_mom_linear = train_model(mom_linear, "MoM_Linear", CONFIG)

    mom_gla = MoMLLM(
        vocab_size=CONFIG["vocab_size"],
        hidden_dim=CONFIG["hidden_dim"],
        num_memories=CONFIG["num_memories"],
        k=CONFIG["k"],
        num_layers=CONFIG["num_layers"],
        update_module=GLAAttention(CONFIG["dim"], CONFIG["hidden_dim"], CONFIG["num_memories"]),
        
    ).to(device)
    losses_mom_gla, accuracy_mom_gla = train_model(mom_gla, "MoM_GLA", CONFIG)


    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_retnet, label='RetNet (Fixed Decay)')
    plt.plot(losses_hgrn, label='HGRN (Gated Decay)') 
    plt.plot(losses_mom_linear, label='MoM (Linear)')
    plt.plot(losses_mom_gla, label='MoM (GLA)')
    plt.plot(losses_mom_gdelta, label='MoM (GDelta)')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_retnet, label='RetNet')
    plt.plot(accuracy_hgrn, label='HGRN') 
    plt.plot(accuracy_mom_linear, label='MoM (Linear)')
    plt.plot(accuracy_mom_gla, label='MoM (GLA)')
    plt.plot(accuracy_mom_gdelta, label='MoM (GDelta)')        
    plt.title('Training Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('MoM-paper-reimplementation/fig/retnet_vs_moms.png')


if __name__ == "__main__":
    run()

            