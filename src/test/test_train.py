from src.module.MoMpipeline import MoMPipeline

import torch
import torch.nn as nn
from tqdm import tqdm



def train(model: MoMPipeline, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, epochs: int):
    model.train()
    total_loss = 0.0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0
        for batch in tqdm(data_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
        total_loss += epoch_loss
        epoch_loss = epoch_loss / len(data_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
    return total_loss / epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mom_pipeline = MoMPipeline(dim_input=10, dim_hidden=32, dim_output=1, dim_embedding=16, num_memories=4, k=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mom_pipeline.parameters(), lr=0.001)