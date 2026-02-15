import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from src.module.llm import TransformerLLM

import src.module.naive_mom as nm
from src.module.mom_llm import MoMLLM, MoMHybridLLM

from src.module.fla_llm import FLALLM
from fla.layers import GatedLinearAttention, HGRN2Attention, DeltaNet, GatedSlotAttention, GatedDeltaNet

from tqdm import tqdm

import random
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

import sys

class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # Nécessaire pour la compatibilité avec sys.stdout
        pass


class AssociativeRecallDataset(Dataset):
    def __init__(self, num_keys=50, num_values=50, seq_len=20, num_samples=10000, num_queries=1, static=False):
        self.vocab_size = num_keys + num_values
        self.num_keys = num_keys
        self.num_values = num_values
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_queries = num_queries
        self.static = static
        
        if self.static:
            self.static_data = [self._generate_one_sample() for _ in range(num_samples)]

    def _generate_one_sample(self):
        keys = [random.randint(0, self.num_keys - 1) for _ in range(self.seq_len)]
        assoc = {}
        values = []
        for k in keys:
            if k not in assoc:
                assoc[k] = random.randint(self.num_keys, self.vocab_size - 1)
            values.append(assoc[k])
        
        sequence = []
        for k, v in zip(keys, values):
            sequence.append(k)
            sequence.append(v)
            
        target_idx = random.sample(range(self.seq_len), self.num_queries)
        for ti in target_idx:
            sequence.append(keys[ti])
            
        x = torch.tensor(sequence, dtype=torch.long)
        y = torch.tensor([values[ti] for ti in target_idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.static:
            return self.static_data[idx]
        return self._generate_one_sample()

def evaluate(model, data_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_bar = tqdm(data_loader, desc="Validation", leave=False)
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            logits, _ = model(x_batch)
            logits = logits[:, -y_batch.size(1):, :]
            
            loss = criterion(logits.reshape(-1, model.vocab_size), y_batch.reshape(-1))
            val_loss += loss.item()
            
            _, predicted = torch.max(logits, dim=2)
            total += y_batch.numel()
            correct += (predicted == y_batch).sum().item()

    avg_val_loss = val_loss / len(data_loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy

def train(model, train_loader, val_loader, num_epochs=10, 
          learning_rate=0.001, device='cpu', log_interval=100, aux_loss_weight=0.01):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    losses = []
    val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f"[Train] Époque {epoch+1}/{num_epochs}", leave=False) as train_bar:
            for x_batch, y_batch in train_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                logits, aux_loss = model(x_batch)
                
                logits = logits[:, -y_batch.size(1):, :]

                loss = criterion(logits.reshape(-1, model.vocab_size), y_batch.reshape(-1)) + aux_loss_weight * aux_loss
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_bar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = train_loss / len(train_loader)
        losses.append(avg_train_loss)


        if (epoch + 1) % log_interval == 0:
            avg_val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            tqdm.write(f"Époque [{epoch+1}/{num_epochs}] | "
                       f"Train Loss: {avg_train_loss:.4f} | "
                       f"Val Loss: {avg_val_loss:.4f} | "
                       f"Val Acc: {val_accuracy * 100:.2f}% | "
                       f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return model, losses, val_accuracy

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_pipeline(model, train_loader, val_loader, test_loader, num_epochs=20, learning_rate=5e-4, device='cpu'):
    model, losses, val_accuracy = train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        log_interval=25
    )
    test_loss, test_accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    return model, losses, val_accuracy, test_loss, test_accuracy
    
if __name__ == "__main__":
    vocab_size = 200
    num_keys = vocab_size // 2
    num_values = vocab_size // 2
    seq_len = 20
    batch_size = 32
    num_epochs = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sys.stdout = Logger("training_log.txt")

    print(f"Using device: {device}")

    num_layers = 4

    mom_dim = 128
    num_memories = 6
    k=2
    update_module = nm.GLAAttention

    base_dim = mom_dim * k

    seq_lens = [10, 20, 40]
    expanded = [False, True]
    models = ['Transformer', 'GLA', 'DeltaNet']


    for seq_len in seq_lens:
        num_queries = seq_len // 2
        train_dataset = AssociativeRecallDataset(num_keys=num_keys, num_values=num_values, seq_len=seq_len, num_samples=5000, num_queries=num_queries, static=False)
        val_dataset = AssociativeRecallDataset(num_keys=num_keys, num_values=num_values, seq_len=seq_len, num_samples=1000, num_queries=num_queries, static=True)
        test_dataset = AssociativeRecallDataset(num_keys=num_keys, num_values=num_values, seq_len=seq_len, num_samples=5000, num_queries=num_queries, static=True)

        print(f"\n=== Séquence longueur: {seq_len}, Batch size: {batch_size}, num_queries: {num_queries} ===")

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        for expanded_flag in expanded:
            losses_save = []
            if not expanded_flag:
                suffix = f"_{seq_len}"
                model_mom = MoMLLM(
                    vocab_size=vocab_size,
                    hidden_dim=mom_dim,
                    num_memories=num_memories,
                    k=k,
                    num_layers=num_layers,
                    mom_implem=nm.MoM,
                    layer_norm=nn.LayerNorm,
                    update_module=update_module,
                    update_module_args = (mom_dim, mom_dim, num_memories)
                )

                model_mom, mom_losses, mom_val_accuracy, mom_test_loss, mom_test_accuracy = run_pipeline(
                    model_mom,
                    train_loader,
                    val_loader,
                    test_loader,
                    num_epochs=num_epochs,
                    learning_rate=5e-4,
                    device=device
                )

                print(f"MoM{suffix} Parameters: {num_parameters(model_mom)}")
                print(f"MoM{suffix} Test Accuracy: {mom_test_accuracy * 100:.2f}%")
                print(f"MoM{suffix} Test Loss: {mom_test_loss:.4f}")

                losses_save.append((mom_losses, "MoM"))

                model_hybrid = MoMHybridLLM(
                    vocab_size=vocab_size,
                    hidden_dim=mom_dim,
                    num_memories=num_memories,
                    k=k,
                    num_layers=num_layers,
                    mom_implem=nm.MoM,
                    layer_norm=nn.LayerNorm,
                    update_module=update_module,
                    update_module_args = (mom_dim, mom_dim, num_memories)
                )

                model_hybrid, hybrid_losses, hybrid_val_accuracy, hybrid_test_loss, hybrid_test_accuracy = run_pipeline(
                    model_hybrid,
                    train_loader,
                    val_loader,
                    test_loader,
                    num_epochs=num_epochs,
                    learning_rate=5e-4,
                    device=device
                )
                print(f"MoM-Hybrid{suffix} Parameters: {num_parameters(model_hybrid)}")
                print(f"MoM-Hybrid{suffix} Test Accuracy: {hybrid_test_accuracy * 100:.2f}%")
                print(f"MoM-Hybrid{suffix} Test Loss: {hybrid_test_loss:.4f}")
                losses_save.append((hybrid_losses, "MoM-Hybrid"))
                base_dim = mom_dim

            else:
                suffix = f"_{seq_len}_expanded"
                base_dim = mom_dim * k
            
            for model_name in models:
                if model_name == 'Transformer':
                    model = TransformerLLM(
                        vocab_size=vocab_size,
                        hidden_dim=base_dim,
                        num_heads=8,
                        num_layers=num_layers,
                        dropout=0.1,
                        layer_norm=nn.LayerNorm
                    )
                else:
                    model = FLALLM(
                        vocab_size=vocab_size,
                        hidden_dim=base_dim,
                        num_layers=num_layers,
                        layer=GatedLinearAttention if model_name == 'GLA' else DeltaNet
                    )
                    if model_name == 'DeltaNet':
                        # to dtype bfloat16 for DeltaNet or it crashes
                        model = model.to(torch.bfloat16)
                
                print(f"{model_name}{suffix} Parameters: {num_parameters(model)}")
                model, losses, val_accuracy, test_loss, test_accuracy = run_pipeline(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    num_epochs=num_epochs,
                    learning_rate=5e-4,
                    device=device
                )
                print(f"{model_name}{suffix} Test Accuracy: {test_accuracy * 100:.2f}%")
                print(f"{model_name}{suffix} Test Loss: {test_loss:.4f}")

                losses_save.append((losses, model_name))
            
            plt.figure()
            for loss_vals, label in losses_save:
                plt.plot(loss_vals, label=label)
            plt.legend()
            plt.xlabel("Époques")
            plt.ylabel("Perte d'entraînement moyenne")
            plt.title(f"Courbe de perte d'entraînement (seq_len={seq_len}, expanded={expanded_flag})")
            plt.savefig(f"training_loss_curve_seq{seq_len}_expanded{expanded_flag}.png")
            plt.close()