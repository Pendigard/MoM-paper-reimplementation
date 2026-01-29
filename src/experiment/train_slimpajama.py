import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import json
import glob
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

VRAC_PATH = "/users/nfs/Vrac/21400184/.cache_hf"
os.environ["HF_HOME"] = VRAC_PATH
os.environ["HF_DATASETS_CACHE"] = os.path.join(VRAC_PATH, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(VRAC_PATH, "models")
logging.basicConfig(level=logging.ERROR)

from src.module.naive_mom import MoM, LinearAttention, GLAAttention, GDeltaAttention
from src.module.retnet import RetNetModule
from src.module.hgrn import HGRN
from src.module.mom_llm import MoMLLM

CONFIG = {
    "vocab_size": 32000,
    "dim": 256,
    "num_layers": 4,
    "num_memories": 4,
    "hidden_dim": 256,
    "top_k": 2,
    "seq_len": 512,
    "batch_size": 4,
    "lr": 1e-4, # de base 3e-4 j'ai baissé pour la reprise
    "max_steps": 50000, 
    "update_module": LinearAttention(),
    "dataset_name": "cerebras/SlimPajama-627B",
    "gradient_accumulation_steps": 8
}

def get_data_loader(tokenizer, config):
    data_files = sorted(glob.glob("/users/nfs/Vrac/21400184/Projet_deepl/MoM-paper-reimplementation/data/*.jsonl.zst"))
    print(f"Dataset : {len(data_files)} fichiers trouvés.")
    
    dataset = load_dataset("json", data_files=data_files, split="train", streaming=False)
    dataset = dataset.shuffle(seed=42)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["seq_len"],
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        return {"input_ids": input_ids}

    return DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    CONFIG["vocab_size"] = len(tokenizer)

    if args.model == "mom":
        update_layer = GLAAttention(
            input_dim=CONFIG["dim"],      
            hidden_dim=CONFIG["hidden_dim"],     
            num_memories=CONFIG["num_memories"]
        )
        model = MoMLLM(
            vocab_size=CONFIG["vocab_size"],
            hidden_dim=CONFIG["hidden_dim"],
            num_memories=CONFIG["num_memories"],
            k=CONFIG["top_k"],
            num_layers=CONFIG["num_layers"],
            update_module=update_layer
        ).to(device)
    elif args.model == "retnet":
        model = RetNetModule(CONFIG).to(device)
    elif args.model == "hgrn":
        model = HGRN(CONFIG).to(device)

    start_step = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Chargement poids : {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            start_step = 50000
            CONFIG["max_steps"] = 100000
            print(f"Reprise step {start_step} -> {CONFIG['max_steps']}")
        else:
            print("Checkpoint introuvable. Exit.")
            return

    dataloader = get_data_loader(tokenizer, CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["max_steps"], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss_history = []

    model.train()
    optimizer.zero_grad()
    
    pbar = tqdm(range(start_step, CONFIG["max_steps"]), initial=start_step, total=CONFIG["max_steps"])
    data_iter = iter(dataloader)
    
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        input_ids = batch["input_ids"].to(device)
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        
        train_input = input_ids[:, :-1] 
        train_target = input_ids[:, 1:] 
        
        logits, aux_loss = model(train_input) 
        
        B, L, V = logits.shape
        task_loss = criterion(logits.reshape(B*L, V), train_target.reshape(-1))
        
        if isinstance(aux_loss, torch.Tensor) and aux_loss.item() != 0:
            loss = task_loss + 0.01 * aux_loss
        else:
            loss = task_loss
        
        loss = loss / CONFIG["gradient_accumulation_steps"]
        loss.backward()
        
        if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_history.append(task_loss.item())
        pbar.set_description(f"Loss: {task_loss.item():.4f}")
        
        if (step + 1) % 5000 == 0:
            torch.save(model.state_dict(), f"{args.model}_gla__final_slimpajama_step{step+1}.pt")
            
    os.makedirs("results", exist_ok=True)
    with open(f"results/loss_{args.model}_extended.json", "w") as f:
        json.dump(loss_history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mom", "retnet", "hgrn"])
    parser.add_argument("--memories", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    train(args)