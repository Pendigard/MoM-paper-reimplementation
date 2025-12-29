import sys
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
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
logging.basicConfig(level=logging.INFO)


from src.module.mom import MoM 

CONFIG = {
    "vocab_size": 32000,    
    "dim": 256,             
    "num_layers": 4,      
    "num_memories": 4,     
    "top_k": 2,            
    "seq_len": 512,         
    "batch_size": 2,      
    "lr": 3e-4,            
    "max_steps": 5000,      
    "dataset_name": "cerebras/SlimPajama-627B" 
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
            batch_size, 
            self.config["num_memories"] + 1, 
            self.config["dim"], 
            self.config["dim"], 
            device=x.device
        )

        for layer in self.layers:
            x, _ = layer(x, M.clone())
            
        x = x.transpose(0, 1)
        
        x = self.norm(x)
        logits = self.head(x)
        return logits

def get_data_loader(tokenizer, config):
    print(f"Chargement du daatset")
    dataset = load_dataset(config["dataset_name"], split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["seq_len"],
            padding="max_length"
        )

    dataset = dataset.map(tokenize, remove_columns=["text", "meta", "redpajama_set_name"])
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    return dataloader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    CONFIG["vocab_size"] = len(tokenizer)

    model = MoMLLM(CONFIG).to(device)
    print(f"Modèle Nano-MoM créé: {sum(p.numel() for p in model.parameters())/1e6:.2f} Millions de paramètres")

    dataloader = get_data_loader(tokenizer, CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    data_iter = iter(dataloader)
    
    pbar = tqdm(range(CONFIG["max_steps"]))
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
            
        input_ids = batch["input_ids"].to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        train_input = input_ids[:, :-1] 
        train_target = input_ids[:, 1:] 
        optimizer.zero_grad()
        logits = model(train_input) 
        
        loss = criterion(logits.reshape(-1, CONFIG["vocab_size"]), train_target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")
        if (step + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"nano_mom_step_{step+1}.pt")
            
            generate_text(model, tokenizer, device)

def generate_text(model, tokenizer, device, prompt="Computer science is"):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(20):
            logits = model(ids)
            last_token_logits = logits[:, -1, :]
            
            if torch.isnan(last_token_logits).any():
                print("aie")
                break
                
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)
            ids = torch.cat([ids, next_token], dim=1)
    
    if ids.shape[1] > 0:
        print(f"Output: {tokenizer.decode(ids[0])}\n")
    else:
        print("Output: (Vide)\n")
        
    model.train()

if __name__ == "__main__":
    train()