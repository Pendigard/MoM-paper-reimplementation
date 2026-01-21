import sys
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import json
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
    "batch_size": 2,      
    "lr": 3e-4,            
    "max_steps": 5000,      
    "update_module": LinearAttention(),
    "dataset_name": "cerebras/SlimPajama-627B" 
}
class MoMLLMss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.embedding = nn.Embedding(config["vocab_size"], config["dim"])
        
        self.layers = nn.ModuleList([])
        
        for _ in range(config["num_layers"]):
            self.layers.append(
                MoM(
                    input_dim=config["dim"], 
                    hidden_dim=config["hidden_dim"], 
                    num_memories=config["num_memories"], 
                    k=config["top_k"],
                    update_module=GDeltaAttention(
                    input_dim=config["dim"],
                    hidden_dim=config["dim"],
                    num_memories=config["num_memories"]
                    )
                )
            )
        
        self.norm = nn.LayerNorm(config["dim"])
        self.head = nn.Linear(config["dim"], config["vocab_size"], bias=False)
        
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

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(0, 1)
        
        batch_size = x.shape[1]
        
        M = torch.zeros(
            self.config["dim"], 
            self.config["dim"], 
            device=x.device
        )
        
        total_aux_loss = 0.0

        for layer in self.layers:
            out_mom, M_new, aux = layer(x, M.clone())
            x = x + out_mom 
            
            total_aux_loss += aux

        x = x.transpose(0, 1)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits, total_aux_loss

def get_data_loader(tokenizer, config):
    print(f"Chargement du dataset LOCAL (Mode Hors-Ligne)...")
    
    local_file = "/users/nfs/Vrac/21400184/Projet_deepl/MoM-paper-reimplementation/data/example_train_0.jsonl.zst"
    
    dataset = load_dataset("json", data_files=local_file, split="train", streaming=False)
    
    dataset = dataset.shuffle(seed=42) 
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["seq_len"],
            padding="max_length"
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "meta"])
    dataset = dataset.with_format("torch")
    
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    return dataloader

def train(args):
    run_name = f"{args.model}_mem{args.memories}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    CONFIG["vocab_size"] = len(tokenizer)

    if args.model == "mom":
        model = MoMLLM(
            vocab_size=CONFIG["vocab_size"],
            hidden_dim=CONFIG["hidden_dim"],
            num_memories=CONFIG["num_memories"],
            k=CONFIG["top_k"],
            num_layers=CONFIG["num_layers"],
            update_module=LinearAttention()
        ).to(device)
        print(f"Modèle Nano-MoM créé: {sum(p.numel() for p in model.parameters())/1e6:.2f} Millions de paramètres")
    elif args.model == "retnet":
        model = RetNetModule(CONFIG).to(device)
        print(f"Modèle RetNet créé: {sum(p.numel() for p in model.parameters())/1e6:.2f} Millions de paramètres")
    elif args.model == "hgrn":
        model = HGRN(CONFIG).to(device)
        print(f"Modèle HGRN créé: {sum(p.numel() for p in model.parameters())/1e6:.2f} Millions de paramètres")


    dataloader = get_data_loader(tokenizer, CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss_history = []

    model.train()
    data_iter = iter(dataloader)
    
    pbar = tqdm(range(CONFIG["max_steps"]))
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            print("Fin du dataset atteinte.")
            break
        except Exception as e:
            print(f"\nFichier corrompu détecté à l'étape {step}. Erreur: {e}")
            continue
            
        input_ids = batch["input_ids"].to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        train_input = input_ids[:, :-1] 
        train_target = input_ids[:, 1:] 
        optimizer.zero_grad()
        logits, aux_loss = model(train_input) 
        
        B, L, V = logits.shape
        task_loss = criterion(logits.reshape(B*L, V), train_target.reshape(-1))
        loss = task_loss + 0.01 * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(task_loss.item())
        pbar.set_description(f"Loss: {task_loss.item():.4f}")
        if (step + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"{args.model}_slimpajama_step{step+1}.pt")
            
            generate_text(model, tokenizer, device)
    os.makedirs("results", exist_ok=True)
    with open(f"results/loss_{run_name}.json", "w") as f:
        json.dump(loss_history, f)
    print(f"Sauvegarde terminée.")

def generate_text(model, tokenizer, device, prompt="Computer science is"):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(20):
            outputs = model(ids)
            if isinstance(outputs, tuple):
                logits = outputs[0] 
            else:
                logits = outputs
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mom", "retnet", "hgrn"], default="mom", help="Type de modèle à entraîner")
    parser.add_argument("--memories", type=int, default=4, help="Nombre de mémoires pour MoM")
    args = parser.parse_args()
    train(args)