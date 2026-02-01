import sys
import os
import torch
import torch.nn as nn
import argparse
import string
import math
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

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
    "lr": 3e-4,
    "max_steps": 50000,
    "dataset_name": "cerebras/SlimPajama-627B",
}

def get_model(model_name, device):
    if model_name== "mom":
        print("MoM")
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
    elif model_name == "retnet":
        print("RetNet")
        model = RetNetModule(CONFIG)
    elif model_name == "hgrn":
        print("HGRN")
        model = HGRN(CONFIG)
    else :
        raise ValueError(f"Type de modèle inconnu")
    
    return model.to(device)

def calcul_perplexity(data_path, path, model_name, num_samples = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(model_name,device)
    checkpoint = torch.load(path, map_location = device, weights_only=False)

    try:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else :
            model.load_state_dict(checkpoint)
    except Exception as e : 
        print (f"ERREUR FATALE lors du chargement : {e}")
        return

    model.eval()

    print("Chargement du dataset")
    dataset = load_dataset("json", data_files=data_path, split="train", streaming=False)
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    total_loss = 0
    total_token = 0
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=tokenizer.pad_token_id)
    print("Calcul de la perplexité")

    with torch.no_grad():
        for i, exemple in enumerate(tqdm(dataset)):
            if len(exemple["text"]) < 50 :
                continue
        
            inputs = tokenizer(
                exemple["text"],
                return_tensors="pt",
                truncation=True,
                max_length=CONFIG["seq_len"]

            ).to(device)

            input_ids = inputs["input_ids"]

            targets = input_ids.clone()

            logits = model(input_ids)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., :-1:].contiguous()

            loss = criterion(shift_logits.view(-1, CONFIG["vocab_size"]), shift_labels.view(-1))
            non_pad_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item()
            total_token += non_pad_tokens

    if total_token == 0:
        print("Aucun token valide trouvé aie aie aie")
        return
    avg_loss = total_loss/total_token

    perplexity = math.exp(avg_loss)    
    print("Résultats")
    print(f"Cross Entropy Loss :{avg_loss:.4f}")
    print(f"Perplexité : {perplexity:.2f}")

if __name__ == "__main__":

    DATA_FILE = "/users/nfs/Vrac/21400184/Projet_deepl/MoM-paper-reimplementation/data/valid/example_train_1026.jsonl.zst"
    # calcul_perplexity(
    #     DATA_FILE,
    #     path = "mom_gla__final_slimpajama_step100000.pt",
    #     model_name = "mom",
    #     num_samples = 500
    # )

    # calcul_perplexity(
    #     DATA_FILE,
    #     path = "hgrn_gla__final_slimpajama_step50000.pt",
    #     model_name = "hgrn",
    #     num_samples = 500
    # )

    calcul_perplexity(
        DATA_FILE,
        path = "retnet_gla__final_slimpajama_step100000.pt",
        model_name = "retnet",
        num_samples = 500
    )


