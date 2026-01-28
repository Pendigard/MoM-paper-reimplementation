import sys
import os
import torch
import torch.nn as nn
import argparse
import string
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.module.retnet import RetNetModule

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


def load_model(path, device):
    model=RetNetModule(CONFIG).to(device)
    state_dict = torch.load(path, map_location = device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_answer(model, tokenizer, device, context, question, max_new_tokens=20):
    # Construction du prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    if input_ids.shape[1] > CONFIG["seq_len"] - max_new_tokens:
        tokens_to_keep = CONFIG["seq_len"] - max_new_tokens
        input_ids = input_ids[:, -tokens_to_keep:]

    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated_ids)
            last_token_logits = logits[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    new_tokens = generated_ids[0, input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return output_text

# j'ai juste fait les fonctions pour faciliter le traitement
def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def normalize_text(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_squad(model, tokenizer, device, num_samples=100):
    dataset = load_dataset("squad", split=f"validation[:{num_samples}]")
    
    exact_matches = 0
    f1_scores = 0 
        
    for i, example in enumerate(tqdm(dataset)):
        context = example["context"]
        question = example["question"]
        answers = example["answers"]["text"] 
        
        prediction = generate_answer(model, tokenizer, device, context, question)
        
        normalized_prediction = normalize_text(prediction)
        normalized_answers = [normalize_text(a) for a in answers]
        
        is_correct = any(normalized_prediction == a for a in normalized_answers)
        
        if is_correct:
            exact_matches += 1
            
        if i < 5:
            print(f"Exemple numero {i}")
            print(f"Question: {question}")
            print(f"Vraies réponses: {answers}")
            print(f"Prédiction: '{prediction}'")
            print(f"Correct: {is_correct}")

    accuracy = exact_matches / len(dataset)
    print(f"Exact Match (EM): {accuracy:.2%}")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Chemin vers le fichier .pt")
    parser.add_argument("--samples", type=int, default=100, help="Nombre d'exemples à tester")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    CONFIG["vocab_size"] = len(tokenizer)

    model = load_model(args.path, device)
    
    evaluate_squad(model, tokenizer, device, num_samples=args.samples)