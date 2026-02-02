import torch
import torch.nn.functional as F
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

# ==================================================
# 1. Import FDA task dataset (from HF)
# ==================================================
from datasets import load_dataset

dataset = load_dataset("hazyresearch/based-fda")["validation"]

# ==================================================
# 2. Each entry in the dataset consists of a text file description, a key, a value and the text itself.
#    We test whether when prompted with a context and a key the model can retrieve the correct value.
# ==================================================
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# ==================================================
# 3. Load one of our models to see what it produces
# ==================================================
from src.module.naive_mom import GLAAttention
from src.module.mom_llm import MoMLLM

from src.experiment.eval_perplexity import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model("mom", device)
path = "/Vrac/21400184/Projet_deepl/mom_gla__final_slimpajama_step50000.pt"
checkpoint = torch.load(path, map_location=device, weights_only=False)

# code copied from src/experiment/eval_perplexity.py
try:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else :
        model.load_state_dict(checkpoint)
except Exception as e : 
    print (f"ERREUR FATALE lors du chargement : {e}")
    sys.exit(0)

model.eval()

# ==================================================
# 4. Test on prompt 0 what the model generates.
#    Test how to calculate the accuracy.
# ==================================================
entry = dataset[0]
key = entry["key"]
value = entry["value"]
text = entry["text"]

prompt = f"{text} \n {key}:"
t_prompt = torch.tensor(tokenizer(prompt)["input_ids"])
init_len = len(t_prompt)

def generate_greedy(
    model,
    input_ids,
    eos_token_id=tokenizer.eos_token_id,
    max_len=25,
    device=device
):
    # expect input to be batched; unsqueeze in the case of a single prompt
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(input_ids)[0]
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if (next_token == eos_token_id).all():
                break

    return input_ids

test_one = False
if test_one:
    output = generate_greedy(model, t_prompt)
    text_out = tokenizer.decode(output[0, init_len:])
    print("Predicted value:", text_out)
    print("True value:", value)

# ==================================================
# 5. Read dataset in batches.
#    Use dataset.iter()
#    Save output to file for later analysis.
# ==================================================
results = dict()
batch_size = 8

for batch_idx, batch in enumerate(dataset.iter(batch_size=batch_size)):
    if batch_idx % 10 == 9:
        print("Finished 10 batches.")

    value = batch["value"]
    prompt = [f"{text} \n {key}:" for text, key in zip(batch["text"], batch["key"])]

    input_ids = torch.tensor(tokenizer(prompt, padding=True)["input_ids"])
    init_len = input_ids.shape[1]
    
    output = generate_greedy(model, input_ids, max_len=25)
    text_out = tokenizer.decode(output[:, init_len:])
    for i in range(batch_size):
        results[batch_size*batch_idx + i] = dict()
        results[batch_size*batch_idx + i]["predicted"] = text_out[i]
        results[batch_size*batch_idx + i]["true"] = text_out[i]

with open(f"{current_dir}/results/fda.json", "w") as fp:
    json.dump(results, fp)