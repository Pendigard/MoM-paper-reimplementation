import matplotlib.pyplot as plt
import json
import os
import numpy as np

def smooth(scalars, weight=0.9):  
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

results_dir = "results"
files = {
    "HGRN": "loss_hgrn_mem4.json", 
    "Linear Baseline (RetNet-like)": "loss_retnet_mem4.json",
    "MoM": "loss_mom_mem4.json"
}

plt.figure(figsize=(10, 6))

for label, filename in files.items():
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            data_smooth = smooth(data, 0.95)
            plt.plot(data_smooth, label=label)
    else:
        print(f"Fichier manquant : {filename}")

plt.xlabel("Training Steps")
plt.ylabel("Training Loss (NLL)")
plt.title("Pretraining loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.savefig("repro_figure_7.png")
print("Graphique généré : repro_figure_7.png")