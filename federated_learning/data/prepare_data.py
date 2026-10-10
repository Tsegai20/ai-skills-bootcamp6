"""
Run this once to generate CSV data files for all sites.
Usage: python prepare_data.py
"""
from datasets import load_dataset
import pandas as pd
import numpy as np
import os

os.makedirs(".", exist_ok=True)

print("Loading AG News dataset...")
dataset = load_dataset("fancyzhx/ag_news")

train_data = dataset["train"]
test_data  = dataset["test"]

np.random.seed(42)

# Separate by class
class_indices = {
    i: [j for j, x in enumerate(train_data) if x["label"] == i]
    for i in range(4)
}

site_configs = {
    "site_a": 0,  # World
    "site_b": 1,  # Sports
    "site_c": 2,  # Business
    "site_d": 3   # Sci/Tech
}

label_names = ["World", "Sports", "Business", "Sci/Tech"]

for site_name, class_idx in site_configs.items():
    indices = np.random.choice(
        class_indices[class_idx], 600, replace=False
    ).tolist()

    texts  = [train_data[i]["text"]  for i in indices]
    labels = [train_data[i]["label"] for i in indices]

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(f"{site_name}.csv", index=False)
    print(f"Saved {site_name}.csv: {len(df)} samples of {label_names[class_idx]}")

# Global test set
test_indices = []
for c in range(4):
    idxs = [i for i, x in enumerate(test_data) if x["label"] == c][:150]
    test_indices.extend(idxs)

test_texts  = [test_data[i]["text"]  for i in test_indices]
test_labels = [test_data[i]["label"] for i in test_indices]
test_df = pd.DataFrame({"text": test_texts, "label": test_labels})
test_df.to_csv("test.csv", index=False)
print(f"Saved test.csv: {len(test_df)} samples (150 per class)")
print("\nAll data files ready")
