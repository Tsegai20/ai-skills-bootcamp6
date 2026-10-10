from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import requests
import time
import os
import logging
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Federated Learning Aggregator")

SITES = {
    "site_a": os.getenv("SITE_A_URL", "http://site_a:8000"),
    "site_b": os.getenv("SITE_B_URL", "http://site_b:8000"),
    "site_c": os.getenv("SITE_C_URL", "http://site_c:8000"),
    "site_d": os.getenv("SITE_D_URL", "http://site_d:8000"),
}

MODEL_NAME  = "distilbert-base-uncased"
NUM_LABELS  = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
TEST_PATH   = os.getenv("TEST_DATA_PATH", "/data/test.csv")
MLFLOW_URI  = os.getenv("MLFLOW_URI", "sqlite:////mlflow/mlflow.db")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = None
tokenizer = None
test_df   = None
history   = []
is_training = False
current_round = 0

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Federated_Learning")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

def load_resources():
    global model, tokenizer, test_df
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    if test_df is None:
        test_df = pd.read_csv(TEST_PATH)
        logger.info(f"Loaded {len(test_df)} test samples")
    if model is None:
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
    return model, tokenizer, test_df

def weights_to_dict(state_dict):
    return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

def dict_to_weights(d):
    return {k: torch.tensor(v) for k, v in d.items()}

def fedavg(results):
    total = sum(r["num_samples"] for r in results)
    weights = [r["num_samples"] / total for r in results]
    averaged = {}
    for key in results[0]["weights"].keys():
        averaged[key] = sum(
            w * torch.tensor(r["weights"][key]).float()
            for w, r in zip(weights, results)
        )
    return averaged

def evaluate(m, tok, df):
    m.eval()
    dataset = NewsDataset(df["text"].tolist(), df["label"].tolist(), tok)
    loader = DataLoader(dataset, batch_size=64)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = m(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[name] = float((all_preds[mask] == all_labels[mask]).mean())
    return float(accuracy), per_class

class RoundConfig(BaseModel):
    epochs: int = 1
    batch_size: int = 32
    lr: float = 2e-5
    timeout: int = 600

class MultiRoundConfig(BaseModel):
    num_rounds: int = 5
    epochs: int = 1
    batch_size: int = 32
    lr: float = 2e-5
    timeout: int = 600

@app.get("/")
def health():
    return {
        "status": "healthy",
        "service": "FL Aggregator",
        "current_round": current_round,
        "is_training": is_training,
        "total_rounds_done": len(history),
        "sites": list(SITES.keys()),
        "device": str(DEVICE)
    }

@app.get("/sites")
def sites_status():
    result = {}
    for name, url in SITES.items():
        try:
            r = requests.get(f"{url}/", timeout=5)
            result[name] = r.json()
        except Exception as e:
            result[name] = {"status": "unreachable", "error": str(e)}
    return result

@app.get("/history")
def get_history():
    return {
        "total_rounds": len(history),
        "history": history,
        "best_accuracy": max((r["global_accuracy"] for r in history), default=0)
    }

@app.post("/round")
def run_round(config: RoundConfig = RoundConfig()):
    global model, is_training, current_round

    if is_training:
        raise HTTPException(409, "Training already in progress")

    is_training = True
    current_round += 1
    round_num = current_round
    t0 = time.time()

    try:
        m, tok, test = load_resources()
        global_weights = weights_to_dict(m.state_dict())

        logger.info(f"Starting round {round_num}")

        # Send to all sites
        site_results = []
        for site_name, site_url in SITES.items():
            try:
                logger.info(f"  Sending to {site_name}...")
                resp = requests.post(
                    f"{site_url}/train",
                    json={
                        "weights":    global_weights,
                        "round_num":  round_num,
                        "epochs":     config.epochs,
                        "batch_size": config.batch_size,
                        "lr":         config.lr
                    },
                    timeout=config.timeout
                )
                if resp.status_code == 200:
                    result = resp.json()
                    site_results.append(result)
                    logger.info(
                        f"  {site_name}: acc={result['local_accuracy']:.4f} "
                        f"t={result['training_time']:.1f}s"
                    )
                else:
                    logger.error(f"  {site_name} error: {resp.text[:100]}")
            except Exception as e:
                logger.error(f"  {site_name} failed: {e}")

        if not site_results:
            raise HTTPException(500, "All sites failed")

        # FedAvg
        logger.info("FedAvg aggregation...")
        averaged = fedavg(site_results)
        m.load_state_dict(averaged)

        # Evaluate
        logger.info("Evaluating global model...")
        global_acc, per_class = evaluate(m, tok, test)

        elapsed = time.time() - t0
        prev_acc = history[-1]["global_accuracy"] if history else 0.0
        improvement = global_acc - prev_acc

        # MLflow logging
        with mlflow.start_run(run_name=f"round_{round_num}"):
            mlflow.log_params({
                "round": round_num,
                "num_sites": len(site_results),
                "epochs": config.epochs
            })
            mlflow.log_metrics({
                "global_accuracy": global_acc,
                "improvement": improvement,
                "round_time": elapsed
            })
            for r in site_results:
                mlflow.log_metric(
                    f"{r['site_name']}_accuracy",
                    r["local_accuracy"],
                    step=round_num
                )

        round_result = {
            "round_num":         round_num,
            "global_accuracy":   global_acc,
            "per_class_accuracy": per_class,
            "improvement":       improvement,
            "round_time":        elapsed,
            "sites_responded":   len(site_results),
            "site_results": [
                {
                    "site":           r["site_name"],
                    "local_accuracy": r["local_accuracy"],
                    "training_time":  r["training_time"],
                    "num_samples":    r["num_samples"]
                }
                for r in site_results
            ]
        }

        history.append(round_result)
        is_training = False

        logger.info(
            f"Round {round_num} done. "
            f"Global: {global_acc:.4f} ({improvement:+.4f}) in {elapsed:.1f}s"
        )

        return round_result

    except Exception as e:
        is_training = False
        logger.error(f"Round {round_num} failed: {e}")
        raise HTTPException(500, str(e))

@app.post("/train")
def run_multi_round(config: MultiRoundConfig = MultiRoundConfig()):
    results = []
    for i in range(config.num_rounds):
        logger.info(f"Starting round {i+1}/{config.num_rounds}")
        result = run_round(RoundConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            timeout=config.timeout
        ))
        results.append(result)

    best = max(results, key=lambda x: x["global_accuracy"])
    return {
        "total_rounds":    config.num_rounds,
        "final_accuracy":  results[-1]["global_accuracy"],
        "best_accuracy":   best["global_accuracy"],
        "best_round":      best["round_num"],
        "results":         results
    }
