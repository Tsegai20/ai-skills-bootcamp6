from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import pandas as pd
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=f"FL Site — {os.getenv('SITE_NAME', 'unknown')}")

# Config
SITE_NAME  = os.getenv("SITE_NAME",  "site_unknown")
SITE_CLASS = int(os.getenv("SITE_CLASS", "0"))
DATA_PATH  = os.getenv("DATA_PATH",  "/data/site_a.csv")
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State
model     = None
tokenizer = None
train_df  = None
status    = {"round": 0, "is_training": False, "last_accuracy": None}

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

def load_resources():
    global model, tokenizer, train_df
    if tokenizer is None:
        logger.info(f"Loading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    if train_df is None:
        logger.info(f"Loading data from {DATA_PATH}")
        train_df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(train_df)} samples")
    if model is None:
        logger.info("Loading model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
    return model, tokenizer, train_df

def weights_to_dict(state_dict):
    return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

def dict_to_weights(d):
    return {k: torch.tensor(v) for k, v in d.items()}

class TrainRequest(BaseModel):
    weights: Dict[str, List]
    round_num: int
    epochs: int = 1
    batch_size: int = 32
    lr: float = 2e-5

class TrainResponse(BaseModel):
    site_name:      str
    round_num:      int
    weights:        Dict[str, List]
    local_accuracy: float
    num_samples:    int
    training_time:  float

@app.get("/")
def health():
    return {
        "status":       "healthy",
        "site":         SITE_NAME,
        "class":        LABEL_NAMES[SITE_CLASS],
        "device":       str(DEVICE),
        "is_training":  status["is_training"],
        "round":        status["round"],
        "last_accuracy":status["last_accuracy"]
    }

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    global model, status

    if status["is_training"]:
        raise HTTPException(409, f"{SITE_NAME} already training")

    status["is_training"] = True
    t0 = time.time()

    try:
        m, tok, df = load_resources()

        # Load global weights
        m.load_state_dict(dict_to_weights(req.weights))
        m.train()

        dataset = NewsDataset(
            df["text"].tolist(),
            df["label"].tolist(),
            tok
        )
        loader = DataLoader(dataset, batch_size=req.batch_size, shuffle=True)
        optimizer = AdamW(m.parameters(), lr=req.lr)

        # Use fp16 if GPU available
        scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

        total_loss = 0
        for epoch in range(req.epochs):
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                if scaler:
                    with torch.cuda.amp.autocast():
                        out = m(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                    scaler.scale(out.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = m(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
                    out.loss.backward()
                    optimizer.step()

                total_loss += out.loss.item()

        # Evaluate
        m.eval()
        correct = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                out = m(input_ids=input_ids, attention_mask=attention_mask)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()

        accuracy = correct / len(dataset)
        elapsed = time.time() - t0

        status.update({"is_training": False, "round": req.round_num,
                       "last_accuracy": accuracy})

        logger.info(f"{SITE_NAME} R{req.round_num}: acc={accuracy:.4f} t={elapsed:.1f}s")

        return TrainResponse(
            site_name=SITE_NAME,
            round_num=req.round_num,
            weights=weights_to_dict(m.state_dict()),
            local_accuracy=accuracy,
            num_samples=len(dataset),
            training_time=elapsed
        )

    except Exception as e:
        status["is_training"] = False
        logger.error(f"Training error: {e}")
        raise HTTPException(500, str(e))
