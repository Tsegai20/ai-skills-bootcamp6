from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=f"FL Site")

SITE_NAME   = os.getenv("SITE_NAME",  "site_unknown")
SITE_CLASS  = int(os.getenv("SITE_CLASS", "0"))
DATA_PATH   = os.getenv("DATA_PATH",  "/data/site_a.csv")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/data/weights")
MODEL_NAME  = "prajjwal1/bert-tiny"
NUM_LABELS  = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

def load_resources():
    global model, tokenizer, train_df
    if tokenizer is None:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if train_df is None:
        train_df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(train_df)} samples")
    if model is None:
        logger.info("Loading bert-tiny model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        logger.info(f"Model loaded on {DEVICE}")
    return model, tokenizer, train_df

class TrainRequest(BaseModel):
    round_num: int
    epochs: int = 1
    batch_size: int = 32
    lr: float =2e-5

class TrainResponse(BaseModel):
    site_name:      str
    round_num:      int
    local_accuracy: float
    num_samples:    int
    training_time:  float
    weights_path:   str

@app.get("/")
def health():
    return {
        "status":        "healthy",
        "site":          SITE_NAME,
        "class":         LABEL_NAMES[SITE_CLASS],
        "device":        str(DEVICE),
        "is_training":   status["is_training"],
        "round":         status["round"],
        "last_accuracy": status["last_accuracy"],
        "model":         MODEL_NAME
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

        # Load global weights from shared volume
        global_weights_path = os.path.join(WEIGHTS_DIR, "global_weights.pt")
        if os.path.exists(global_weights_path):
            state_dict = torch.load(
                global_weights_path,
                map_location=DEVICE,
                weights_only=True
            )
            m.load_state_dict(state_dict)
            logger.info("Loaded global weights from shared volume")
        else:
            logger.info("No global weights — using initialized model")

        m.train()
        dataset   = NewsDataset(df["text"].tolist(), df["label"].tolist(), tok)
        loader    = DataLoader(dataset, batch_size=req.batch_size, shuffle=True)
        optimizer = AdamW(m.parameters(), lr=req.lr)

        for epoch in range(req.epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = m(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE)
                )
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
            logger.info(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

        m.eval()
        correct = 0
        with torch.no_grad():
            for batch in loader:
                out = m(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE)
                )
                preds = out.logits.argmax(dim=-1)
                correct += (preds == batch["labels"].to(DEVICE)).sum().item()

        accuracy = correct / len(dataset)
        elapsed  = time.time() - t0

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        weights_path = os.path.join(WEIGHTS_DIR, f"{SITE_NAME}_weights.pt")
        torch.save(m.state_dict(), weights_path)
        logger.info(f"Saved weights to {weights_path} ({os.path.getsize(weights_path)/1024:.1f} KB)")

        status.update({"is_training": False, "round": req.round_num, "last_accuracy": accuracy})
        logger.info(f"{SITE_NAME} R{req.round_num}: acc={accuracy:.4f} t={elapsed:.1f}s")

        return TrainResponse(
            site_name=SITE_NAME,
            round_num=req.round_num,
            local_accuracy=accuracy,
            num_samples=len(dataset),
            training_time=elapsed,
            weights_path=weights_path
        )

    except Exception as e:
        status["is_training"] = False
        logger.error(f"Training error: {e}")
        raise HTTPException(500, str(e))
