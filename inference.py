from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification


BEST_DIR_MODEL = Path("models/indoBERT_best")               # tokenizer
CKPT_DIR = BEST_DIR_MODEL / "checkpoint-2390"               # model
ID2LABEL = {0: "Negatif", 1: "Positif"}


def load_model():
    """
    Load tokenizer + model IndoBERT fine-tuned secara lokal.
    Dipanggil sekali saja (nanti kita cache di Streamlit).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(BEST_DIR_MODEL)
    model = BertForSequenceClassification.from_pretrained(CKPT_DIR)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(
    text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    device: torch.device,
    max_length: int = 128,
) -> Dict:
    """
    Lakukan prediksi sentimen untuk satu teks.

    Return dict berisi:
      - label_id (0/1)
      - label_name ("Negatif"/"Positif")
      - probs (np.array panjang 2)
      - logits (np.array panjang 2)
    """
    if not text or text.strip() == "":
        raise ValueError("Teks input kosong.")

    # Tokenisasi
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits  # shape: [1, 2]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        logits_np = logits[0].cpu().numpy()

    label_id = int(np.argmax(probs))
    label_name = ID2LABEL.get(label_id, str(label_id))

    return {
        "label_id": label_id,
        "label_name": label_name,
        "probs": probs,
        "logits": logits_np,
    }
