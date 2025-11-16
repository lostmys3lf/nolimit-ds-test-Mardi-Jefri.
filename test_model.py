from pathlib import Path

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Path ke folder checkpoint lokalmu
BEST_DIR_MODEL = Path("models/indoBERT_best")                 # tokenizer
CKPT_DIR       = BEST_DIR_MODEL / "checkpoint-2390"           # model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load tokenizer + model
tokenizer = BertTokenizer.from_pretrained(BEST_DIR_MODEL)
model     = BertForSequenceClassification.from_pretrained(CKPT_DIR)
model.to(device)
model.eval()

# 2. Contoh teks uji
text = "aku senang banget karena masih ada orang tua yang peduli sama aku"

# 3. Tokenisasi
encodings = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128,
)

encodings = {k: v.to(device) for k, v in encodings.items()}

# 4. Inference tanpa gradient
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

# 5. Mapping label
label_id = int(probs.argmax())
id2label = {0: "negatif", 1: "positif"}

print("Logits:", logits)
print("Probabilitas [negatif, positif]:", probs)
print("Prediksi label:", label_id, f"({id2label[label_id]})")
