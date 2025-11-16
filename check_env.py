import torch
import transformers
import peft
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from transformers import (
    set_seed,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
)

print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("pandas      :", pd.__version__)
print("numpy       :", np.__version__)
print("sklearn     :", sklearn.__version__)
print("seaborn     :", sns.__version__)
print("streamlit   :", st.__version__)

print("\n✅ Semua import berhasil. Environment lokal sudah match kebutuhan Colab.")
