"""
BERT vs Traditional Models Comparison
======================================
Run on Colab. Evaluates BERT on dev.csv and compares
against saved traditional model metrics.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

NEURAL_PATH = "/content/drive/MyDrive/models/model_a/neural/"
TRAD_PATH   = "/content/drive/MyDrive/models/model_a/traditional/"
DATA_PATH   = "/content/drive/MyDrive/data/raw/dev.csv"
OUT_PATH    = "/content/drive/MyDrive/models/model_a/traditional/bert_vs_traditional.csv"

EVAL_SAMPLES = 4887  # Full validation set

# ── Load BERT ──────────────────────────────────────────────────────────────
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained(NEURAL_PATH)
model     = AutoModelForMultipleChoice.from_pretrained(NEURAL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

# ── Load dev data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[df['answer'].isin(['A','B','C','D'])].dropna(subset=['article','question','A','B','C','D'])
df = df.sample(n=min(EVAL_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
print(f"Evaluating on {len(df)} samples...")

label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# ── BERT Evaluation ────────────────────────────────────────────────────────
y_true, y_pred = [], []

for idx, row in df.iterrows():
    if idx % 50 == 0:
        print(f"  {idx}/{len(df)}...")

    article  = str(row['article'])[:300]
    question = str(row['question'])
    options  = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
    label    = label_map[row['answer']]

    try:
        encoding = tokenizer(
            [[article + " " + question, opt] for opt in options],
            return_tensors="pt", padding=True,
            truncation=True, max_length=256
        )
        inputs = {k: v.unsqueeze(0).to(device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred = torch.argmax(logits, dim=-1).item()
        y_true.append(label)
        y_pred.append(pred)
    except Exception as e:
        y_true.append(label)
        y_pred.append(0)

bert_acc  = accuracy_score(y_true, y_pred)
bert_f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
bert_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
bert_rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)

print(f"\nBERT Results: Acc={bert_acc:.4f}  F1={bert_f1:.4f}  Prec={bert_prec:.4f}  Rec={bert_rec:.4f}")

# ── Load Traditional Metrics ───────────────────────────────────────────────
trad_binary   = pd.read_csv(f"{TRAD_PATH}/binary_classification_metrics.csv")
trad_ensemble = pd.read_csv(f"{TRAD_PATH}/ensemble_binary_metrics.csv")

# ── Build Comparison Table ─────────────────────────────────────────────────
bert_row = pd.DataFrame([{
    "Model":     "BERT (RoBERTa-RACE)",
    "Accuracy":  round(bert_acc, 6),
    "Precision": round(bert_prec, 6),
    "Recall":    round(bert_rec, 6),
    "Macro F1":  round(bert_f1, 6),
    "ROC-AUC":   "-",
    "Type":      "Neural"
}])

# Standardize columns
def prep(df, model_type):
    df = df.copy()
    df["Type"] = model_type
    col_map = {}
    for c in df.columns:
        if "precision" in c.lower(): col_map[c] = "Precision"
        if "recall"    in c.lower(): col_map[c] = "Recall"
        if "f1"        in c.lower(): col_map[c] = "Macro F1"
        if "roc"       in c.lower(): col_map[c] = "ROC-AUC"
        if "accuracy"  in c.lower(): col_map[c] = "Accuracy"
        if "model"     in c.lower(): col_map[c] = "Model"
    return df.rename(columns=col_map)

trad_df = pd.concat([
    prep(trad_binary, "Traditional"),
    prep(trad_ensemble, "Ensemble")
], ignore_index=True)

comparison = pd.concat([trad_df, bert_row], ignore_index=True)
cols = ["Model", "Type", "Accuracy", "Precision", "Recall", "Macro F1", "ROC-AUC"]
cols = [c for c in cols if c in comparison.columns]
comparison = comparison[cols].sort_values("Accuracy", ascending=False)

print("\n=== BERT vs Traditional ===")
print(comparison.to_string(index=False))

comparison.to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
