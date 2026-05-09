import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt',    quiet=True)
nltk.download('wordnet',  quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.translate.bleu_score   import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
try:
    from rouge_score import rouge_scorer as _rs
    _ROUGE = True
except ImportError:
    print("[WARN] pip install rouge-score for ROUGE metrics")
    _ROUGE = False

_sf = SmoothingFunction().method1

def _bleu(hyp, ref):
    h = hyp.lower().split(); r = ref.lower().split()
    return sentence_bleu([r], h, smoothing_function=_sf) if h else 0.0

def _rouge(hyp, ref):
    if not _ROUGE:
        return 0.0, 0.0, 0.0
    s = _rs.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True).score(ref, hyp)
    return s['rouge1'].fmeasure, s['rouge2'].fmeasure, s['rougeL'].fmeasure

def _meteor(hyp, ref):
    try:
        return nltk_meteor([ref.lower().split()], hyp.lower().split())
    except Exception:
        return 0.0

train_df = pd.read_csv('/content/drive/MyDrive/data/raw/train.csv')
val_df   = pd.read_csv('/content/drive/MyDrive/data/raw/dev.csv')

STOPWORDS = {
    'a','an','the','is','it','in','on','at','to','of','and','or','but',
    'for','with','as','by','from','this','that','was','are','be','been',
    'has','have','had','do','did','will','would','could','should','may',
    'might','its','i','you','he','she','we','they','his','her','their'
}

def split_sentences(text):
    parts = re.split(r'[.!?]+', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]

def sentence_features(sentence, question, answer, position, total):
    q_words = set(question.lower().split()) - STOPWORDS
    s_words = set(sentence.lower().split()) - STOPWORDS
    a_words = set(answer.lower().split())  - STOPWORDS

    q_overlap  = len(s_words & q_words) / len(q_words) if q_words else 0.0
    a_overlap  = len(s_words & a_words) / len(a_words) if a_words else 0.0
    pos_norm   = position / max(total - 1, 1)
    length     = len(sentence.split())
    len_norm   = min(length / 30.0, 1.0)
    is_first   = 1 if position == 0 else 0
    is_last    = 1 if position == total - 1 else 0
    return [q_overlap, a_overlap, pos_norm, length, len_norm, is_first, is_last]

def create_hint_training_data(df, max_rows=10000):
    X, y = [], []
    df = df.sample(n=min(max_rows, len(df)), random_state=42)
    for _, row in df.iterrows():
        if row.get('answer') not in ['A', 'B', 'C', 'D']:
            continue
        article  = str(row['article'])  if not pd.isna(row['article'])  else ''
        question = str(row['question']) if not pd.isna(row['question']) else ''
        correct  = str(row[row['answer']]).lower()
        sentences = split_sentences(article)
        if not sentences:
            continue
        total = len(sentences)
        correct_tokens = set(correct.split()) - STOPWORDS
        for i, sent in enumerate(sentences):
            feats = sentence_features(sent, question, correct, i, total)
            sent_tokens = set(sent.lower().split()) - STOPWORDS
            overlap_ratio = (
                len(correct_tokens & sent_tokens) / len(correct_tokens)
                if correct_tokens else 0.0
            )
            label = 1 if overlap_ratio >= 0.5 else 0
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

print("Creating hint training data...")
X_train, y_train = create_hint_training_data(train_df, max_rows=10000)
print(f"Train samples: {len(X_train)}  |  Positive (hint): {y_train.sum()}")

print("Training Logistic Regression hint model...")
model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
model.fit(X_train, y_train)
print("Hint model trained.")

print("\nEvaluating on val set...")
X_val, y_val = create_hint_training_data(val_df, max_rows=1000)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, target_names=['Not Hint', 'Hint']))

val_acc = (y_pred == y_val).mean()

# Hint Precision@K: for each question rank sentences by P(hint=1), take top-K,
# count fraction that are gold hint sentences (contain correct answer)
def hint_precision_at_k(model, df, max_rows=300, k=3):
    precisions = []
    df = df.sample(n=min(max_rows, len(df)), random_state=42)
    for _, row in df.iterrows():
        if row.get('answer') not in ['A', 'B', 'C', 'D']:
            continue
        article  = str(row['article'])  if not pd.isna(row['article'])  else ''
        question = str(row['question']) if not pd.isna(row['question']) else ''
        correct  = str(row[row['answer']]).lower()
        sentences = split_sentences(article)
        if len(sentences) < 2:
            continue
        total = len(sentences)
        feats  = np.array([sentence_features(s, question, correct, i, total)
                           for i, s in enumerate(sentences)])
        probs  = model.predict_proba(feats)[:, 1]
        top_k_idx = np.argsort(probs)[::-1][:k]
        hits = sum(1 for i in top_k_idx if correct in sentences[i].lower())
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0

print("\nComputing Hint Precision@3...")
p_at_3 = hint_precision_at_k(model, val_df, max_rows=300, k=3)
print(f"Hint Precision@3: {p_at_3:.4f}")

hint_results = pd.DataFrame([{
    'model': 'LogisticRegression',
    'task': 'hint_sentence_scoring',
    'val_samples': len(y_val),
    'accuracy': round(val_acc, 4),
    'positive_rate': round(y_val.mean(), 4),
    'precision_at_3': round(p_at_3, 4)
}])
print(hint_results.to_string(index=False))

os.makedirs('/content/drive/MyDrive/models', exist_ok=True)

hint_bundle = {'hint_model': model}
with open('/content/drive/MyDrive/models/hint_model.pkl', 'wb') as f:
    pickle.dump(hint_bundle, f)

hint_results.to_csv('/content/drive/MyDrive/models/hint_results.csv', index=False)

print("\nDone. Saved:")
print("  - hint_model.pkl")
print("  - hint_results.csv")

# ── BLEU / ROUGE / METEOR evaluation ────────────────────────────────────────
print("\n[BLEU / ROUGE / METEOR — Hint Generation]")
print("="*80)
hint_gen_rows = []
for _, row in val_df.head(200).iterrows():
    if row.get('answer') not in ['A','B','C','D']:
        continue
    article  = str(row['article'])     if not pd.isna(row['article'])    else ''
    question = str(row['question'])    if not pd.isna(row['question'])   else ''
    correct  = str(row[row['answer']]) if not pd.isna(row[row['answer']]) else ''
    sentences = split_sentences(article)
    if not sentences or not correct:
        continue
    refs = [s for s in sentences if correct.lower() in s.lower()]
    if not refs:
        correct_tokens = set(correct.lower().split()) - STOPWORDS
        refs = [s for s in sentences
                if len(set(s.lower().split()) & correct_tokens) / max(len(correct_tokens),1) >= 0.5]
    if not refs:
        continue
    total = len(sentences)
    q_words = set(question.lower().split()) - STOPWORDS
    a_words = set(correct.lower().split())  - STOPWORDS
    feats   = []
    for i, sent in enumerate(sentences):
        s_words   = set(sent.lower().split()) - STOPWORDS
        q_ov = len(s_words & q_words) / len(q_words) if q_words else 0.0
        a_ov = len(s_words & a_words) / len(a_words) if a_words else 0.0
        pos  = i / max(total - 1, 1)
        ln   = len(sent.split())
        feats.append([q_ov, a_ov, pos, ln, min(ln/30.0, 1.0),
                      1 if i == 0 else 0, 1 if i == total-1 else 0])
    probs   = model.predict_proba(np.array(feats))[:, 1]
    top_idx = np.argsort(probs)[::-1][:3]
    generated = [sentences[i] for i in top_idx]
    b_s, r1_s, r2_s, rl_s, m_s = [], [], [], [], []
    for gen in generated:
        b        = max(_bleu(gen, ref)               for ref in refs)
        r1,r2,rl = max((_rouge(gen, ref) for ref in refs), key=lambda x: x[0])
        m        = max(_meteor(gen, ref)             for ref in refs)
        b_s.append(b); r1_s.append(r1); r2_s.append(r2); rl_s.append(rl); m_s.append(m)
    hint_gen_rows.append({
        'BLEU':    np.mean(b_s),
        'ROUGE-1': np.mean(r1_s),
        'ROUGE-2': np.mean(r2_s),
        'ROUGE-L': np.mean(rl_s),
        'METEOR':  np.mean(m_s),
    })

hint_gen_df = pd.DataFrame(hint_gen_rows)
print(f"Samples: {len(hint_gen_df)}")
print(hint_gen_df.mean().round(4).to_string())
hint_gen_df.mean().round(4).to_frame('Score').reset_index().rename(
    columns={'index':'Metric'}).to_csv(
    '/content/drive/MyDrive/models/hint_generation_metrics.csv', index=False)
print("[OK] hint_generation_metrics.csv saved")
print("="*80)
