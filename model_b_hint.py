import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_df = pd.read_csv('/content/drive/MyDrive/raw/train.csv')
val_df   = pd.read_csv('/content/drive/MyDrive/raw/dev.csv')

STOPWORDS = {
    'a','an','the','is','it','in','on','at','to','of','and','or','but',
    'for','with','as','by','from','this','that','was','are','be','been',
    'has','have','had','do','did','will','would','could','should','may',
    'might','its','i','you','he','she','we','they','his','her','their'
}

def split_sentences(text):
    parts = re.split(r'[.!?]+', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]

def sentence_features(sentence, question, position, total):
    q_words = set(question.lower().split()) - STOPWORDS
    s_words = set(sentence.lower().split()) - STOPWORDS
    overlap  = len(s_words & q_words) / len(q_words) if q_words else 0.0
    pos_norm = position / max(total - 1, 1)
    length   = len(sentence.split())
    return [overlap, pos_norm, length]

def create_hint_training_data(df, max_rows=3000):
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
        for i, sent in enumerate(sentences):
            feats = sentence_features(sent, question, i, total)
            label = 1 if correct in sent.lower() else 0
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

print("Creating hint training data...")
X_train, y_train = create_hint_training_data(train_df, max_rows=3000)
print(f"Train samples: {len(X_train)}  |  Positive (hint): {y_train.sum()}")

print("Training Logistic Regression hint model...")
model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
model.fit(X_train, y_train)
print("Hint model trained.")

print("\nEvaluating on val set...")
X_val, y_val = create_hint_training_data(val_df, max_rows=500)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, target_names=['Not Hint', 'Hint']))

val_acc  = (y_pred == y_val).mean()
hint_results = pd.DataFrame([{
    'model': 'LogisticRegression',
    'task': 'hint_sentence_scoring',
    'val_samples': len(y_val),
    'accuracy': round(val_acc, 4),
    'positive_rate': round(y_val.mean(), 4)
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
