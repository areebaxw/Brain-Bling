"""
PHASE 3 - MODEL A: BINARY CLASSIFIERS (balanced class weights)
Feature assignment per spec:
  Logistic Regression : One-Hot (article + question + option)  — answer verification
  SVM                 : One-Hot (article + question + option)  — answer verification
  Naive Bayes         : Bag-of-words of question tokens only   — question type classification
  Random Forest       : Handcrafted lexical features           — difficulty estimation
"""

import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = '/content/drive/MyDrive/data/processed'
RAW_DIR       = '/content/drive/MyDrive/data/raw'
OUTPUT_DIR    = '/content/drive/MyDrive/models'
TRAIN_SAMPLE  = 50000


def load_data():
    print("\n[LOADING DATA]")
    print("=" * 80)

    # Primary: One-Hot features (LR + SVM)
    X_train_oh = sp.load_npz(f'{PROCESSED_DIR}/X_train_onehot.npz')
    X_val_oh   = sp.load_npz(f'{PROCESSED_DIR}/X_val_onehot.npz')
    y_train    = np.load(f'{PROCESSED_DIR}/y_train.npy')
    y_val      = np.load(f'{PROCESSED_DIR}/y_val.npy')

    print(f"[OK] X_train_oh: {X_train_oh.shape}  |  positives: {y_train.sum()} / {len(y_train)}")
    print(f"[OK] X_val_oh:   {X_val_oh.shape}")

    # Lexical features (RF — difficulty estimation)
    X_train_lex = X_val_lex = None
    lex_train_path = f'{PROCESSED_DIR}/train_lexical_features.csv'
    lex_val_path   = f'{PROCESSED_DIR}/val_lexical_features.csv'
    if os.path.exists(lex_train_path) and os.path.exists(lex_val_path):
        X_train_lex = pd.read_csv(lex_train_path).values.astype(np.float32)
        X_val_lex   = pd.read_csv(lex_val_path).values.astype(np.float32)
        print(f"[OK] X_train_lex: {X_train_lex.shape}")
    else:
        print("[WARN] Lexical features not found — RF will fall back to One-Hot")

    # Question-only bag-of-words (NB — question type classification)
    X_train_q = X_val_q = None
    try:
        train_df = pd.read_csv(f'{RAW_DIR}/train.csv')
        val_df   = pd.read_csv(f'{RAW_DIR}/dev.csv')
        q_train  = [str(r['question']) for _, r in train_df.iterrows() for _ in range(4)]
        q_val    = [str(r['question']) for _, r in val_df.iterrows() for _ in range(4)]
        qvec     = CountVectorizer(max_features=5000, binary=True)
        X_train_q = qvec.fit_transform(q_train)
        X_val_q   = qvec.transform(q_val)
        print(f"[OK] X_train_q (question BoW): {X_train_q.shape}")
    except Exception as e:
        print(f"[WARN] Question BoW failed: {e} — NB will fall back to One-Hot")

    # Stratified subsample — apply same indices to all feature sets
    sampled = None
    if len(y_train) > TRAIN_SAMPLE:
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos   = TRAIN_SAMPLE // 4
        n_neg   = TRAIN_SAMPLE - n_pos
        rng     = np.random.default_rng(42)
        sampled = np.concatenate([
            rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        ])
        rng.shuffle(sampled)
        X_train_oh  = X_train_oh[sampled]
        y_train     = y_train[sampled]
        if X_train_lex is not None:
            X_train_lex = X_train_lex[sampled]
        if X_train_q is not None:
            X_train_q = X_train_q[sampled]
        print(f"[Sampled] X_train: {X_train_oh.shape}  |  positives: {y_train.sum()} / {len(y_train)}")

    return X_train_oh, X_val_oh, X_train_lex, X_val_lex, X_train_q, X_val_q, y_train, y_val


def evaluate_option_selection(model, X_val, y_val, model_name):
    """
    Option-level accuracy: for each question (4 rows), pick option with highest P(correct).
    """
    try:
        probs = model.predict_proba(X_val)[:, 1]
    except Exception:
        return 0.0, 0.0

    n_questions = len(y_val) // 4
    correct = 0
    all_pred, all_true = [], []

    for i in range(n_questions):
        s = i * 4
        q_probs = probs[s:s+4]
        q_labels = y_val[s:s+4]
        true_idx = np.argmax(q_labels)
        pred_idx = np.argmax(q_probs)
        all_true.append(true_idx)
        all_pred.append(pred_idx)
        if pred_idx == true_idx:
            correct += 1

    acc = correct / n_questions
    f1  = f1_score(all_true, all_pred, average='macro', zero_division=0)
    return acc, f1


def train_and_evaluate(X_train_oh, X_val_oh, X_train_lex, X_val_lex,
                       X_train_q, X_val_q, y_train, y_val):
    print("\n[TRAINING CLASSIFIERS]")
    print("=" * 80)

    # Feature routing per spec
    # LR  : One-Hot (article + question + option)
    # SVM : One-Hot (article + question + option)
    # NB  : Bag-of-words of question tokens only
    # RF  : Handcrafted lexical features
    nb_train = X_train_q  if X_train_q  is not None else X_train_oh
    nb_val   = X_val_q    if X_val_q    is not None else X_val_oh
    rf_train = X_train_lex if X_train_lex is not None else X_train_oh
    rf_val   = X_val_lex   if X_val_lex   is not None else X_val_oh

    classifiers = {
        'Logistic Regression': (LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1,
            class_weight='balanced', C=1.0
        ), X_train_oh, X_val_oh),
        'SVM': (SGDClassifier(
            loss='modified_huber', max_iter=1000, random_state=42,
            class_weight='balanced', n_jobs=-1
        ), X_train_oh, X_val_oh),
        'Naive Bayes': (MultinomialNB(alpha=0.5), nb_train, nb_val),
        'Random Forest': (RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1
        ), rf_train, rf_val),
    }

    feature_desc = {
        'Logistic Regression': 'One-Hot (article+question+option)',
        'SVM':                 'One-Hot (article+question+option)',
        'Naive Bayes':         'Bag-of-words (question tokens only)',
        'Random Forest':       'Handcrafted lexical features',
    }

    binary_results  = []
    option_results  = []
    trained_models  = {}
    all_predictions = {}

    for name, (clf, X_tr, X_vl) in classifiers.items():
        print(f"\n[TRAINING] {name}  |  features: {feature_desc[name]}")
        print("-" * 60)

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_vl)

        print(classification_report(y_val, y_pred,
                                    target_names=['Non-Answer', 'Answer']))

        acc   = accuracy_score(y_val, y_pred)
        prec  = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec   = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        mf1   = f1_score(y_val, y_pred, average='macro', zero_division=0)
        try:
            proba = clf.predict_proba(X_vl)[:, 1]
            auc   = roc_auc_score(y_val, proba)
        except Exception:
            auc   = 0.5

        binary_results.append({
            'Model': name, 'Features': feature_desc[name],
            'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'Macro F1': mf1, 'ROC-AUC': auc
        })

        opt_acc, opt_f1 = evaluate_option_selection(clf, X_vl, y_val, name)
        option_results.append({
            'Model': name, 'Exact Match': opt_acc, 'Option Macro F1': opt_f1
        })

        trained_models[name] = clf
        try:
            all_predictions[name] = clf.predict_proba(X_vl)
        except Exception:
            all_predictions[name] = None

    return trained_models, all_predictions, binary_results, option_results


def print_and_save(trained_models, all_predictions, binary_results, option_results):
    bin_df = pd.DataFrame(binary_results)
    opt_df = pd.DataFrame(option_results)

    print("\n[BINARY CLASSIFICATION SUMMARY]")
    print("=" * 80)
    print(bin_df.to_string(index=False))

    print("\n[OPTION SELECTION SUMMARY]")
    print("=" * 80)
    print(opt_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[SAVING]")
    bin_df.to_csv(f'{OUTPUT_DIR}/binary_classification_metrics.csv', index=False)
    print("[OK] binary_classification_metrics.csv saved")

    opt_df.to_csv(f'{OUTPUT_DIR}/option_selection_metrics.csv', index=False)
    print("[OK] option_selection_metrics.csv saved")

    with open(f'{OUTPUT_DIR}/trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)
    print("[OK] trained_models.pkl saved")

    with open(f'{OUTPUT_DIR}/predictions.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)
    print("[OK] predictions.pkl saved")

    best_bin = bin_df.loc[bin_df['Macro F1'].idxmax()]
    best_opt = opt_df.loc[opt_df['Exact Match'].idxmax()]
    print(f"Best Model (Macro F1):   {best_bin['Model']} — F1={best_bin['Macro F1']:.4f}")
    print(f"Best Model (Exact Match): {best_opt['Model']} — Acc={best_opt['Exact Match']:.4f}")

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3 - MODEL A: BINARY CLASSIFIERS (balanced class weights)")
    print("=" * 80)

    X_train_oh, X_val_oh, X_train_lex, X_val_lex, X_train_q, X_val_q, y_train, y_val = load_data()
    trained_models, predictions, binary_results, option_results = train_and_evaluate(
        X_train_oh, X_val_oh, X_train_lex, X_val_lex, X_train_q, X_val_q, y_train, y_val
    )
    print_and_save(trained_models, predictions, binary_results, option_results)
