"""
PHASE 5 - MODEL A: ENSEMBLE
"""

import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


PROCESSED_DIR = '/content/drive/MyDrive/data/processed'
OUTPUT_DIR    = '/content/drive/MyDrive/models'


def load_data():
    print("\n[LOADING DATA]")
    print("=" * 80)

    X_val_onehot = sp.load_npz(f'{PROCESSED_DIR}/X_val_onehot.npz')
    X_val_tfidf  = sp.load_npz(f'{PROCESSED_DIR}/X_val_tfidf.npz')
    val_lex      = pd.read_csv(f'{PROCESSED_DIR}/val_lexical_features.csv').values
    y_val        = np.load(f'{PROCESSED_DIR}/y_val.npy')

    with open(f'{PROCESSED_DIR}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    with open(f'{OUTPUT_DIR}/trained_models.pkl', 'rb') as f:
        trained_models = pickle.load(f)

    with open(f'{OUTPUT_DIR}/predictions.pkl', 'rb') as f:
        predictions = pickle.load(f)

    print(f"[OK] X_val_onehot: {X_val_onehot.shape}")
    print(f"[OK] y_val: {y_val.shape}")
    print(f"[OK] Models loaded: {list(trained_models.keys())}")

    return X_val_onehot, X_val_tfidf, val_lex, y_val, metadata, trained_models, predictions


def combine_features(X_sparse, X_lex):
    X_lex_sparse = sp.csr_matrix(X_lex)
    return sp.hstack([X_sparse, X_lex_sparse], format='csr')


def evaluate_binary(y_true, y_pred, y_proba, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }


def evaluate_option_selection(y_proba, y_val, metadata, model_name):
    val_answers  = metadata['val_answers']
    n_questions  = len(y_val) // 4

    true_options = [val_answers[i * 4] for i in range(n_questions)]

    pred_options = []
    for q in range(n_questions):
        probs    = y_proba[q * 4: q * 4 + 4]
        best_idx = np.argmax(probs)
        pred_options.append(['A', 'B', 'C', 'D'][best_idx])

    acc = accuracy_score(true_options, pred_options)
    option_map   = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    y_true_n = [option_map[o] for o in true_options]
    y_pred_n = [option_map[o] for o in pred_options]
    f1 = f1_score(y_true_n, y_pred_n, average='macro')

    return {
        'Model': model_name,
        'Option Accuracy': acc,
        'Option Macro F1': f1
    }


def soft_voting(predictions):
    print("\n[SOFT VOTING]")
    print("=" * 80)

    probas = np.vstack([pred[1] for pred in predictions.values()])
    avg_proba = probas.mean(axis=0)
    y_pred    = (avg_proba >= 0.5).astype(int)

    print(classification_report(
        list(predictions.values())[0][0].__class__,
        y_pred,
        target_names=['Non-Answer', 'Answer'],
        digits=4
    ) if False else "")

    return y_pred, avg_proba


def hard_voting(predictions, y_val):
    print("\n[HARD VOTING]")
    print("=" * 80)

    preds = np.vstack([pred[0] for pred in predictions.values()])
    y_pred = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=preds
    )
    avg_proba = np.vstack([pred[1] for pred in predictions.values()]).mean(axis=0)

    return y_pred, avg_proba


def stacking(predictions, y_val, X_val_onehot, val_lex):
    print("\n[STACKING]")
    print("=" * 80)

    # build meta-features from base model probabilities
    meta_X = np.column_stack([pred[1] for pred in predictions.values()])

    # split meta features into train/val halves for stacking
    n = len(y_val)
    half = n // 2

    meta_X_train = meta_X[:half]
    y_meta_train = y_val[:half]
    meta_X_val   = meta_X[half:]
    y_meta_val   = y_val[half:]

    meta_clf = LogisticRegression(max_iter=300, solver='saga', n_jobs=-1, random_state=42)
    meta_clf.fit(meta_X_train, y_meta_train)

    y_proba = meta_clf.predict_proba(meta_X_val)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    # pad first half with soft voting proba to keep same length
    soft_proba = np.vstack([pred[1] for pred in predictions.values()]).mean(axis=0)
    full_proba = np.concatenate([soft_proba[:half], y_proba])
    full_pred  = np.concatenate([(soft_proba[:half] >= 0.5).astype(int), y_pred])

    return meta_clf, full_pred, full_proba, y_meta_val, y_proba


def run_phase5():
    print("\n" + "=" * 80)
    print("PHASE 5 - MODEL A: ENSEMBLE")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_val_onehot, X_val_tfidf, val_lex, y_val, metadata, trained_models, predictions = load_data()

    binary_metrics  = []
    option_metrics  = []

    # individual model results from Phase 3
    # predictions[name] = full predict_proba array of shape (n, 2)
    # extract y_pred from argmax, y_proba from column 1
    base_model_names = ['Logistic Regression', 'SVM', 'Naive Bayes', 'Random Forest']
    valid_predictions = {}
    for name in base_model_names:
        if name not in predictions or predictions[name] is None:
            print(f"[WARN] {name} predictions not found, skipping")
            continue
        proba_2d = predictions[name]          # shape (n, 2)
        y_proba  = proba_2d[:, 1]            # positive class probability
        y_pred   = proba_2d.argmax(axis=1)   # predicted class
        valid_predictions[name] = (y_pred, y_proba)
        binary_metrics.append(evaluate_binary(y_val, y_pred, y_proba, name + ' (Base)'))
        option_metrics.append(evaluate_option_selection(y_proba, y_val, metadata, name + ' (Base)'))

    # soft voting
    print("\n[SOFT VOTING]")
    print("=" * 80)
    sv_proba = np.vstack([p[1] for p in valid_predictions.values()]).mean(axis=0)
    sv_pred  = (sv_proba >= 0.5).astype(int)
    print(classification_report(y_val, sv_pred, target_names=['Non-Answer', 'Answer'], digits=4))
    binary_metrics.append(evaluate_binary(y_val, sv_pred, sv_proba, 'Soft Voting'))
    option_metrics.append(evaluate_option_selection(sv_proba, y_val, metadata, 'Soft Voting'))

    # hard voting
    print("\n[HARD VOTING]")
    print("=" * 80)
    preds_stack = np.vstack([p[0] for p in valid_predictions.values()])
    hv_pred = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=2).argmax(), axis=0, arr=preds_stack
    )
    hv_proba = np.vstack([p[1] for p in valid_predictions.values()]).mean(axis=0)
    print(classification_report(y_val, hv_pred, target_names=['Non-Answer', 'Answer'], digits=4))
    binary_metrics.append(evaluate_binary(y_val, hv_pred, hv_proba, 'Hard Voting'))
    option_metrics.append(evaluate_option_selection(hv_proba, y_val, metadata, 'Hard Voting'))

    # stacking
    print("\n[STACKING]")
    print("=" * 80)
    n    = len(y_val)
    half = n // 2
    meta_X       = np.column_stack([p[1] for p in valid_predictions.values()])
    meta_X_train = meta_X[:half]
    y_meta_train = y_val[:half]
    meta_X_val   = meta_X[half:]
    y_meta_val   = y_val[half:]

    meta_clf = LogisticRegression(max_iter=300, solver='saga', n_jobs=-1, random_state=42)
    meta_clf.fit(meta_X_train, y_meta_train)

    st_proba_half = meta_clf.predict_proba(meta_X_val)[:, 1]
    st_pred_half  = (st_proba_half >= 0.5).astype(int)
    print(classification_report(y_meta_val, st_pred_half, target_names=['Non-Answer', 'Answer'], digits=4))

    st_full_proba = np.concatenate([sv_proba[:half], st_proba_half])
    st_full_pred  = np.concatenate([sv_pred[:half], st_pred_half])
    binary_metrics.append(evaluate_binary(y_val, st_full_pred, st_full_proba, 'Stacking'))
    option_metrics.append(evaluate_option_selection(st_full_proba, y_val, metadata, 'Stacking'))

    # print summary tables
    print("\n[BINARY CLASSIFICATION SUMMARY]")
    print("=" * 80)
    binary_df = pd.DataFrame(binary_metrics)
    print(binary_df.to_string(index=False))

    print("\n[OPTION SELECTION SUMMARY]")
    print("=" * 80)
    option_df = pd.DataFrame(option_metrics)
    print(option_df.to_string(index=False))

    # save
    binary_df.to_csv(f'{OUTPUT_DIR}/ensemble_binary_metrics.csv', index=False)
    option_df.to_csv(f'{OUTPUT_DIR}/ensemble_option_metrics.csv', index=False)

    ensemble_models = {
        'soft_voting_proba': sv_proba,
        'hard_voting_proba': hv_proba,
        'stacking_meta_clf': meta_clf
    }
    with open(f'{OUTPUT_DIR}/ensemble_models.pkl', 'wb') as f:
        pickle.dump(ensemble_models, f)

    print("\n[SAVING]")
    print("=" * 80)
    print("[OK] ensemble_binary_metrics.csv saved")
    print("[OK] ensemble_option_metrics.csv saved")
    print("[OK] ensemble_models.pkl saved")

    best_binary = binary_df.loc[binary_df['Accuracy'].idxmax()]
    best_option = option_df.loc[option_df['Option Accuracy'].idxmax()]

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE!")
    print("=" * 80)
    print(f"Best Binary Model:  {best_binary['Model']} ({best_binary['Accuracy']:.4f})")
    print(f"Best Option Model:  {best_option['Model']} ({best_option['Option Accuracy']:.4f})")


if __name__ == "__main__":
    run_phase5()