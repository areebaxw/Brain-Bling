"""
EVALUATE.PY — Unified Metric Computation
=========================================
Aggregates and reports evaluation metrics for Model A and Model B.
Individual model evaluations are stored in:
  models/model_a/traditional/binary_classification_metrics.csv
  models/model_a/traditional/option_selection_metrics.csv
  models/model_a/traditional/ensemble_binary_metrics.csv
  models/model_a/traditional/ensemble_option_metrics.csv
  models/model_a/traditional/phase4_comparison.csv   (unsupervised)
  models/model_b/traditional/distractor_results.csv
  models/model_b/traditional/hint_results.csv
"""

import os
import pandas as pd
import numpy as np

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_A_DIR = os.path.join(MODELS_DIR, "model_a", "traditional")
MODEL_B_DIR = os.path.join(MODELS_DIR, "model_b", "traditional")


def load_csv(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n[{label}]")
        print(df.to_string(index=False))
        return df
    else:
        print(f"\n[{label}] — file not found: {path}")
        return None


def evaluate_model_a():
    print("=" * 80)
    print("MODEL A — ANSWER VERIFIER EVALUATION")
    print("=" * 80)

    load_csv(f"{MODEL_A_DIR}/binary_classification_metrics.csv",
             "Binary Classification (Accuracy / Macro F1 / ROC-AUC)")

    load_csv(f"{MODEL_A_DIR}/option_selection_metrics.csv",
             "Option Selection (Exact Match / Option Macro F1)")

    load_csv(f"{MODEL_A_DIR}/ensemble_binary_metrics.csv",
             "Ensemble — Binary (Soft Voting / Hard Voting / Stacking)")

    load_csv(f"{MODEL_A_DIR}/ensemble_option_metrics.csv",
             "Ensemble — Option Accuracy")

    load_csv(f"{MODEL_A_DIR}/phase4_comparison.csv",
             "Unsupervised / Semi-Supervised (Purity / Silhouette / F1)")


def evaluate_model_b():
    print("\n" + "=" * 80)
    print("MODEL B — DISTRACTOR & HINT GENERATOR EVALUATION")
    print("=" * 80)

    dist_df = load_csv(f"{MODEL_B_DIR}/distractor_results.csv",
                       "Distractor Ranker (per-question P / R / F1)")
    if dist_df is not None:
        print(f"\n  Aggregate Distractor Metrics:")
        print(f"  Precision : {dist_df['ml_precision'].mean():.4f}")
        print(f"  Recall    : {dist_df['ml_recall'].mean():.4f}")
        print(f"  F1        : {dist_df['ml_f1'].mean():.4f}")
        partial = (dist_df['ml_f1'] > 0.1).mean()
        print(f"  Partial-Match Accuracy (F1>0.1): {partial:.4f}")

    conf_path = f"{MODEL_B_DIR}/distractor_confusion_matrix.csv"
    if os.path.exists(conf_path):
        load_csv(conf_path, "Distractor Confusion Matrix (TP/FP/FN)")

    load_csv(f"{MODEL_B_DIR}/hint_results.csv",
             "Hint Model (Accuracy / Precision@3)")


def summarise():
    print("\n" + "=" * 80)
    print("SUMMARY — BEST MODELS")
    print("=" * 80)

    em_path = f"{MODEL_A_DIR}/option_selection_metrics.csv"
    if os.path.exists(em_path):
        df = pd.read_csv(em_path)
        best = df.loc[df["Exact Match"].idxmax()]
        print(f"  Best Exact Match (Model A base) : {best['Model']} — {best['Exact Match']:.4f}")

    ens_path = f"{MODEL_A_DIR}/ensemble_option_metrics.csv"
    if os.path.exists(ens_path):
        df = pd.read_csv(ens_path)
        col = "Option Accuracy"
        best = df.loc[df[col].idxmax()]
        print(f"  Best Exact Match (Ensemble)     : {best['Model']} — {best[col]:.4f}")

    hint_path = f"{MODEL_B_DIR}/hint_results.csv"
    if os.path.exists(hint_path):
        df = pd.read_csv(hint_path)
        if "precision_at_3" in df.columns:
            print(f"  Hint Precision@3                : {df['precision_at_3'].iloc[0]:.4f}")
        print(f"  Hint Accuracy                   : {df['accuracy'].iloc[0]:.4f}")


if __name__ == "__main__":
    evaluate_model_a()
    evaluate_model_b()
    summarise()
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
