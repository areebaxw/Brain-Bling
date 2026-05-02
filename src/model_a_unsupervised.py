"""
PHASE 4 - MODEL A: UNSUPERVISED & SEMI-SUPERVISED LEARNING
===========================================================
Implements:
- K-Means Clustering
- Gaussian Mixture Models (GMM)
- Label Propagation (Semi-Supervised)

Evaluation:
- Clustering Purity
- Silhouette Score
- Semi-Supervised F1
- Comparison table vs supervised baselines
"""

import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (
    accuracy_score, f1_score, silhouette_score,
    classification_report
)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(processed_dir):
    print("\n[LOADING DATA]")
    print("=" * 80)

    X_train = sp.load_npz(f'{processed_dir}/X_train_onehot.npz')
    X_val   = sp.load_npz(f'{processed_dir}/X_val_onehot.npz')

    y_train = np.load(f'{processed_dir}/y_train.npy')
    y_val   = np.load(f'{processed_dir}/y_val.npy')

    print(f"[OK] X_train: {X_train.shape}")
    print(f"[OK] X_val:   {X_val.shape}")
    print(f"[OK] y_train: {y_train.shape}")
    print(f"[OK] y_val:   {y_val.shape}")

    return X_train, X_val, y_train, y_val


# ============================================================================
# 2. DIMENSIONALITY REDUCTION (required before clustering)
# ============================================================================

def reduce_dimensions(X_train, X_val, n_components=100, max_rows=50000):
    """
    TruncatedSVD reduces sparse One-Hot matrix to dense low-dim representation.
    Subsample train rows to fit Colab RAM.
    """
    print("\n[DIMENSIONALITY REDUCTION]")
    print("=" * 80)

    # subsample for memory
    if X_train.shape[0] > max_rows:
        idx = np.random.choice(X_train.shape[0], max_rows, replace=False)
        X_train_sub = X_train[idx]
        print(f"Subsampled train to {max_rows} rows for SVD")
    else:
        X_train_sub = X_train
        idx = np.arange(X_train.shape[0])

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_sub)
    X_val_reduced   = svd.transform(X_val)

    # L2 normalize for cosine-like distance
    X_train_reduced = normalize(X_train_reduced)
    X_val_reduced   = normalize(X_val_reduced)

    print(f"[OK] Train reduced: {X_train_reduced.shape}")
    print(f"[OK] Val reduced:   {X_val_reduced.shape}")
    print(f"Variance explained: {svd.explained_variance_ratio_.sum():.4f}")

    return X_train_reduced, X_val_reduced, idx


# ============================================================================
# 3. CLUSTERING PURITY
# ============================================================================

def clustering_purity(y_true, cluster_labels):
    """
    For each cluster, find the majority class.
    Purity = fraction of samples in their majority class.
    """
    n = len(y_true)
    total_correct = 0
    clusters = np.unique(cluster_labels)

    for c in clusters:
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        labels_in_cluster = y_true[mask]
        majority_count = np.bincount(labels_in_cluster).max()
        total_correct += majority_count

    return total_correct / n


# ============================================================================
# 4. K-MEANS CLUSTERING
# ============================================================================

def run_kmeans(X_train, y_train, X_val, y_val):
    print("\n[K-MEANS CLUSTERING]")
    print("=" * 80)

    # 2 clusters — Answer (1) and Non-Answer (0)
    print("Fitting MiniBatchKMeans (k=2)...")
    kmeans = MiniBatchKMeans(
        n_clusters=2,
        random_state=42,
        batch_size=5000,
        max_iter=100,
        n_init=3
    )
    kmeans.fit(X_train)

    # cluster labels on val
    val_clusters = kmeans.predict(X_val)

    # map cluster to class (0 or 1) by majority vote
    cluster_to_label = {}
    for c in [0, 1]:
        mask = val_clusters == c
        if mask.sum() == 0:
            cluster_to_label[c] = 0
            continue
        majority = np.bincount(y_val[mask]).argmax()
        cluster_to_label[c] = majority

    y_pred = np.array([cluster_to_label[c] for c in val_clusters])

    purity    = clustering_purity(y_val, val_clusters)
    sil_score = silhouette_score(X_val[:5000], val_clusters[:5000], sample_size=2000)
    acc       = accuracy_score(y_val, y_pred)
    f1        = f1_score(y_val, y_pred, average='macro')

    print(f"Clustering Purity:  {purity:.4f}")
    print(f"Silhouette Score:   {sil_score:.4f}")
    print(f"Mapped Accuracy:    {acc:.4f}")
    print(f"Mapped Macro F1:    {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=['Non-Answer', 'Answer'], digits=4))

    return kmeans, {
        'Model': 'K-Means (k=2)',
        'Purity': purity,
        'Silhouette': sil_score,
        'Accuracy': acc,
        'Macro F1': f1
    }


# ============================================================================
# 5. GAUSSIAN MIXTURE MODEL
# ============================================================================

def run_gmm(X_train, y_train, X_val, y_val):
    print("\n[GAUSSIAN MIXTURE MODEL]")
    print("=" * 80)

    print("Fitting GMM (n_components=2)...")
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='diag',
        max_iter=100,
        random_state=42,
        verbose=0
    )
    gmm.fit(X_train)

    val_clusters = gmm.predict(X_val)

    # map component to class by majority vote
    cluster_to_label = {}
    for c in [0, 1]:
        mask = val_clusters == c
        if mask.sum() == 0:
            cluster_to_label[c] = 0
            continue
        majority = np.bincount(y_val[mask]).argmax()
        cluster_to_label[c] = majority

    y_pred = np.array([cluster_to_label[c] for c in val_clusters])

    purity    = clustering_purity(y_val, val_clusters)
    sil_score = silhouette_score(X_val[:5000], val_clusters[:5000], sample_size=2000)
    acc       = accuracy_score(y_val, y_pred)
    f1        = f1_score(y_val, y_pred, average='macro')

    print(f"Clustering Purity:  {purity:.4f}")
    print(f"Silhouette Score:   {sil_score:.4f}")
    print(f"Mapped Accuracy:    {acc:.4f}")
    print(f"Mapped Macro F1:    {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=['Non-Answer', 'Answer'], digits=4))

    return gmm, {
        'Model': 'GMM (n=2, diag)',
        'Purity': purity,
        'Silhouette': sil_score,
        'Accuracy': acc,
        'Macro F1': f1
    }


# ============================================================================
# 6. LABEL PROPAGATION (SEMI-SUPERVISED)
# ============================================================================

def run_label_propagation(X_train, y_train, X_val, y_val, labeled_fraction=0.1):
    """
    Label Propagation: use small labeled set, rest unlabeled (-1).
    """
    print("\n[LABEL PROPAGATION - SEMI-SUPERVISED]")
    print("=" * 80)

    n = len(y_train)
    n_labeled = int(n * labeled_fraction)

    # randomly pick labeled samples
    rng = np.random.default_rng(42)
    labeled_idx = rng.choice(n, n_labeled, replace=False)

    # build semi-supervised labels: -1 = unlabeled
    y_semi = np.full(n, -1, dtype=int)
    y_semi[labeled_idx] = y_train[labeled_idx]

    print(f"Total train samples:   {n}")
    print(f"Labeled samples:       {n_labeled} ({labeled_fraction*100:.0f}%)")
    print(f"Unlabeled samples:     {n - n_labeled}")

    print("Fitting LabelSpreading...")
    # LabelSpreading is more robust than LabelPropagation on larger data
    model = LabelSpreading(
        kernel='knn',
        n_neighbors=7,
        alpha=0.2,
        max_iter=30,
        n_jobs=-1
    )
    model.fit(X_train, y_semi)

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average='macro')

    print(f"Semi-Supervised Accuracy: {acc:.4f}")
    print(f"Semi-Supervised Macro F1: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=['Non-Answer', 'Answer'], digits=4))

    return model, {
        'Model': f'Label Spreading ({labeled_fraction*100:.0f}% labeled)',
        'Purity': '-',
        'Silhouette': '-',
        'Accuracy': acc,
        'Macro F1': f1
    }


# ============================================================================
# 7. COMPARISON TABLE
# ============================================================================

def build_comparison_table(unsupervised_results, supervised_metrics_path):
    print("\n[COMPARISON TABLE]")
    print("=" * 80)

    # load supervised results from Phase 3
    if os.path.exists(supervised_metrics_path):
        sup_df = pd.read_csv(supervised_metrics_path)
        sup_rows = []
        for _, row in sup_df.iterrows():
            sup_rows.append({
                'Model': row['Model'] + ' (Supervised)',
                'Purity': '-',
                'Silhouette': '-',
                'Accuracy': round(row['Accuracy'], 4),
                'Macro F1': round(row['Macro F1'], 4)
            })
    else:
        sup_rows = []
        print("Warning: supervised metrics not found, skipping comparison")

    all_rows = sup_rows + unsupervised_results
    df = pd.DataFrame(all_rows)

    print("\nFull Comparison — Supervised vs Unsupervised/Semi-Supervised:")
    print(df.to_string(index=False))

    return df


# ============================================================================
# 8. MAIN
# ============================================================================

def run_phase4(
    processed_dir='/content/drive/MyDrive/data/processed',
    output_dir='models',
    supervised_metrics_path='models/binary_classification_metrics.csv'
):
    print("\n" + "=" * 80)
    print("PHASE 4 - UNSUPERVISED & SEMI-SUPERVISED LEARNING")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # load
    X_train, X_val, y_train, y_val = load_data(processed_dir)

    # reduce dimensions (required for clustering)
    X_train_r, X_val_r, sub_idx = reduce_dimensions(
        X_train, X_val, n_components=100, max_rows=50000
    )
    y_train_sub = y_train[sub_idx]

    unsupervised_results = []

    # K-Means
    kmeans, km_metrics = run_kmeans(X_train_r, y_train_sub, X_val_r, y_val)
    unsupervised_results.append(km_metrics)

    # GMM
    gmm, gmm_metrics = run_gmm(X_train_r, y_train_sub, X_val_r, y_val)
    unsupervised_results.append(gmm_metrics)

    # Label Propagation — use smaller subset (RAM constraint)
    max_lp = 20000
    lp_idx = np.random.choice(len(X_train_r), min(max_lp, len(X_train_r)), replace=False)
    X_lp = X_train_r[lp_idx]
    y_lp = y_train_sub[lp_idx]
    lp_val_idx = np.random.choice(len(X_val_r), min(10000, len(X_val_r)), replace=False)
    X_lp_val = X_val_r[lp_val_idx]
    y_lp_val = y_val[lp_val_idx]

    lp_model, lp_metrics = run_label_propagation(X_lp, y_lp, X_lp_val, y_lp_val, labeled_fraction=0.1)
    unsupervised_results.append(lp_metrics)

    # comparison table
    comparison_df = build_comparison_table(unsupervised_results, supervised_metrics_path)

    # save
    comparison_df.to_csv(f'{output_dir}/phase4_comparison.csv', index=False)

    models_to_save = {
        'kmeans': kmeans,
        'gmm': gmm,
        'label_spreading': lp_model
    }
    with open(f'{output_dir}/phase4_models.pkl', 'wb') as f:
        pickle.dump(models_to_save, f)

    print("\n[SAVING]")
    print("=" * 80)
    print("[OK] phase4_comparison.csv saved")
    print("[OK] phase4_models.pkl saved")

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE!")
    print("=" * 80)

    return comparison_df


if __name__ == "__main__":
    run_phase4(
        processed_dir='/content/drive/MyDrive/data/processed',
        output_dir='models',
        supervised_metrics_path='models/binary_classification_metrics.csv'
    )