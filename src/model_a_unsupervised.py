"""
PHASE 4 - MODEL A: UNSUPERVISED/SEMI-SUPERVISED LEARNING
==========================================================================================================

Implements unsupervised clustering and semi-supervised learning approaches for the RACE dataset.

Algorithms:
1. K-Means Clustering (k=2 for binary, k=4 for multi-class) - OPTIMIZED for speed
2. Gaussian Mixture Models (GMM with k=2 and k=4)
3. Label Spreading (semi-supervised, faster than Label Propagation)

Evaluation Metrics:
- Clustering quality: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- Classification accuracy: NMI (Normalized Mutual Information), ARI (Adjusted Rand Index)

Dataset: Combined One-Hot (5000 dims) + Lexical Features (6 dims) = 5006 total features
Sample size: 20,000 binary samples from 5,000 questions (5000 train per split)
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    normalized_mutual_info_score, accuracy_score, f1_score, adjusted_rand_score
)

warnings.filterwarnings('ignore')

# =================================================================================================
# DATA LOADING
# =================================================================================================

def load_preprocessed_data():
    """Load all preprocessed features and labels"""
    print("[LOADING DATA]")
    print("=" * 80)
    
    data_dir = Path("data/processed")
    
    # Load sparse feature matrices
    print("Loading feature matrices...")
    X_train_onehot = sp.load_npz(data_dir / "X_train_onehot.npz")
    X_val_onehot = sp.load_npz(data_dir / "X_val_onehot.npz")
    X_test_onehot = sp.load_npz(data_dir / "X_test_onehot.npz")
    
    # Load lexical features
    print("Loading lexical features...")
    train_lexical = pd.read_csv(data_dir / "train_lexical_features.csv", index_col=0).values
    val_lexical = pd.read_csv(data_dir / "val_lexical_features.csv", index_col=0).values
    test_lexical = pd.read_csv(data_dir / "test_lexical_features.csv", index_col=0).values
    
    # Load labels and metadata
    print("Loading labels...")
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    
    with open(data_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✓ Train One-Hot: {X_train_onehot.shape}")
    print(f"✓ Val One-Hot: {X_val_onehot.shape}")
    print(f"✓ Train Lexical: {train_lexical.shape}")
    print(f"✓ Val Lexical: {val_lexical.shape}")
    print(f"✓ Train labels: {y_train.shape}")
    print(f"✓ Val labels: {y_val.shape}")
    
    return {
        'X_train_onehot': X_train_onehot, 'X_val_onehot': X_val_onehot,
        'X_test_onehot': X_test_onehot,
        'train_lexical': train_lexical, 'val_lexical': val_lexical,
        'test_lexical': test_lexical,
        'y_train': y_train, 'y_val': y_val,
        'metadata': metadata
    }


def combine_features(X_onehot, lexical_features):
    """Combine One-Hot encoding with lexical features"""
    # Convert sparse to dense
    if sp.issparse(X_onehot):
        X_dense = X_onehot.toarray()
    else:
        X_dense = X_onehot
    
    # Concatenate
    X_combined = np.hstack([X_dense, lexical_features])
    return X_combined


# =================================================================================================
# UNSUPERVISED CLUSTERING
# =================================================================================================

def train_kmeans(X_train, X_val, y_val, n_clusters=2):
    """Train K-Means clustering (optimized for speed)"""
    print(f"  Training K-Means (k={n_clusters})...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Use fewer n_init for speed (default is 10, reduce to 3 for faster training)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, verbose=0, max_iter=300)
    y_pred_train = model.fit_predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Evaluate clustering quality
    silhouette = silhouette_score(X_val_scaled, y_pred_val)
    davies_bouldin = davies_bouldin_score(X_val_scaled, y_pred_val)
    calinski = calinski_harabasz_score(X_val_scaled, y_pred_val)
    nmi = normalized_mutual_info_score(y_val, y_pred_val)
    ari = adjusted_rand_score(y_val, y_pred_val)
    
    print(f"    Silhouette Score: {silhouette:.4f}")
    print(f"    Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"    Calinski-Harabasz Index: {calinski:.4f}")
    print(f"    Normalized Mutual Info: {nmi:.4f}")
    print(f"    Adjusted Rand Index: {ari:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'y_pred_val': y_pred_val,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski': calinski,
        'nmi': nmi,
        'ari': ari
    }


def train_gmm(X_train, X_val, y_val, n_components=2):
    """Train Gaussian Mixture Model"""
    print(f"  Training Gaussian Mixture Model (components={n_components})...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    model.fit(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Evaluate clustering quality
    silhouette = silhouette_score(X_val_scaled, y_pred_val)
    davies_bouldin = davies_bouldin_score(X_val_scaled, y_pred_val)
    calinski = calinski_harabasz_score(X_val_scaled, y_pred_val)
    nmi = normalized_mutual_info_score(y_val, y_pred_val)
    ari = adjusted_rand_score(y_val, y_pred_val)
    bic = model.bic(X_val_scaled)
    aic = model.aic(X_val_scaled)
    
    print(f"    Silhouette Score: {silhouette:.4f}")
    print(f"    Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"    Calinski-Harabasz Index: {calinski:.4f}")
    print(f"    Normalized Mutual Info: {nmi:.4f}")
    print(f"    Adjusted Rand Index: {ari:.4f}")
    print(f"    BIC: {bic:.4f}, AIC: {aic:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'y_pred_val': y_pred_val,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski': calinski,
        'nmi': nmi,
        'ari': ari,
        'bic': bic,
        'aic': aic
    }


# =================================================================================================
# SEMI-SUPERVISED LEARNING
# =================================================================================================

def train_label_spreading(X_train, X_val, y_train, y_val, alpha=0.2):
    """
    Train Label Spreading (semi-supervised, faster variant).
    Uses a fraction of labeled data + remaining as unlabeled.
    """
    print(f"  Training Label Spreading (alpha={alpha})...")
    
    # Combine train and val for semi-supervised setting
    X_combined = np.vstack([X_train, X_val])
    
    # Create labels: labeled for train, unlabeled (-1) for val
    n_labeled = len(y_train)
    y_combined = np.hstack([y_train, np.full(len(y_val), -1)])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Label Spreading (faster than Label Propagation, uses transition matrix regularization)
    model = LabelSpreading(kernel='rbf', gamma=0.1, alpha=alpha, n_jobs=-1)
    y_pred = model.fit_predict(X_scaled, y_combined)
    
    # Predictions on validation set
    y_pred_val = y_pred[n_labeled:]
    
    # Evaluate on validation set
    accuracy = accuracy_score(y_val, y_pred_val)
    macro_f1 = f1_score(y_val, y_pred_val, average='macro')
    nmi = normalized_mutual_info_score(y_val, y_pred_val)
    
    print(f"    Accuracy on validation: {accuracy:.4f}")
    print(f"    Macro F1 on validation: {macro_f1:.4f}")
    print(f"    Normalized Mutual Info: {nmi:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'y_pred_val': y_pred_val,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'nmi': nmi
    }


# =================================================================================================
# OPTION SELECTION EVALUATION (For Clustering)
# =================================================================================================

def predict_best_options_from_clusters(y_pred_clusters, metadata):
    """
    For clustering predictions, select best option per question.
    Using cluster membership probability (cluster 1 = answer) for ranking.
    """
    options = metadata['options']
    correct_answers = metadata['answers']
    val_ids = metadata['val_ids']
    
    predictions_per_q = {}
    
    # Group by question ID
    for idx, (option, pred_cluster) in enumerate(zip(options, y_pred_clusters)):
        q_id = val_ids[idx // 4] if idx < len(val_ids) * 4 else None
        if q_id not in predictions_per_q:
            predictions_per_q[q_id] = []
        # Store option and cluster prediction
        predictions_per_q[q_id].append((option, pred_cluster))
    
    # Select best option (cluster=1 has highest score)
    correct_count = 0
    total_count = 0
    
    for q_idx, q_id in enumerate(val_ids):
        if q_id in predictions_per_q:
            options_for_q = predictions_per_q[q_id]
            # Find option with cluster membership closest to 1 (assumed "answer" cluster)
            if len(options_for_q) == 4:
                # For clustering, use cluster ID as proxy (prefer cluster 1 if binary)
                best_option = max(options_for_q, key=lambda x: x[1])[0]
                correct_answer = correct_answers[q_idx]
                
                if best_option == correct_answer:
                    correct_count += 1
                total_count += 1
    
    if total_count > 0:
        accuracy = correct_count / total_count
        return accuracy
    return 0.0


# =================================================================================================
# MAIN EXECUTION
# =================================================================================================

def train_and_evaluate_unsupervised():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("PHASE 4 - MODEL A: UNSUPERVISED/SEMI-SUPERVISED LEARNING")
    print("=" * 80 + "\n")
    
    # Load data
    data = load_preprocessed_data()
    print()
    
    # Combine features
    print("[COMBINING FEATURES]")
    print("=" * 80)
    X_train = combine_features(data['X_train_onehot'], data['train_lexical'])
    X_val = combine_features(data['X_val_onehot'], data['val_lexical'])
    print(f"✓ Train combined features: {X_train.shape}")
    print(f"✓ Val combined features: {X_val.shape}")
    print()
    
    # Dictionary to store results
    results = {}
    
    # K-MEANS CLUSTERING
    print("[K-MEANS CLUSTERING]")
    print("=" * 80)
    
    print("\n1. K-Means (k=2: Binary Classification)")
    kmeans_2 = train_kmeans(X_train, X_val, data['y_val'], n_clusters=2)
    results['kmeans_2'] = kmeans_2
    
    print("\n2. K-Means (k=4: Multi-class)")
    kmeans_4 = train_kmeans(X_train, X_val, data['y_val'], n_clusters=4)
    results['kmeans_4'] = kmeans_4
    print()
    
    # GAUSSIAN MIXTURE MODELS
    print("[GAUSSIAN MIXTURE MODELS]")
    print("=" * 80)
    
    print("\n1. GMM (components=2: Binary)")
    gmm_2 = train_gmm(X_train, X_val, data['y_val'], n_components=2)
    results['gmm_2'] = gmm_2
    
    print("\n2. GMM (components=4: Multi-class)")
    gmm_4 = train_gmm(X_train, X_val, data['y_val'], n_components=4)
    results['gmm_4'] = gmm_4
    print()
    
    # SEMI-SUPERVISED LEARNING
    print("[SEMI-SUPERVISED LEARNING]")
    print("=" * 80)
    
    print("\n1. Label Spreading")
    ls = train_label_spreading(X_train, X_val, data['y_train'], data['y_val'])
    results['label_spreading'] = ls
    print()
    
    # SAVE RESULTS
    print("[SAVING RESULTS]")
    print("=" * 80)
    
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Save clustering results summary
    clustering_results = {
        'kmeans_2': {
            'silhouette': kmeans_2['silhouette'],
            'davies_bouldin': kmeans_2['davies_bouldin'],
            'calinski': kmeans_2['calinski'],
            'nmi': kmeans_2['nmi'],
            'ari': kmeans_2['ari']
        },
        'kmeans_4': {
            'silhouette': kmeans_4['silhouette'],
            'davies_bouldin': kmeans_4['davies_bouldin'],
            'calinski': kmeans_4['calinski'],
            'nmi': kmeans_4['nmi'],
            'ari': kmeans_4['ari']
        },
        'gmm_2': {
            'silhouette': gmm_2['silhouette'],
            'davies_bouldin': gmm_2['davies_bouldin'],
            'calinski': gmm_2['calinski'],
            'nmi': gmm_2['nmi'],
            'ari': gmm_2['ari'],
            'bic': gmm_2['bic'],
            'aic': gmm_2['aic']
        },
        'gmm_4': {
            'silhouette': gmm_4['silhouette'],
            'davies_bouldin': gmm_4['davies_bouldin'],
            'calinski': gmm_4['calinski'],
            'nmi': gmm_4['nmi'],
            'ari': gmm_4['ari'],
            'bic': gmm_4['bic'],
            'aic': gmm_4['aic']
        },
        'label_spreading': {
            'accuracy': ls['accuracy'],
            'macro_f1': ls['macro_f1'],
            'nmi': ls['nmi']
        }
    }
    
    with open(output_dir / "unsupervised_clustering_results.pkl", "wb") as f:
        pickle.dump(clustering_results, f)
    print("✓ Clustering results saved: unsupervised_clustering_results.pkl")
    
    # Save trained models
    models_to_save = {
        'kmeans_2': {'model': kmeans_2['model'], 'scaler': kmeans_2['scaler']},
        'kmeans_4': {'model': kmeans_4['model'], 'scaler': kmeans_4['scaler']},
        'gmm_2': {'model': gmm_2['model'], 'scaler': gmm_2['scaler']},
        'gmm_4': {'model': gmm_4['model'], 'scaler': gmm_4['scaler']},
        'label_spreading': {'model': ls['model'], 'scaler': ls['scaler']}
    }
    
    with open(output_dir / "unsupervised_models.pkl", "wb") as f:
        pickle.dump(models_to_save, f)
    print("✓ Trained models saved: unsupervised_models.pkl")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY - CLUSTERING QUALITY METRICS")
    print("=" * 80)
    
    summary_df = pd.DataFrame({
        'Method': ['K-Means (k=2)', 'K-Means (k=4)', 'GMM (c=2)', 'GMM (c=4)', 'Label Spread.'],
        'Silhouette': [
            kmeans_2['silhouette'],
            kmeans_4['silhouette'],
            gmm_2['silhouette'],
            gmm_4['silhouette'],
            np.nan
        ],
        'Davies-Bouldin': [
            kmeans_2['davies_bouldin'],
            kmeans_4['davies_bouldin'],
            gmm_2['davies_bouldin'],
            gmm_4['davies_bouldin'],
            np.nan
        ],
        'NMI': [
            kmeans_2['nmi'],
            kmeans_4['nmi'],
            gmm_2['nmi'],
            gmm_4['nmi'],
            ls['nmi']
        ],
        'Accuracy': [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            ls['accuracy']
        ]
    })
    
    print(summary_df.to_string(index=False))
    
    # Comparison with supervised models from Phase 3
    print("\n" + "=" * 80)
    print("COMPARISON: UNSUPERVISED vs SUPERVISED (PHASE 3)")
    print("=" * 80)
    print(f"Supervised Best (Naive Bayes): 52.74% option selection accuracy")
    print(f"Semi-Supervised (Label Spread.): {ls['accuracy']*100:.2f}% binary classification accuracy")
    print()
    print(f"✓ Phase 4 training and evaluation complete!")
    print(f"\nOutput directory: {output_dir}")
    print(f"Generated files:")
    print(f"  - unsupervised_clustering_results.pkl (all metrics)")
    print(f"  - unsupervised_models.pkl (trained models)")
    
    return results


if __name__ == "__main__":
    train_and_evaluate_unsupervised()
