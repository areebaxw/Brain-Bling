"""
PHASE 3 - MODEL A: TRADITIONAL ML CLASSIFIERS
==============================================
Train and evaluate multiple traditional ML models:
- Logistic Regression (answer verification)
- SVM (answer verification & ranking)
- Naive Bayes (question type classification)
- Random Forest (difficulty estimation)

Also includes template-based question generation and ML-based ranking:
- Generate question variants using fixed templates
- Rank generated variants with model confidence/margin scores
- Output predicted best option (A/B/C/D)

Features: One-Hot Encoding + Lexical Features
Evaluation: Accuracy, Macro F1, Confusion Matrix
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
from pathlib import Path
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================

def load_preprocessed_data(processed_dir='data/processed'):
    """
    Load all preprocessed feature matrices and labels
    """
    print("\n[LOADING DATA]")
    print("=" * 80)
    
    # Load sparse matrices
    print("Loading feature matrices...")
    X_train_onehot = sp.load_npz(f'{processed_dir}/X_train_onehot.npz')
    X_val_onehot = sp.load_npz(f'{processed_dir}/X_val_onehot.npz')
    
    X_train_tfidf = sp.load_npz(f'{processed_dir}/X_train_tfidf.npz')
    X_val_tfidf = sp.load_npz(f'{processed_dir}/X_val_tfidf.npz')
    
    # Load lexical features
    print("Loading lexical features...")
    train_lex = pd.read_csv(f'{processed_dir}/train_lexical_features.csv').values
    val_lex = pd.read_csv(f'{processed_dir}/val_lexical_features.csv').values
    
    # Load labels
    print("Loading labels...")
    y_train = np.load(f'{processed_dir}/y_train.npy')
    y_val = np.load(f'{processed_dir}/y_val.npy')
    
    # Load metadata
    print("Loading metadata...")
    with open(f'{processed_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"[OK] Train One-Hot: {X_train_onehot.shape}")
    print(f"[OK] Val One-Hot: {X_val_onehot.shape}")
    print(f"[OK] Train Lexical: {train_lex.shape}")
    print(f"[OK] Val Lexical: {val_lex.shape}")
    print(f"[OK] Train labels: {y_train.shape}")
    print(f"[OK] Val labels: {y_val.shape}")
    
    return {
        'X_train_onehot': X_train_onehot,
        'X_val_onehot': X_val_onehot,
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'train_lex': train_lex,
        'val_lex': val_lex,
        'y_train': y_train,
        'y_val': y_val,
        'metadata': metadata
    }


# ============================================================================
# 2. COMBINE FEATURES
# ============================================================================

def combine_features(X_dense, X_lex, use_tfidf=False, X_tfidf=None):
    """
    Combine One-Hot (or TF-IDF) features with lexical features
    Keep sparse representation when possible.
    """
    if sp.issparse(X_dense):
        X_lex_sparse = sp.csr_matrix(X_lex)
        return sp.hstack([X_dense, X_lex_sparse], format='csr')
    return np.hstack([X_dense, X_lex])


def balance_binary_samples(X_train, y_train, random_state=42):
    """
    Oversample minority class for better recall on the positive ("Answer") class.
    """
    y_train = np.array(y_train)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_train, y_train
    if len(pos_idx) == len(neg_idx):
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    if len(pos_idx) < len(neg_idx):
        extra_idx = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
    else:
        extra_idx = rng.choice(neg_idx, size=len(pos_idx) - len(neg_idx), replace=True)

    if sp.issparse(X_train):
        X_extra = X_train[extra_idx]
        X_bal = sp.vstack([X_train, X_extra], format='csr')
    else:
        X_extra = X_train[extra_idx]
        X_bal = np.vstack([X_train, X_extra])

    y_bal = np.concatenate([y_train, y_train[extra_idx]])
    return X_bal, y_bal


def optimize_threshold(y_true, y_pred_proba):
    """
    Tune threshold to improve macro-F1 and recall on validation.
    """
    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_pred = (y_pred_proba >= 0.5).astype(int)

    for threshold in np.arange(0.20, 0.81, 0.02):
        y_pred = (y_pred_proba >= threshold).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)
            best_pred = y_pred

    return best_pred, best_threshold


# ============================================================================
# 3. QUESTION GENERATION + RANKING HELPERS
# ============================================================================

def clean_text_for_generation(text):
    """
    Match the preprocessing normalization used for model training.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_lexical_features_for_generated_question(article, question, options):
    """
    Recreate lexical features for 4 options of one question.
    Returns shape: (4, 7)
    """
    article_words = set(article.split())
    question_words = set(question.split())
    article_len = len(article.split())

    rows = []
    for option_text in options:
        option_words = set(option_text.split())

        # 1. Jaccard Similarity
        union_words = question_words.union(option_words)
        if union_words:
            jaccard = len(question_words.intersection(option_words)) / len(union_words)
        else:
            jaccard = 0.0

        # 2. Word density
        if option_words:
            word_density = len(question_words.intersection(option_words)) / len(option_words)
        else:
            word_density = 0.0

        # 3. Option length ratio
        option_len = len(option_text.split())
        if article_len > 0:
            length_ratio = option_len / article_len
        else:
            length_ratio = 0.0

        # 4. Capital density
        capital_count = sum(1 for c in option_text if c.isupper())
        capital_density = capital_count / max(len(option_text), 1)

        # 5. Number presence
        has_numbers = 1 if any(c.isdigit() for c in option_text) else 0

        # 6. Common words with article
        common_word_count = len(option_words.intersection(article_words))

        # 7. Average word length
        if option_words:
            avg_word_length = float(np.mean([len(w) for w in option_words]))
        else:
            avg_word_length = 0.0

        rows.append([
            jaccard,
            word_density,
            length_ratio,
            capital_density,
            has_numbers,
            common_word_count,
            avg_word_length
        ])

    return np.array(rows, dtype=float)


def generate_question_templates(original_question):
    """
    Generate template-based question variants from an existing question.
    """
    if not isinstance(original_question, str):
        original_question = ""

    q = re.sub(r'\s+', ' ', original_question).strip()
    if not q:
        return ["According to the passage, what is true?"]

    q_stem = q.rstrip(" ?")
    if q_stem:
        lower_stem = q_stem[0].lower() + q_stem[1:]
    else:
        lower_stem = q_stem

    candidates = [
        q if q.endswith("?") else f"{q}?",
        f"According to the passage, {lower_stem}?",
        f"Based on the article, {lower_stem}?",
        f"From the passage, which option best answers: {lower_stem}?",
        f"Which choice is correct according to the passage: {lower_stem}?"
    ]

    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate.strip().lower()
        if key and key not in seen:
            deduped.append(candidate.strip())
            seen.add(key)

    return deduped


def score_options_with_trained_models(article, question_text, options, trained_models, onehot_vectorizer):
    """
    Score A/B/C/D options for a (article, generated_question, options) tuple.
    Returns per-model probabilities and ensemble probabilities.
    """
    cleaned_article = clean_text_for_generation(article)
    cleaned_question = clean_text_for_generation(question_text)
    cleaned_options = [clean_text_for_generation(opt) for opt in options]

    combined_texts = [
        f"{cleaned_article} {cleaned_question} {opt}".strip()
        for opt in cleaned_options
    ]

    X_onehot_sparse = onehot_vectorizer.transform(combined_texts)
    X_onehot_dense = X_onehot_sparse.toarray()
    X_lex = extract_lexical_features_for_generated_question(
        cleaned_article, cleaned_question, cleaned_options
    )
    X_combined = np.hstack([X_onehot_dense, X_lex])

    model_scores = {}

    # Logistic Regression (combined features)
    lr_model = trained_models.get('LogisticRegression')
    if lr_model is not None:
        model_scores['LogisticRegression'] = lr_model.predict_proba(X_combined)[:, 1]

    # SVM (combined + scaler)
    svm_bundle = trained_models.get('SVM')
    if svm_bundle is not None:
        svm_model = svm_bundle['model']
        svm_scaler = svm_bundle['scaler']
        X_scaled = svm_scaler.transform(X_combined)
        svm_decision = svm_model.decision_function(X_scaled)
        model_scores['SVM'] = 1 / (1 + np.exp(-svm_decision))

    # Naive Bayes (one-hot only)
    nb_model = trained_models.get('NaiveBayes')
    if nb_model is not None:
        model_scores['NaiveBayes'] = nb_model.predict_proba(X_onehot_dense)[:, 1]

    # Random Forest (one-hot only)
    rf_model = trained_models.get('RandomForest')
    if rf_model is not None:
        model_scores['RandomForest'] = rf_model.predict_proba(X_onehot_dense)[:, 1]

    if not model_scores:
        raise ValueError("No trained models available to score generated questions.")

    stacked_scores = np.vstack([scores for scores in model_scores.values()])
    ensemble_scores = np.mean(stacked_scores, axis=0)

    return model_scores, ensemble_scores


def rank_generated_questions(row, trained_models, onehot_vectorizer):
    """
    Generate template variants and rank them via ensemble confidence.
    """
    option_labels = ['A', 'B', 'C', 'D']
    option_texts = [row[label] for label in option_labels]
    template_questions = generate_question_templates(row['question'])

    ranked = []
    for generated_question in template_questions:
        _, ensemble_scores = score_options_with_trained_models(
            row['article'],
            generated_question,
            option_texts,
            trained_models,
            onehot_vectorizer
        )

        best_idx = int(np.argmax(ensemble_scores))
        sorted_scores = np.sort(ensemble_scores)
        top_score = float(sorted_scores[-1])
        second_score = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
        margin = top_score - second_score

        ranked.append({
            'generated_question': generated_question,
            'predicted_answer': option_labels[best_idx],
            'confidence': top_score,
            'margin': margin,
            'rank_score': (0.7 * top_score) + (0.3 * margin)
        })

    ranked.sort(key=lambda x: x['rank_score'], reverse=True)
    return ranked


# ============================================================================
# 4. TRAIN MODELS
# ============================================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression classifier"""
    print("\n  Training Logistic Regression...")
    X_train_bal, y_train_bal = balance_binary_samples(X_train, y_train, random_state=42)

    model = LogisticRegression(
        max_iter=1800,
        random_state=42,
        solver='saga',
        n_jobs=-1,
        verbose=0,
        class_weight='balanced'
    )

    model.fit(X_train_bal, y_train_bal)

    # Predictions + threshold tuning
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred, threshold = optimize_threshold(y_val, y_pred_proba)
    print(f"  [Threshold tuned] Logistic Regression threshold={threshold:.2f}")

    return model, y_pred, y_pred_proba


def train_svm(X_train, y_train, X_val, y_val):
    """Train Linear SVM classifier (faster alternative to RBF)"""
    print("  Training Linear SVM...")
    X_train_bal, y_train_bal = balance_binary_samples(X_train, y_train, random_state=42)

    # Standardize features for sparse-safe SVM
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val)

    # Use LinearSVC for faster training
    model = LinearSVC(
        C=1.5,
        max_iter=2500,
        random_state=42,
        verbose=0,
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train_bal)

    # Predictions + threshold tuning
    # Use decision function for probability estimates
    y_decision = model.decision_function(X_val_scaled)
    # Convert decision function to probability (sigmoid)
    y_pred_proba = 1 / (1 + np.exp(-y_decision))
    y_pred, threshold = optimize_threshold(y_val, y_pred_proba)
    print(f"  [Threshold tuned] SVM threshold={threshold:.2f}")

    return {'model': model, 'scaler': scaler}, y_pred, y_pred_proba


def train_naive_bayes(X_train, y_train, X_val, y_val):
    """Train Gaussian Naive Bayes classifier"""
    print("  Training Naive Bayes...")
    X_train_bal, y_train_bal = balance_binary_samples(X_train, y_train, random_state=42)

    # Use MultinomialNB for count features (better for One-Hot)
    model = MultinomialNB(alpha=0.01)

    model.fit(X_train_bal, y_train_bal)

    # Predictions + threshold tuning
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred, threshold = optimize_threshold(y_val, y_pred_proba)
    print(f"  [Threshold tuned] Naive Bayes threshold={threshold:.2f}")

    return model, y_pred, y_pred_proba


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier"""
    print("  Training Random Forest...")

    # RandomForest needs dense arrays; project sparse features to top frequent columns.
    if sp.issparse(X_train):
        col_scores = np.asarray(X_train.sum(axis=0)).ravel()
        top_k = min(1000, X_train.shape[1])
        top_indices = np.argsort(col_scores)[-top_k:]
        X_train_dense = X_train[:, top_indices].toarray()
        X_val_dense = X_val[:, top_indices].toarray()
    else:
        X_train_dense = X_train
        X_val_dense = X_val

    X_train_dense, y_train_bal = balance_binary_samples(X_train_dense, y_train, random_state=42)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        class_weight='balanced'
    )

    model.fit(X_train_dense, y_train_bal)

    # Predictions + threshold tuning
    y_pred_proba = model.predict_proba(X_val_dense)[:, 1]
    y_pred, threshold = optimize_threshold(y_val, y_pred_proba)
    print(f"  [Threshold tuned] Random Forest threshold={threshold:.2f}")

    return model, y_pred, y_pred_proba


# ============================================================================
# 4. EVALUATE MODELS
# ============================================================================

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Compute comprehensive evaluation metrics
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    
    return metrics


def print_model_report(y_true, y_pred, model_name):
    """Print detailed classification report"""
    print(f"\n  Classification Report - {model_name}:")
    print("  " + "=" * 70)
    print(classification_report(
        y_true, y_pred,
        target_names=['Non-Answer', 'Answer'],
        digits=4
    ))


# ============================================================================
# 5. PREDICT BEST OPTIONS
# ============================================================================

def predict_best_options(models_predictions, n_questions):
    """
    For each question, select the option (A/B/C/D) with highest average probability
    across all models (ensemble voting)
    """
    final_predictions = []
    
    for q_idx in range(n_questions):
        start_idx = q_idx * 4
        end_idx = start_idx + 4
        
        # Collect predictions from all models for this question's 4 options
        option_scores = np.zeros(4)
        
        for model_name, (y_pred, y_pred_proba) in models_predictions.items():
            # Use probability scores for ranking
            for opt_idx in range(4):
                sample_idx = start_idx + opt_idx
                option_scores[opt_idx] += y_pred_proba[sample_idx]
        
        # Average scores across models
        option_scores /= len(models_predictions)
        
        # Select option with highest score
        best_option_idx = np.argmax(option_scores)
        best_option = ['A', 'B', 'C', 'D'][best_option_idx]
        
        final_predictions.append(best_option)
    
    return final_predictions


# ============================================================================
# 6. EVALUATE ON ORIGINAL TASK (OPTION SELECTION)
# ============================================================================

def evaluate_option_selection(y_pred_options, y_true_options, model_name):
    """
    Evaluate accuracy on the original task (select correct option from A/B/C/D)
    """
    accuracy = accuracy_score(y_true_options, y_pred_options)
    
    # Create mapping for confusion matrix
    option_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    y_true_numeric = [option_map[opt] for opt in y_true_options]
    y_pred_numeric = [option_map[opt] for opt in y_pred_options]
    
    f1_macro = f1_score(y_true_numeric, y_pred_numeric, average='macro')
    
    return {
        'Model': model_name,
        'Option Selection Accuracy': accuracy,
        'Option Selection F1 (Macro)': f1_macro
    }


# ============================================================================
# 7. SAVE PREDICTED OPTIONS
# ============================================================================

def save_predicted_options_table(
    option_predictions_by_model,
    true_options,
    metadata,
    output_path
):
    """
    Save per-question predicted options from each model and ensemble.
    """
    n_questions = len(true_options)
    rows = []

    val_ids = metadata.get('val_ids', [])

    for q_idx in range(n_questions):
        sample_idx = q_idx * 4
        question_id = val_ids[sample_idx] if sample_idx < len(val_ids) else q_idx

        row = {
            'question_index': q_idx,
            'question_id': question_id,
            'true_answer': true_options[q_idx]
        }

        for model_name, predictions in option_predictions_by_model.items():
            row[f'pred_{model_name}'] = predictions[q_idx]

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_path, index=False)
    return pred_df


# ============================================================================
# 8. TEMPLATE QUESTION GENERATION + ML RANKING
# ============================================================================

def generate_and_rank_template_questions(
    processed_dir,
    output_dir,
    trained_models,
    max_questions=200
):
    """
    Generate template-based question variants and rank them using trained models.
    """
    print("\n[TEMPLATE QUESTION GENERATION + RANKING]")
    print("=" * 80)

    val_clean_path = f'{processed_dir}/val_clean.csv'
    onehot_vectorizer_path = f'{processed_dir}/onehot_vectorizer.pkl'

    if not os.path.exists(val_clean_path):
        print(f"Warning: {val_clean_path} not found. Skipping generation phase.")
        return None
    if not os.path.exists(onehot_vectorizer_path):
        print(f"Warning: {onehot_vectorizer_path} not found. Skipping generation phase.")
        return None

    print("Loading validation dataset and vectorizer...")
    val_df = pd.read_csv(val_clean_path)
    with open(onehot_vectorizer_path, 'rb') as f:
        onehot_vectorizer = pickle.load(f)

    required_cols = {'article', 'question', 'A', 'B', 'C', 'D'}
    if not required_cols.issubset(set(val_df.columns)):
        print("Warning: validation data missing required columns. Skipping generation phase.")
        return None

    n_total = len(val_df)
    n_use = min(max_questions, n_total)
    print(f"Generating template questions for {n_use} validation samples...")

    summary_rows = []
    candidate_rows = []

    for idx in range(n_use):
        row = val_df.iloc[idx]
        ranked = rank_generated_questions(row, trained_models, onehot_vectorizer)
        best = ranked[0]

        true_answer = row['answer'] if 'answer' in val_df.columns else None
        is_correct = (best['predicted_answer'] == true_answer) if true_answer is not None else None

        summary_rows.append({
            'sample_index': idx,
            'question_id': row['id'] if 'id' in val_df.columns else idx,
            'original_question': row['question'],
            'selected_generated_question': best['generated_question'],
            'predicted_answer': best['predicted_answer'],
            'true_answer': true_answer,
            'is_correct': is_correct,
            'confidence': best['confidence'],
            'margin': best['margin'],
            'rank_score': best['rank_score']
        })

        for rank_idx, candidate in enumerate(ranked, start=1):
            candidate_rows.append({
                'sample_index': idx,
                'question_id': row['id'] if 'id' in val_df.columns else idx,
                'rank': rank_idx,
                'generated_question': candidate['generated_question'],
                'predicted_answer': candidate['predicted_answer'],
                'confidence': candidate['confidence'],
                'margin': candidate['margin'],
                'rank_score': candidate['rank_score']
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_use:
            print(f"  Processed {idx + 1}/{n_use}")

    summary_df = pd.DataFrame(summary_rows)
    candidates_df = pd.DataFrame(candidate_rows)

    summary_path = f'{output_dir}/generated_questions_ranked.csv'
    candidates_path = f'{output_dir}/generated_question_candidates.csv'
    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {candidates_path}")

    if 'is_correct' in summary_df.columns and summary_df['is_correct'].notna().any():
        gen_accuracy = summary_df['is_correct'].mean()
        print(f"Generated Question Selection Accuracy (A/B/C/D): {gen_accuracy:.4f}")
    else:
        gen_accuracy = None

    return {
        'summary': summary_df,
        'candidates': candidates_df,
        'accuracy': gen_accuracy
    }


# ============================================================================
# 9. MAIN TRAINING PIPELINE
# ============================================================================

def train_and_evaluate(
    processed_dir='data/processed',
    output_dir='models',
    run_generation=False,
    generation_max_questions=200
):
    """
    Main training pipeline:
    1. Load preprocessed data
    2. Combine features
    3. Train multiple models
    4. Evaluate on validation set
    5. Generate comparison table
    6. Save trained models
    """
    
    print("\n" + "=" * 80)
    print("PHASE 3 - MODEL A: TRADITIONAL ML CLASSIFIERS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== LOAD DATA ==========
    data = load_preprocessed_data(processed_dir)
    
    X_train_onehot = data['X_train_onehot']
    X_val_onehot = data['X_val_onehot']
    X_train_tfidf = data['X_train_tfidf']
    X_val_tfidf = data['X_val_tfidf']
    train_lex = data['train_lex']
    val_lex = data['val_lex']
    y_train = data['y_train']
    y_val = data['y_val']
    metadata = data['metadata']
    
    # ========== COMBINE FEATURES ==========
    print("\n[COMBINING FEATURES]")
    print("=" * 80)
    print("Combining One-Hot Encoding with Lexical Features...")
    
    X_train_onehot_combined = combine_features(X_train_onehot, train_lex)
    X_val_onehot_combined = combine_features(X_val_onehot, val_lex)
    X_train_tfidf_combined = combine_features(X_train_tfidf, train_lex)
    X_val_tfidf_combined = combine_features(X_val_tfidf, val_lex)
    
    print(f"[OK] Train One-Hot+Lexical: {X_train_onehot_combined.shape}")
    print(f"[OK] Val One-Hot+Lexical: {X_val_onehot_combined.shape}")
    print(f"[OK] Train TF-IDF+Lexical: {X_train_tfidf_combined.shape}")
    print(f"[OK] Val TF-IDF+Lexical: {X_val_tfidf_combined.shape}")
    
    # ========== TRAIN MODELS ==========
    print("\n[TRAINING MODELS]")
    print("=" * 80)
    
    trained_models = {}
    all_predictions = {}
    metrics_list = []
    
    # Logistic Regression
    print("\n1. Logistic Regression")
    lr_model, lr_pred, lr_proba = train_logistic_regression(
        X_train_onehot_combined, y_train, X_val_onehot_combined, y_val
    )
    trained_models['LogisticRegression'] = lr_model
    all_predictions['LogisticRegression'] = (lr_pred, lr_proba)
    metrics_list.append(evaluate_model(y_val, lr_pred, lr_proba, 'Logistic Regression'))
    print_model_report(y_val, lr_pred, 'Logistic Regression')
    
    # SVM
    print("\n2. Support Vector Machine")
    svm_model_dict, svm_pred, svm_proba = train_svm(
        X_train_onehot_combined, y_train, X_val_onehot_combined, y_val
    )
    trained_models['SVM'] = svm_model_dict
    all_predictions['SVM'] = (svm_pred, svm_proba)
    metrics_list.append(evaluate_model(y_val, svm_pred, svm_proba, 'SVM'))
    print_model_report(y_val, svm_pred, 'SVM')
    
    # Naive Bayes
    print("\n3. Naive Bayes")
    nb_model, nb_pred, nb_proba = train_naive_bayes(
        X_train_onehot, y_train, X_val_onehot, y_val
    )
    trained_models['NaiveBayes'] = nb_model
    all_predictions['NaiveBayes'] = (nb_pred, nb_proba)
    metrics_list.append(evaluate_model(y_val, nb_pred, nb_proba, 'Naive Bayes'))
    print_model_report(y_val, nb_pred, 'Naive Bayes')
    
    # Random Forest
    print("\n4. Random Forest")
    rf_model, rf_pred, rf_proba = train_random_forest(
        X_train_onehot_combined, y_train, X_val_onehot_combined, y_val
    )
    trained_models['RandomForest'] = rf_model
    all_predictions['RandomForest'] = (rf_pred, rf_proba)
    metrics_list.append(evaluate_model(y_val, rf_pred, rf_proba, 'Random Forest'))
    print_model_report(y_val, rf_pred, 'Random Forest')
    
    # ========== CREATE COMPARISON TABLE ==========
    print("\n[EVALUATION SUMMARY]")
    print("=" * 80)
    
    metrics_df = pd.DataFrame(metrics_list)
    print("\nBinary Classification Metrics (Answer Verification):")
    print(metrics_df.to_string(index=False))
    
    # ========== OPTION SELECTION PREDICTIONS ==========
    print("\n[OPTION SELECTION EVALUATION]")
    print("=" * 80)
    print("Evaluating option selection accuracy (A/B/C/D)...")
    
    # Get metadata for validation set
    n_val_questions = len(y_val) // 4
    val_answers = metadata['val_answers']
    
    # Reconstruct original answers from metadata
    val_true_options = []
    for i in range(n_val_questions):
        idx = i * 4
        val_true_options.append(val_answers[idx])
    
    option_metrics_list = []
    option_predictions_by_model = {}
    
    # Evaluate each model's option selection
    for model_name in ['LogisticRegression', 'SVM', 'NaiveBayes', 'RandomForest']:
        y_pred, y_pred_proba = all_predictions[model_name]
        
        # Predict best options
        pred_options = []
        for q_idx in range(n_val_questions):
            start_idx = q_idx * 4
            end_idx = start_idx + 4
            
            # Get probabilities for all 4 options
            option_probs = y_pred_proba[start_idx:end_idx]
            best_idx = np.argmax(option_probs)
            best_option = ['A', 'B', 'C', 'D'][best_idx]
            
            pred_options.append(best_option)
        
        # Evaluate
        option_metrics = evaluate_option_selection(
            pred_options, val_true_options, model_name
        )
        option_metrics_list.append(option_metrics)
        option_predictions_by_model[model_name] = pred_options
    
    # Ensemble voting (average probabilities)
    ensemble_proba = np.zeros_like(all_predictions['LogisticRegression'][1])
    for model_name in ['LogisticRegression', 'SVM', 'NaiveBayes', 'RandomForest']:
        ensemble_proba += all_predictions[model_name][1]
    ensemble_proba /= 4
    
    ensemble_options = []
    for q_idx in range(n_val_questions):
        start_idx = q_idx * 4
        end_idx = start_idx + 4
        
        option_probs = ensemble_proba[start_idx:end_idx]
        best_idx = np.argmax(option_probs)
        best_option = ['A', 'B', 'C', 'D'][best_idx]
        
        ensemble_options.append(best_option)
    
    ensemble_metrics = evaluate_option_selection(
        ensemble_options, val_true_options, 'Ensemble (Soft Voting)'
    )
    option_metrics_list.append(ensemble_metrics)
    option_predictions_by_model['EnsembleSoftVoting'] = ensemble_options
    
    # Display option selection table
    option_metrics_df = pd.DataFrame(option_metrics_list)
    print("\nOption Selection Accuracy (A/B/C/D Task):")
    print(option_metrics_df.to_string(index=False))
    
    predicted_options_df = save_predicted_options_table(
        option_predictions_by_model=option_predictions_by_model,
        true_options=val_true_options,
        metadata=metadata,
        output_path=f'{output_dir}/predicted_options_val.csv'
    )
    print("Saved: predicted_options_val.csv")

    generation_results = None
    if run_generation:
        generation_results = generate_and_rank_template_questions(
            processed_dir=processed_dir,
            output_dir=output_dir,
            trained_models=trained_models,
            max_questions=generation_max_questions
        )
    
    # ========== SAVE RESULTS ==========
    print("\n[SAVING RESULTS]")
    print("=" * 80)
    
    # Save metrics
    metrics_df.to_csv(f'{output_dir}/binary_classification_metrics.csv', index=False)
    option_metrics_df.to_csv(f'{output_dir}/option_selection_metrics.csv', index=False)
    
    # Save trained models
    with open(f'{output_dir}/trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)
    
    # Save predictions
    with open(f'{output_dir}/predictions.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)
    
    print("[OK] Binary classification metrics saved")
    print("[OK] Option selection metrics saved")
    print("[OK] Trained models saved")
    print("[OK] Predictions saved")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - binary_classification_metrics.csv (answer verification metrics)")
    print("  - option_selection_metrics.csv (option selection accuracy)")
    print("  - predicted_options_val.csv (predicted A/B/C/D per question)")
    print("  - trained_models.pkl (4 trained classifiers)")
    print("  - predictions.pkl (all model predictions)")
    if generation_results is not None:
        print("  - generated_questions_ranked.csv (best template per question)")
        print("  - generated_question_candidates.csv (all ranked templates)")
    
    best_binary = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
    best_option = option_metrics_df.loc[option_metrics_df['Option Selection Accuracy'].idxmax()]
    
    print(f"\nBest Binary Classification Model: {best_binary['Model']} ({best_binary['Accuracy']:.4f})")
    print(f"Best Option Selection Model: {best_option['Model']} ({best_option['Option Selection Accuracy']:.4f})")
    
    return {
        'metrics': metrics_df,
        'option_metrics': option_metrics_df,
        'predicted_options': predicted_options_df,
        'generation': generation_results,
        'models': trained_models,
        'predictions': all_predictions
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = train_and_evaluate(
        processed_dir='/content/drive/MyDrive/data/processed',
        output_dir='/content/drive/MyDrive/models'
    )
    
    print("\n[OK] Phase 3 training and evaluation complete!")
