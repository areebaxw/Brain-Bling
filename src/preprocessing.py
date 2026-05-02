"""
PHASE 2 - PREPROCESSING MODULE
================================
Comprehensive preprocessing pipeline for RACE dataset including:
1. Text cleaning (lowercasing, punctuation removal)
2. One-Hot Encoding (primary feature representation)
3. TF-IDF vectorization (optional)
4. Cosine similarity matrices
5. Handcrafted lexical features
6. Save processed data and feature matrices
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import string
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing extra whitespace
    - Removing special characters
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataset(df):
    """
    Preprocess all text columns in the dataset
    """
    df = df.copy()
    
    text_cols = ['article', 'question', 'A', 'B', 'C', 'D']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    return df


# ============================================================================
# 2. FEATURE ENGINEERING: ONE-HOT ENCODING
# ============================================================================

def create_combined_text(df):
    """
    Create combined text (article + question + option) for each row and option.
    Returns a list of texts and metadata for tracking.
    """
    combined_texts = []
    option_labels = []
    row_ids = []
    correct_answers = []
    
    for idx, row in df.iterrows():
        article = row['article']
        question = row['question']
        row_id = row.get('id', idx)
        correct_answer = row.get('answer', None)
        
        for option in ['A', 'B', 'C', 'D']:
            option_text = row[option]
            # Combine: article + question + option
            combined = f"{article} {question} {option_text}"
            
            combined_texts.append(combined)
            option_labels.append(option)
            row_ids.append(row_id)
            correct_answers.append(correct_answer)
    
    return combined_texts, option_labels, row_ids, correct_answers


def create_onehot_features(texts, fit_vectorizer=None, max_features=5000):
    """
    Create One-Hot Encoding features using CountVectorizer.
    Returns sparse matrix and fitted vectorizer.
    """
    if fit_vectorizer is None:
        # Fit new vectorizer (training data)
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=False,  # Already lowercased
            min_df=2,  # Ignore terms appearing in < 2 documents
            max_df=0.95  # Ignore terms appearing in > 95% of documents
        )
        X = vectorizer.fit_transform(texts)
    else:
        # Use existing vectorizer (validation/test data)
        X = fit_vectorizer.transform(texts)
        vectorizer = fit_vectorizer
    
    return X, vectorizer


# ============================================================================
# 3. TF-IDF VECTORIZATION
# ============================================================================

def create_tfidf_features(texts, fit_vectorizer=None, max_features=20000):
    """
    Create TF-IDF features.
    Returns sparse matrix and fitted vectorizer.
    """
    if fit_vectorizer is None:
        # Fit new vectorizer (training data)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=False,  # Already lowercased
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X = vectorizer.fit_transform(texts)
    else:
        # Use existing vectorizer (validation/test data)
        X = fit_vectorizer.transform(texts)
        vectorizer = fit_vectorizer
    
    return X, vectorizer


# ============================================================================
# 4. COSINE SIMILARITY MATRICES
# ============================================================================

def compute_cosine_similarity_matrix(feature_matrix, n_questions):
    """
    Compute cosine similarity between options for each question.
    Returns a matrix of shape (n_questions, 4) with similarity scores.
    
    For each question, compute the average similarity of each option to the other 3 options.
    This measures how distinct each option is from the others.
    """
    similarity_matrix = []
    
    # Convert sparse to dense if needed for cosine_similarity
    if sp.issparse(feature_matrix):
        feature_matrix_dense = feature_matrix.toarray()
    else:
        feature_matrix_dense = feature_matrix
    
    for q_idx in range(n_questions):
        # Get indices for this question's 4 options
        start_idx = q_idx * 4
        end_idx = start_idx + 4
        
        if end_idx > feature_matrix_dense.shape[0]:
            # Handle edge case
            similarity_matrix.append([0.0, 0.0, 0.0, 0.0])
            continue
        
        # Get feature vectors for all 4 options
        option_vectors = feature_matrix_dense[start_idx:end_idx]
        
        # Compute pairwise cosine similarity between options
        # Shape: (4, 4) where sim_matrix[i,j] = similarity between option i and option j
        sim_matrix = cosine_similarity(option_vectors)
        
        # For each option, compute average similarity to other options
        q_similarities = []
        for opt_idx in range(4):
            # Get similarities to other options (exclude self-similarity)
            other_similarities = [sim_matrix[opt_idx][j] for j in range(4) if j != opt_idx]
            avg_similarity = np.mean(other_similarities) if other_similarities else 0.0
            q_similarities.append(avg_similarity)
        
        similarity_matrix.append(q_similarities)
    
    return np.array(similarity_matrix)


# ============================================================================
# 5. HANDCRAFTED LEXICAL FEATURES
# ============================================================================

def extract_lexical_features(df, combined_texts, option_labels):
    """
    Extract handcrafted lexical features for each sample:
    - Word overlap (Jaccard similarity) between question and option
    - Question word density in option
    - Option length ratio
    - Named entity presence (approximate)
    - Keyword match count
    """
    features = []
    
    for idx, row in df.iterrows():
        question_words = set(row['question'].split())
        
        for opt_idx, option in enumerate(['A', 'B', 'C', 'D']):
            option_words = set(row[option].split())
            
            # 1. Jaccard Similarity (word overlap)
            if len(question_words.union(option_words)) > 0:
                jaccard = len(question_words.intersection(option_words)) / len(question_words.union(option_words))
            else:
                jaccard = 0.0
            
            # 2. Word density (% of option words that are in question)
            if len(option_words) > 0:
                word_density = len(question_words.intersection(option_words)) / len(option_words)
            else:
                word_density = 0.0
            
            # 3. Option length ratio (option length / article length)
            article_len = len(row['article'].split())
            option_len = len(row[option].split())
            if article_len > 0:
                length_ratio = option_len / article_len
            else:
                length_ratio = 0.0
            
            # 4. Capital letter count (indicator of proper nouns/entities)
            capital_count = sum(1 for c in row[option] if c.isupper())
            capital_density = capital_count / max(len(row[option]), 1)
            
            # 5. Number presence (does option contain numbers?)
            has_numbers = 1 if any(c.isdigit() for c in row[option]) else 0
            
            # 6. Common word count (overlap with article)
            article_words = set(row['article'].split())
            common_words = len(option_words.intersection(article_words))
            
            # 7. Average word length
            avg_word_len = np.mean([len(w) for w in option_words]) if option_words else 0
            
            features.append({
                'jaccard_similarity': jaccard,
                'word_density': word_density,
                'length_ratio': length_ratio,
                'capital_density': capital_density,
                'has_numbers': has_numbers,
                'common_word_count': common_words,
                'avg_word_length': avg_word_len
            })
    
    return pd.DataFrame(features)


# ============================================================================
# 6. LABEL ENCODING (FOR TRAINING)
# ============================================================================

def create_binary_labels(df, option_labels):
    """
    Create binary labels for training:
    1 if option is the correct answer, 0 otherwise
    """
    labels = []
    
    for idx, row in df.iterrows():
        correct_answer = row['answer']
        
        for option in ['A', 'B', 'C', 'D']:
            label = 1 if option == correct_answer else 0
            labels.append(label)
    
    return np.array(labels)


# ============================================================================
# 7. MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_and_save(
    train_path='data/raw/train.csv',
    val_path='data/raw/dev.csv',
    test_path='data/raw/test.csv',
    output_dir='data/processed',
    onehot_features=5000,
    tfidf_features=20000,
    sample_size=None
):
    """
    Main preprocessing pipeline:
    1. Load datasets
    2. Clean text
    3. Create features (One-Hot, TF-IDF)
    4. Extract lexical features
    5. Save all processed data
    
    Args:
        sample_size: If specified, sample this many rows from each dataset for faster processing
    """
    
    print("="*80)
    print("PHASE 2 - PREPROCESSING PIPELINE")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== LOAD DATA ==========
    print("\n[1/7] Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Remove index column if exists
    for df in [train_df, val_df, test_df]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Sample if requested
    if sample_size:
        print(f"  Sampling {sample_size} rows from each dataset...")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(sample_size, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    
    print(f"  Train: {train_df.shape}")
    print(f"  Validation: {val_df.shape}")
    print(f"  Test: {test_df.shape}")
    
    # ========== TEXT CLEANING ==========
    print("\n[2/7] Cleaning text (lowercasing, punctuation removal)...")
    train_df_clean = preprocess_dataset(train_df)
    val_df_clean = preprocess_dataset(val_df)
    test_df_clean = preprocess_dataset(test_df)
    print("  [OK] Text cleaned")
    
    # ========== CREATE COMBINED TEXT ==========
    print("\n[3/7] Creating combined text features (article + question + option)...")
    train_texts, train_opts, train_ids, train_answers = create_combined_text(train_df_clean)
    val_texts, val_opts, val_ids, val_answers = create_combined_text(val_df_clean)
    test_texts, test_opts, test_ids, test_answers = create_combined_text(test_df_clean)
    
    print(f"  Train: {len(train_texts)} samples ({len(train_df)} questions × 4 options)")
    print(f"  Validation: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    # ========== ONE-HOT ENCODING ==========
    print(f"\n[4/7] Creating One-Hot Encoding features (max_features={onehot_features})...")
    X_train_onehot, onehot_vectorizer = create_onehot_features(
        train_texts, max_features=onehot_features
    )
    X_val_onehot, _ = create_onehot_features(val_texts, fit_vectorizer=onehot_vectorizer)
    X_test_onehot, _ = create_onehot_features(test_texts, fit_vectorizer=onehot_vectorizer)
    
    print(f"  Train One-Hot shape: {X_train_onehot.shape}")
    print(f"  Validation One-Hot shape: {X_val_onehot.shape}")
    print(f"  Test One-Hot shape: {X_test_onehot.shape}")
    
    # ========== TF-IDF VECTORIZATION ==========
    print(f"\n[5/7] Creating TF-IDF features (max_features={tfidf_features})...")
    X_train_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_texts, max_features=tfidf_features
    )
    X_val_tfidf, _ = create_tfidf_features(val_texts, fit_vectorizer=tfidf_vectorizer)
    X_test_tfidf, _ = create_tfidf_features(test_texts, fit_vectorizer=tfidf_vectorizer)
    
    print(f"  Train TF-IDF shape: {X_train_tfidf.shape}")
    print(f"  Validation TF-IDF shape: {X_val_tfidf.shape}")
    print(f"  Test TF-IDF shape: {X_test_tfidf.shape}")
    
    # ========== LEXICAL FEATURES ==========
    print("\n[6/7] Extracting handcrafted lexical features...")
    train_lex_features = extract_lexical_features(train_df_clean, train_texts, train_opts)
    val_lex_features = extract_lexical_features(val_df_clean, val_texts, val_opts)
    test_lex_features = extract_lexical_features(test_df_clean, test_texts, test_opts)
    
    print(f"  Train lexical features shape: {train_lex_features.shape}")
    print(f"  Validation lexical features shape: {val_lex_features.shape}")
    print(f"  Test lexical features shape: {test_lex_features.shape}")
    print(f"  Features: {list(train_lex_features.columns)}")
    
    # ========== SAVE EVERYTHING ==========
    print("\n[7/7] Saving processed data and feature matrices...")
    
    # Save cleaned datasets
    train_df_clean.to_csv(f'{output_dir}/train_clean.csv', index=False)
    val_df_clean.to_csv(f'{output_dir}/val_clean.csv', index=False)
    test_df_clean.to_csv(f'{output_dir}/test_clean.csv', index=False)
    
    # Save feature matrices
    import scipy.sparse as sp
    sp.save_npz(f'{output_dir}/X_train_onehot.npz', X_train_onehot)
    sp.save_npz(f'{output_dir}/X_val_onehot.npz', X_val_onehot)
    sp.save_npz(f'{output_dir}/X_test_onehot.npz', X_test_onehot)
    
    sp.save_npz(f'{output_dir}/X_train_tfidf.npz', X_train_tfidf)
    sp.save_npz(f'{output_dir}/X_val_tfidf.npz', X_val_tfidf)
    sp.save_npz(f'{output_dir}/X_test_tfidf.npz', X_test_tfidf)
    
    # Save lexical features
    train_lex_features.to_csv(f'{output_dir}/train_lexical_features.csv', index=False)
    val_lex_features.to_csv(f'{output_dir}/val_lexical_features.csv', index=False)
    test_lex_features.to_csv(f'{output_dir}/test_lexical_features.csv', index=False)
    
    # Save vectorizers
    with open(f'{output_dir}/onehot_vectorizer.pkl', 'wb') as f:
        pickle.dump(onehot_vectorizer, f)
    
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Save labels
    y_train = create_binary_labels(train_df_clean, train_opts)
    y_val = create_binary_labels(val_df_clean, val_opts)
    
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_val.npy', y_val)
    
    # Save metadata
    metadata = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'train_options': train_opts,
        'val_options': val_opts,
        'test_options': test_opts,
        'train_answers': train_answers,
        'val_answers': val_answers,
        'test_answers': test_answers
    }
    
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("  [OK] Cleaned datasets saved")
    print("  [OK] One-Hot feature matrices saved")
    print("  [OK] TF-IDF feature matrices saved")
    print("  [OK] Lexical features saved")
    print("  [OK] Vectorizers saved")
    print("  [OK] Labels saved")
    print("  [OK] Metadata saved")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - Cleaned datasets: *_clean.csv")
    print("  - Feature matrices: X_*_onehot.npz, X_*_tfidf.npz")
    print("  - Lexical features: *_lexical_features.csv")
    print("  - Vectorizers: onehot_vectorizer.pkl, tfidf_vectorizer.pkl")
    print("  - Labels: y_train.npy, y_val.npy")
    print("  - Metadata: metadata.pkl")
    
    return {
        'train_df': train_df_clean,
        'val_df': val_df_clean,
        'test_df': test_df_clean,
        'X_train_onehot': X_train_onehot,
        'X_val_onehot': X_val_onehot,
        'X_test_onehot': X_test_onehot,
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'train_lex': train_lex_features,
        'val_lex': val_lex_features,
        'test_lex': test_lex_features,
        'y_train': y_train,
        'y_val': y_val,
        'onehot_vectorizer': onehot_vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer,
        'metadata': metadata
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run preprocessing pipeline on full dataset (no sampling)
    results = preprocess_and_save(
        train_path='/content/drive/MyDrive/raw/train.csv',
        val_path='/content/drive/MyDrive/raw/dev.csv',
        test_path='/content/drive/MyDrive/raw/test.csv',
        output_dir='/content/drive/MyDrive/data/processed',
        onehot_features=5000,
        tfidf_features=20000,
        sample_size=None
    )
    
    print("\n[OK] All preprocessing complete and data saved!")
