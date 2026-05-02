import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(train_path, val_path, test_path, sample_size=None):
    """
    Load datasets from CSV files.
    
    Args:
        sample_size: If specified, sample this many rows from each dataset
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Remove extra column if exists
    for df in [train_df, val_df, test_df]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Sample if requested
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(sample_size, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    
    return train_df, val_df, test_df


# ============================================================================
# 2. PREPROCESS
# ============================================================================

def preprocess(df):
    """
    Basic preprocessing: Remove leading/trailing whitespace.
    Keep data as-is for TF-IDF to handle.
    """
    df = df.copy()
    
    # Strip whitespace from text columns
    text_cols = ['article', 'question', 'A', 'B', 'C', 'D']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
    
    return df


# ============================================================================
# 3. CREATE BINARY DATASET (OPTIMIZED)
# ============================================================================

def create_binary_dataset(df, is_test=False):
    """
    Transform dataset to binary classification format (optimized).
    
    For each question: create 4 rows (one per option).
    Text: article + question + option
    Label: 1 if option == correct answer, 0 otherwise
    
    Returns:
        - texts: list of combined texts
        - labels: list of binary labels
        - metadata: list of dicts with question info (for evaluation)
    """
    # Pre-allocate lists
    n_samples = len(df) * 4
    texts = []
    labels = [] if not is_test else None
    metadata = []
    
    # Extract columns as arrays for faster access
    articles = df['article'].values
    questions = df['question'].values
    ids = df['id'].values
    answers = df['answer'].values if not is_test else None
    
    # Get option columns
    option_texts = {opt: df[opt].values for opt in ['A', 'B', 'C', 'D']}
    option_labels = ['A', 'B', 'C', 'D']
    
    # Build dataset
    for i in range(len(df)):
        article = articles[i]
        question = questions[i]
        row_id = ids[i]
        correct_answer = answers[i] if answers is not None else None
        
        for opt_idx, opt_label in enumerate(option_labels):
            # Combine text: article + question + option
            option_text = option_texts[opt_label][i]
            combined_text = f"{article} {question} {option_text}"
            texts.append(combined_text)
            
            # Binary label: 1 if correct, 0 otherwise
            if not is_test:
                label = 1 if opt_label == correct_answer else 0
                labels.append(label)
            
            # Metadata for evaluation
            metadata.append({
                'id': row_id,
                'question': question,
                'option': opt_label,
                'correct_answer': correct_answer
            })
    
    return texts, labels, metadata


# ============================================================================
# 4. VECTORIZE
# ============================================================================

def vectorize(train_texts, val_texts, test_texts=None, verbose=True):
    """
    Apply TF-IDF vectorization.
    Fit on training data only, transform validation and test.
    """
    if verbose:
        print("  Creating TF-IDF vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english'
    )
    
    # Fit on training data
    if verbose:
        print("  Fitting on training data...")
    X_train = vectorizer.fit_transform(train_texts)
    
    # Transform validation and test
    if verbose:
        print("  Transforming validation data...")
    X_val = vectorizer.transform(val_texts)
    
    X_test = None
    if test_texts is not None:
        if verbose:
            print("  Transforming test data...")
        X_test = vectorizer.transform(test_texts)
    
    return X_train, X_val, X_test, vectorizer


# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

def train_model(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, verbose=0)
    model.fit(X_train, y_train)
    return model


# ============================================================================
# 6. PREDICT OPTIONS
# ============================================================================

def predict_options(model, X_val, metadata_val, num_questions):
    """
    Predict the best option for each question.
    
    For each question:
    - Get probabilities for all 4 options
    - Select option with highest probability
    
    Returns:
        - predicted_options: list of predicted options (A/B/C/D)
        - question_ids: list of question IDs
        - questions: list of question texts
    """
    # Get probabilities: shape (num_samples, 2)
    # We want probability of class 1 (correct answer)
    probs = model.predict_proba(X_val)[:, 1]
    
    predicted_options = []
    question_ids = []
    questions = []
    
    # For each question (4 options per question)
    for i in range(num_questions):
        idx_start = i * 4
        idx_end = idx_start + 4
        
        # Probabilities for options A, B, C, D
        option_probs = probs[idx_start:idx_end]
        
        # Get option with max probability
        best_option_idx = np.argmax(option_probs)
        best_option = ['A', 'B', 'C', 'D'][best_option_idx]
        
        predicted_options.append(best_option)
        question_ids.append(metadata_val[idx_start]['id'])
        questions.append(metadata_val[idx_start]['question'])
    
    return predicted_options, question_ids, questions


# ============================================================================
# 7. EVALUATE
# ============================================================================

def evaluate(y_true, y_pred, questions, question_ids, num_samples=5):
    """
    Evaluate model performance.
    
    - Compute accuracy (based on correct option selection)
    - Print sample predictions
    """
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # Print sample predictions
    print("Sample Predictions (first {} questions):".format(min(num_samples, len(y_true))))
    print("-" * 70)
    print(f"{'Question':<50} | {'Pred':<4} | {'True':<4}")
    print("-" * 70)
    
    for i in range(min(num_samples, len(y_true))):
        q = questions[i][:47] + "..." if len(questions[i]) > 50 else questions[i]
        pred = y_pred[i]
        true = y_true[i]
        match = "✓" if pred == true else "✗"
        print(f"{q:<50} | {pred:<4} | {true:<4} {match}")
    
    print("-" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_data(
        "data/raw/train.csv",
        "data/raw/dev.csv",
        "data/raw/test.csv",
        sample_size=None  # Set to integer (e.g., 5000) to use only a sample
    )
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    # Preprocess
    print("\nPreprocessing data...")
    train_df = preprocess(train_df)
    val_df = preprocess(val_df)
    test_df = preprocess(test_df)
    
    # Create binary datasets
    print("Creating binary classification dataset...")
    print("  Training set...")
    X_train_texts, y_train, meta_train = create_binary_dataset(train_df, is_test=False)
    print(f"    ✓ {len(X_train_texts)} samples")
    
    print("  Validation set...")
    X_val_texts, y_val, meta_val = create_binary_dataset(val_df, is_test=False)
    print(f"    ✓ {len(X_val_texts)} samples")
    
    print("  Test set...")
    X_test_texts, _, meta_test = create_binary_dataset(test_df, is_test=True)
    print(f"    ✓ {len(X_test_texts)} samples")
    
    # Vectorize
    print("\nVectorizing (TF-IDF)...")
    X_train_vec, X_val_vec, X_test_vec, vectorizer = vectorize(
        X_train_texts,
        X_val_texts,
        X_test_texts,
        verbose=True
    )
    print(f"  Vectorizer: max_features=10000, stop_words='english'")
    print(f"  Train vector shape: {X_train_vec.shape}")
    print(f"  Val vector shape: {X_val_vec.shape}")
    
    # Train model
    print("\nTraining Logistic Regression...")
    print(f"  Training on {len(y_train)} binary samples...")
    model = train_model(X_train_vec, y_train)
    print("  ✓ Model trained successfully")
    
    # Predict on validation set
    print("\nPredicting on validation set...")
    num_val_questions = len(val_df)
    pred_options, pred_ids, pred_questions = predict_options(
        model, X_val_vec, meta_val, num_val_questions
    )
    print(f"  ✓ Predictions complete ({num_val_questions} questions)")
    
    # Evaluate
    print("\nEvaluating...")
    actual_answers = val_df['answer'].tolist()
    evaluate(actual_answers, pred_options, pred_questions, pred_ids, num_samples=5)
    
    print("\n✓ Model A (TF-IDF + Logistic Regression) completed!")