"""
QUESTION RANKER — Training Script (run on Colab)
Trains a Random Forest classifier to score generated questions by
fluency and relevance using RACE questions as positive examples.
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = '/content/drive/MyDrive/data/processed'
RAW_DIR       = '/content/drive/MyDrive/raw'
OUTPUT_DIR    = '/content/drive/MyDrive/models'
SAMPLE_SIZE   = 10000

STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'it', 'its', 'this', 'that', 'from', 'as', 'not'
])


def extract_ranker_features(question, article, answer):
    """
    Extract fluency + relevance features for a (question, article, answer) triple.
    Returns a 1D feature vector of length 8.
    """
    q_words = set(re.sub(r'[^a-z ]', '', question.lower()).split()) - STOPWORDS
    a_words = set(re.sub(r'[^a-z ]', '', article.lower()).split()) - STOPWORDS

    # F1: keyword overlap with article
    overlap = len(q_words & a_words) / max(len(q_words), 1)

    # F2: question length score (8-15 words is ideal)
    q_len = len(question.split())
    length_score = 1.0 if 8 <= q_len <= 15 else (0.7 if q_len < 8 else 0.5)

    # F3: wh-word presence
    wh_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
    has_wh = float(any(w in question.lower() for w in wh_words))

    # F4: answer length (1-3 words ideal)
    a_len = len(answer.split())
    ans_len_score = 1.0 if 1 <= a_len <= 3 else 0.5

    # F5: answer is a proper noun (capitalized)
    is_proper = float(len(answer) > 0 and answer[0].isupper())

    # F6: answer appears in article
    ans_in_article = float(answer.lower() in article.lower())

    # F7: question ends with '?'
    ends_question = float(question.strip().endswith('?'))

    # F8: question has a blank marker
    has_blank = float('_______' in question)

    return [overlap, length_score, has_wh, ans_len_score,
            is_proper, ans_in_article, ends_question, has_blank]


def make_negative(question, article):
    """
    Create a negative example by shuffling or truncating the question.
    """
    words = question.split()
    if len(words) > 4:
        np.random.shuffle(words)
        return ' '.join(words), ''
    # truncate
    return ' '.join(words[:3]) + '?', ''


def build_training_data(df):
    X, y = [], []
    for _, row in df.iterrows():
        article  = str(row.get('article', ''))
        question = str(row.get('question', ''))
        answer   = str(row.get(row.get('answer', 'A'), ''))

        if not article or not question:
            continue

        # Positive example: real RACE question
        feat_pos = extract_ranker_features(question, article, answer)
        X.append(feat_pos)
        y.append(1)

        # Negative example: shuffled question
        neg_q, neg_a = make_negative(question, article)
        feat_neg = extract_ranker_features(neg_q, article, neg_a)
        X.append(feat_neg)
        y.append(0)

    return np.array(X, dtype=np.float32), np.array(y)


if __name__ == "__main__":
    print("=" * 80)
    print("QUESTION RANKER — Training Random Forest")
    print("=" * 80)

    print("\n[1/4] Loading data...")
    train_df = pd.read_csv(f'{RAW_DIR}/train.csv')
    val_df   = pd.read_csv(f'{RAW_DIR}/dev.csv')

    train_df = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)), random_state=42)
    print(f"  Train: {train_df.shape}")

    print("\n[2/4] Building training examples...")
    X_train, y_train = build_training_data(train_df)
    X_val,   y_val   = build_training_data(val_df.head(1000))
    print(f"  X_train: {X_train.shape}  positives: {y_train.sum()}/{len(y_train)}")
    print(f"  X_val:   {X_val.shape}")

    print("\n[3/4] Training Random Forest ranker...")
    ranker = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    ranker.fit(X_train, y_train)

    y_pred = ranker.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred, target_names=['Negative', 'Positive']))

    print("\n[4/4] Saving...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f'{OUTPUT_DIR}/question_ranker.pkl', 'wb') as f:
        pickle.dump(ranker, f)
    print("[OK] question_ranker.pkl saved")

    print("\nFeature importances:")
    feature_names = ['overlap', 'length_score', 'has_wh', 'ans_len',
                     'is_proper', 'ans_in_article', 'ends_?', 'has_blank']
    for name, imp in sorted(zip(feature_names, ranker.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.4f}")

    print("\n" + "=" * 80)
    print("QUESTION RANKER TRAINING COMPLETE")
    print("=" * 80)
