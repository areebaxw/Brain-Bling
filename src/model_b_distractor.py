import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")
train_df = pd.read_csv('/content/drive/MyDrive/raw/train.csv')
print(f"Train dataset loaded: {train_df.shape}")
val_df   = pd.read_csv('/content/drive/MyDrive/raw/dev.csv')
print(f"Val dataset loaded: {val_df.shape}")

def split_into_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'not', 'its', 'it', 'this',
    'that', 'these', 'those', 'from', 'as', 'into', 'than', 'so', 'if',
    'he', 'she', 'they', 'we', 'his', 'her', 'their', 'our', 'your', 'my'
])

def extract_candidates_from_sentence(sentence, correct_answer):
    """
    Extract candidates using string matching + frequency-based word selection.
    Better filtering for higher quality candidates.
    No NLP libraries required per project constraints.
    """
    words = sentence.split()
    candidates = []
    
    # Extract n-grams with string matching and improved filtering
    for i in range(len(words)):
        for length in range(1, 4):  # 1-3 word phrases
            if i + length <= len(words):
                candidate = ' '.join(words[i:i+length])
                candidate_clean = re.sub(r'[^a-zA-Z0-9 ]', '', candidate).strip()
                # Better filter: min 4 chars, not stopword-only, not a number, not correct answer
                if (len(candidate_clean) >= 4 and
                    candidate_clean.lower() != correct_answer.lower() and
                    candidate_clean.lower() not in STOPWORDS and
                    not candidate_clean.isdigit() and
                    any(c.isalpha() for c in candidate_clean)):
                    candidates.append(candidate_clean)
    
    # Frequency-based word selection (words appearing multiple times = important)
    word_freq = Counter(words)
    freq_candidates = [
        re.sub(r'[^a-zA-Z0-9]', '', word) for word, count in word_freq.items()
        if count >= 2 and len(word) >= 4
        and word.lower() not in STOPWORDS
        and word.lower() != correct_answer.lower()
        and any(c.isalpha() for c in word)
    ]
    candidates.extend(freq_candidates)
    
    return list(set(c for c in candidates if c))

def extract_candidates(article, correct_answer):
    """
    Extract candidates using string matching + frequency-based word selection.
    No NLP libraries required per project constraints.
    """
    sentences = split_into_sentences(article)
    all_candidates = []
    
    for sentence in sentences:
        candidates = extract_candidates_from_sentence(sentence, correct_answer)
        all_candidates.extend(candidates)
    
    return list(set(all_candidates))


def character_match_score(candidate, correct_answer):
    """Compute character match score between candidate and correct answer."""
    candidate_chars = set(candidate.lower())
    correct_chars = set(correct_answer.lower())
    
    intersection = len(candidate_chars & correct_chars)
    union = len(candidate_chars | correct_chars)
    
    return intersection / union if union > 0 else 0


def passage_frequency_count(candidate, article):
    """Count how many times candidate appears in passage."""
    words = article.lower().split()
    candidate_lower = candidate.lower()
    
    count = sum(1 for word in words if candidate_lower in word)
    return count


def get_char_ngram_overlap(text1, text2, n=2):
    """Compute character n-gram overlap between two texts."""
    def get_char_ngrams(text, n):
        text = text.lower().replace(' ', '')
        return set(text[i:i+n] for i in range(len(text)-n+1))
    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def augment_candidates(correct_answer, article):
    """
    Augment candidate pool with string manipulation variants.
    Covers negation, number substitution, morphological variants.
    No NLP libraries required per project constraints.
    """
    extra = []
    words = correct_answer.split()

    # Negation variant
    extra.append('not ' + correct_answer)

    # Number substitution (if answer contains digits)
    if any(c.isdigit() for c in correct_answer):
        for n in range(1, 10):
            variant = re.sub(r'\d+', str(n), correct_answer)
            if variant != correct_answer:
                extra.append(variant)
                break

    # Morphological variants for single words
    if len(words) == 1 and len(words[0]) > 4:
        word = words[0].lower()
        for prefix in ['un', 're', 'over', 'under', 'mis']:
            if not word.startswith(prefix):
                extra.append(prefix + word)
        if word.endswith('ing'):
            extra.append(word[:-3] + 'ed')
        elif word.endswith('ed'):
            extra.append(word[:-2] + 'ing')
        elif word.endswith('ly'):
            extra.append(word[:-2])

    # Top frequent content words from article as additional candidates
    article_words = article.lower().split()
    word_freq = Counter(article_words)
    top_words = [
        w for w, c in word_freq.most_common(30)
        if len(w) >= 4 and w not in STOPWORDS and w != correct_answer.lower()
    ]
    extra.extend(top_words[:15])

    return list(set(c for c in extra if c and c.lower() != correct_answer.lower()))


def extract_features(candidates, correct_answer, article, use_onehot=True, onehot_vectorizer=None):
    """
    Extract features using One-Hot Encoding + Cosine Similarity (primary method).
    No NLP libraries required per project constraints.
    """
    # Try to load one-hot vectorizer if requested
    if use_onehot and onehot_vectorizer is None:
        try:
            if os.path.exists('/content/drive/MyDrive/data/processed/onehot_vectorizer.pkl'):
                with open('/content/drive/MyDrive/data/processed/onehot_vectorizer.pkl', 'rb') as f:
                    onehot_vectorizer = pickle.load(f)
                print("[INFO] Loaded one-hot vectorizer from preprocessing")
        except Exception as e:
            print(f"[WARN] Could not load one-hot vectorizer: {e}. Using TF-IDF.")
            use_onehot = False
    
    # Compute similarity using One-Hot (primary) or TF-IDF (fallback)
    cos_sims = np.zeros(len(candidates))
    try:
        if use_onehot and onehot_vectorizer is not None:
            # Use one-hot encoding (primary method per constraints)
            cand_vecs = onehot_vectorizer.transform(candidates + [correct_answer])
            corr_vec = onehot_vectorizer.transform([correct_answer])
            cos_sims = cosine_similarity(cand_vecs, corr_vec).ravel()
        else:
            # Use TF-IDF fitted on article context for better IDF weights
            vectorizer = TfidfVectorizer(
                max_features=5000, min_df=1,
                token_pattern=r"(?u)\b\w+\b"
            )
            # Fit on article sentences + candidates + correct answer for rich IDF
            article_sentences = split_into_sentences(article)
            context_texts = [s for s in article_sentences if s.strip()]
            fit_texts = context_texts + candidates + [correct_answer]
            if len(fit_texts) >= 2:
                vectorizer.fit(fit_texts)
                cand_vecs = vectorizer.transform(candidates)
                corr_vec  = vectorizer.transform([correct_answer])
                cos_sims  = cosine_similarity(cand_vecs, corr_vec).ravel()
            else:
                cos_sims = np.array([character_match_score(c, correct_answer) for c in candidates])
    except Exception:
        # Fallback: character-level overlap similarity
        cos_sims = np.array([character_match_score(c, correct_answer) for c in candidates])
    
    # Additional simple features
    char_matches    = np.array([character_match_score(c, correct_answer) for c in candidates])
    freq_counts     = np.array([passage_frequency_count(c, article) for c in candidates])
    word_lengths    = np.array([len(c.split()) for c in candidates])
    char_lengths    = np.array([len(c) for c in candidates])
    char_bigrams    = np.array([get_char_ngram_overlap(c, correct_answer, 2) for c in candidates])
    char_trigrams   = np.array([get_char_ngram_overlap(c, correct_answer, 3) for c in candidates])
    # Word count match: 1.0 if same word count as correct answer, else ratio
    ans_wc = max(len(correct_answer.split()), 1)
    wc_match = np.array([
        min(len(c.split()), ans_wc) / max(len(c.split()), ans_wc)
        for c in candidates
    ])
    # Character length ratio: similarity in length to correct answer
    ans_cl = max(len(correct_answer), 1)
    len_ratio = np.array([
        min(len(c), ans_cl) / max(len(c), ans_cl)
        for c in candidates
    ])

    # Combine features
    all_features = np.column_stack([cos_sims, char_matches, freq_counts, word_lengths, char_lengths, char_bigrams, char_trigrams, wc_match, len_ratio])
    
    return all_features, onehot_vectorizer

def create_training_data(df):
    """
    Create training data by using actual wrong options as positives and
    article candidates as negatives. This ensures real positive examples
    since RACE wrong options are rarely extractable from the article.
    No NLP libraries required per project constraints.
    """
    training_data = []
    labels = []
    rng = np.random.default_rng(42)
    
    for idx, row in df.iterrows():
        if row.get('answer') not in ['A', 'B', 'C', 'D']:
            continue
        article = str(row['article']) if not pd.isna(row['article']) else ''
        correct_answer = str(row[row['answer']])
        wrong_options  = [str(row[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != row['answer']]
        
        if not article or not correct_answer:
            continue
        
        # Add actual wrong options as guaranteed positives
        for opt in wrong_options:
            training_data.append((opt, correct_answer, article))
            labels.append(1)
        
        # Add article candidates as negatives (undersample to 5x positives)
        candidates = extract_candidates(article, correct_answer)
        negatives = [c for c in candidates if c not in wrong_options]
        max_negatives = len(wrong_options) * 5
        if len(negatives) > max_negatives:
            neg_idx = rng.choice(len(negatives), max_negatives, replace=False)
            negatives = [negatives[i] for i in neg_idx]
        
        for neg in negatives:
            training_data.append((neg, correct_answer, article))
            labels.append(0)
    
    return training_data, labels

def train_ranker(training_data, labels, use_onehot=True, onehot_vectorizer=None):
    """
    Train the distractor ranker using One-Hot Encoding + Cosine Similarity.
    No NLP libraries required per project constraints.
    """
    MAX_SAMPLES = 100000
    if len(training_data) > MAX_SAMPLES:
        print(f"[INFO] Sampling {MAX_SAMPLES} from {len(training_data)} training samples")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(training_data), MAX_SAMPLES, replace=False)
        training_data = [training_data[i] for i in idx]
        labels = [labels[i] for i in idx]
    
    candidates  = [t[0] for t in training_data]
    correct_ans = [t[1] for t in training_data]
    articles    = [t[2] for t in training_data]
    
    # Compute features for all samples
    print("[INFO] Computing features for training...")
    all_features = []
    
    for i in range(len(candidates)):
        features, _ = extract_features([candidates[i]], correct_ans[i], articles[i],
                                      use_onehot=use_onehot, onehot_vectorizer=onehot_vectorizer)
        all_features.append(features[0])
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i+1}/{len(candidates)} samples...")
    
    X = np.array(all_features)
    y = np.array(labels)
    
    pos_count = int(np.sum(y))
    neg_count = int(len(y) - pos_count)
    print(f"[INFO] Class distribution — Positive: {pos_count}, Negative: {neg_count}")
    print(f"[INFO] Training RandomForest on {X.shape[0]} samples with {X.shape[1]} features")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',  # Compensates for remaining imbalance
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf

def diversity_penalty(candidate, selected_distractors):
    """
    Compute similarity between candidate and already-selected distractors.
    Returns max token overlap with any selected distractor (0=diverse, 1=identical).
    """
    if not selected_distractors:
        return 0.0
    cand_tokens = set(candidate.lower().split())
    max_overlap = 0.0
    for selected in selected_distractors:
        sel_tokens = set(selected.lower().split())
        if cand_tokens | sel_tokens:
            overlap = len(cand_tokens & sel_tokens) / len(cand_tokens | sel_tokens)
            max_overlap = max(max_overlap, overlap)
    return max_overlap


def generate_distractors_ml(ranker, article, correct_answer, top_k=3, use_onehot=True, onehot_vectorizer=None):
    """
    Generate distractors using One-Hot Encoding + Cosine Similarity ranking.
    Applies diversity penalty to ensure distractors are not trivially similar.
    Augments candidates with string manipulation variants.
    No NLP libraries required per project constraints.
    """
    candidates = extract_candidates(article, correct_answer)
    # Augment with string manipulation variants
    candidates = list(set(candidates + augment_candidates(correct_answer, article)))
    
    if len(candidates) == 0:
        return [], onehot_vectorizer
    
    features, onehot_vectorizer = extract_features(candidates, correct_answer, article, use_onehot, onehot_vectorizer)
    
    # Handle edge case where predict_proba has only 1 column (single class in training)
    proba = ranker.predict_proba(features)
    if proba.shape[1] > 1:
        scores = proba[:, 1]
    else:
        scores = proba[:, 0]
    
    candidate_scores = list(zip(candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top-k with diversity penalty
    DIVERSITY_THRESHOLD = 0.5  # Max allowed token overlap between distractors
    selected = []
    for cand, score in candidate_scores:
        if len(selected) >= top_k:
            break
        penalty = diversity_penalty(cand, selected)
        if penalty < DIVERSITY_THRESHOLD:
            selected.append(cand)
    
    # If not enough diverse distractors, fill remaining slots from top scored
    if len(selected) < top_k:
        for cand, score in candidate_scores:
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= top_k:
                break
    
    return selected, onehot_vectorizer

def token_f1(candidate, reference):
    c_toks = set(candidate.lower().split())
    r_toks = set(reference.lower().split())
    common = c_toks & r_toks
    if not common:
        return 0.0, 0.0, 0.0
    p = len(common) / len(c_toks)
    r = len(common) / len(r_toks)
    f = 2 * p * r / (p + r)
    return p, r, f


def evaluate_distractors(generated_distractors, original_wrong_options):
    if not generated_distractors or not original_wrong_options:
        return 0.0, 0.0, 0.0
    all_p, all_r, all_f = [], [], []
    for gen in generated_distractors:
        best_p, best_r, best_f = 0.0, 0.0, 0.0
        for ref in original_wrong_options:
            p, r, f = token_f1(gen, ref)
            if f > best_f:
                best_p, best_r, best_f = p, r, f
        all_p.append(best_p)
        all_r.append(best_r)
        all_f.append(best_f)
    return (
        sum(all_p) / len(all_p),
        sum(all_r) / len(all_r),
        sum(all_f) / len(all_f)
    )

def evaluate_accuracy(df, ranker, use_onehot=True, onehot_vectorizer=None):
    """
    Evaluate partial-match accuracy: % of questions where at least one
    generated distractor has token F1 > 0.1 with any reference wrong option.
    No NLP libraries required per project constraints.
    """
    partial_match, total = 0, 0
    for idx, row in df.iterrows():
        if row.get('answer') not in ['A', 'B', 'C', 'D']:
            continue
        article        = str(row['article']) if not pd.isna(row['article']) else ''
        correct_answer = str(row[row['answer']])
        wrong_options  = [str(row[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != row['answer']]
        if not article or not correct_answer:
            continue
        dists, _ = generate_distractors_ml(ranker, article, correct_answer, top_k=3, use_onehot=use_onehot, onehot_vectorizer=onehot_vectorizer)
        _, _, f = evaluate_distractors(dists, wrong_options)
        if f > 0.1:
            partial_match += 1
        total += 1
    return partial_match / total if total > 0 else 0


print("="*80)
print("DISTRACTOR MODEL TRAINING")
print("="*80)

# Configuration
USE_ONEHOT = True  # Primary method per project constraints
TRAIN_SAMPLE_SIZE = 20000  # Training sample size
VAL_SAMPLE_SIZE = 100  # Validation sample size

print(f"\n[CONFIG] Using one-hot encoding (primary): {USE_ONEHOT}")
print(f"[CONFIG] Training sample size: {TRAIN_SAMPLE_SIZE}")
print(f"[CONFIG] Validation sample size: {VAL_SAMPLE_SIZE}")

# Load one-hot vectorizer (primary method per constraints)
onehot_vectorizer = None
if USE_ONEHOT:
    try:
        if os.path.exists('/content/drive/MyDrive/data/processed/onehot_vectorizer.pkl'):
            with open('/content/drive/MyDrive/data/processed/onehot_vectorizer.pkl', 'rb') as f:
                onehot_vectorizer = pickle.load(f)
            print("[INFO] Loaded one-hot vectorizer from preprocessing")
        else:
            print("[WARN] One-hot vectorizer not found. Will use TF-IDF instead.")
            USE_ONEHOT = False
    except Exception as e:
        print(f"[WARN] Failed to load one-hot vectorizer: {e}. Using TF-IDF.")
        USE_ONEHOT = False

print(f"\n[1/4] Creating training data ({TRAIN_SAMPLE_SIZE} row sample)...")
sample_df = train_df.sample(n=min(TRAIN_SAMPLE_SIZE, len(train_df)), random_state=42)
training_data, labels = create_training_data(sample_df)
print(f"Training samples: {len(training_data)}")

print(f"\n[2/4] Training ML ranker...")
ranker = train_ranker(training_data, labels, use_onehot=USE_ONEHOT, onehot_vectorizer=onehot_vectorizer)
print("Ranker trained.")

print(f"\n[3/4] Evaluating on {VAL_SAMPLE_SIZE} val samples...")
results = []
eval_count = 0
for idx, row in val_df.head(VAL_SAMPLE_SIZE).iterrows():
    if row.get('answer') not in ['A', 'B', 'C', 'D']:
        continue
    article        = str(row['article']) if not pd.isna(row['article']) else ''
    correct_answer = str(row[row['answer']])
    wrong_options  = [str(row[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != row['answer']]
    if not article or not correct_answer:
        continue
    eval_count += 1
    if eval_count % 20 == 0:
        print(f"  Progress: {eval_count}/{VAL_SAMPLE_SIZE} samples done...")
    ml_d, onehot_vectorizer = generate_distractors_ml(ranker, article, correct_answer, use_onehot=USE_ONEHOT, onehot_vectorizer=onehot_vectorizer)
    ml_p,  ml_r,  ml_f  = evaluate_distractors(ml_d,  wrong_options)
    results.append({'question_id': row.get('id', idx),
                    'ml_precision': ml_p,   'ml_recall': ml_r,   'ml_f1': ml_f,
                    'num_ml_distractors': len(ml_d)})

results_df = pd.DataFrame(results)
print(f"\n[4/4] Results Summary (token-level F1 on {len(results_df)} samples):")
print(results_df[['ml_precision', 'ml_recall', 'ml_f1']].mean())
print(f"Partial-Match Accuracy (F1>0.1): {evaluate_accuracy(val_df.head(VAL_SAMPLE_SIZE), ranker, use_onehot=USE_ONEHOT, onehot_vectorizer=onehot_vectorizer):.4f}")

os.makedirs('/content/drive/MyDrive/models', exist_ok=True)
results_df.to_csv('/content/drive/MyDrive/models/distractor_results.csv', index=False)
with open('/content/drive/MyDrive/models/distractor_model.pkl', 'wb') as f:
    pickle.dump(ranker, f)
print("Done. Files saved to Drive.")
print("="*80)
