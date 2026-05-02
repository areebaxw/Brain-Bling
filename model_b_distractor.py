import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from collections import Counter
import os

train_df = pd.read_csv('/content/drive/MyDrive/raw/train.csv')
val_df   = pd.read_csv('/content/drive/MyDrive/raw/dev.csv')

def split_into_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def extract_candidates_from_sentence(sentence, correct_answer):
    words = sentence.split()
    candidates = []
    
    for i in range(len(words)):
        for length in range(1, 4):
            if i + length <= len(words):
                candidate = ' '.join(words[i:i+length])
                if len(candidate) > 2 and candidate.lower() != correct_answer.lower():
                    candidates.append(candidate)
    
    word_freq = Counter(words)
    freq_candidates = [word for word, count in word_freq.items() if count > 1 and len(word) > 2 and word.lower() != correct_answer.lower()]
    candidates.extend(freq_candidates)
    
    return list(set(candidates))

def extract_candidates(article, correct_answer):
    sentences = split_into_sentences(article)
    all_candidates = []
    
    for sentence in sentences:
        candidates = extract_candidates_from_sentence(sentence, correct_answer)
        all_candidates.extend(candidates)
    
    return list(set(all_candidates))

def character_match_score(candidate, correct_answer):
    candidate_chars = set(candidate.lower())
    correct_chars = set(correct_answer.lower())
    
    intersection = len(candidate_chars & correct_chars)
    union = len(candidate_chars | correct_chars)
    
    return intersection / union if union > 0 else 0

def passage_frequency_count(candidate, article):
    words = article.lower().split()
    candidate_lower = candidate.lower()
    
    count = sum(1 for word in words if candidate_lower in word)
    return count

def extract_features(candidates, correct_answer, article):
    try:
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
        vectorizer.fit(candidates + [correct_answer])
        cand_vecs = vectorizer.transform(candidates)
        corr_vec  = vectorizer.transform([correct_answer])
        cos_sims  = cosine_similarity(cand_vecs, corr_vec).ravel()
    except Exception:
        cos_sims = np.zeros(len(candidates))
    char_matches = np.array([character_match_score(c, correct_answer) for c in candidates])
    freq_counts  = np.array([passage_frequency_count(c, article) for c in candidates])
    return np.column_stack([cos_sims, char_matches, freq_counts])

def create_training_data(df):
    training_data = []
    labels = []
    
    for idx, row in df.iterrows():
        if row.get('answer') not in ['A', 'B', 'C', 'D']:
            continue
        article = str(row['article']) if not pd.isna(row['article']) else ''
        question = str(row['question']) if not pd.isna(row['question']) else ''
        correct_answer = str(row[row['answer']])
        wrong_options = [str(row[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != row['answer']]
        if not article or not correct_answer:
            continue
        
        candidates = extract_candidates(article, correct_answer)
        
        for candidate in candidates:
            is_wrong_option = candidate in wrong_options
            training_data.append((candidate, correct_answer, article))
            labels.append(1 if is_wrong_option else 0)
    
    return training_data, labels

def train_ranker(training_data, labels):
    MAX_SAMPLES = 200000
    if len(training_data) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(training_data), MAX_SAMPLES, replace=False)
        training_data = [training_data[i] for i in idx]
        labels = [labels[i] for i in idx]
    candidates  = [t[0] for t in training_data]
    correct_ans = [t[1] for t in training_data]
    articles    = [t[2] for t in training_data]
    try:
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
        vectorizer.fit(candidates + correct_ans)
        cand_vecs = vectorizer.transform(candidates)
        corr_vecs = vectorizer.transform(correct_ans)
        norms_c   = np.asarray(cand_vecs.multiply(cand_vecs).sum(axis=1)).ravel() ** 0.5
        norms_r   = np.asarray(corr_vecs.multiply(corr_vecs).sum(axis=1)).ravel() ** 0.5
        dots      = np.asarray(cand_vecs.multiply(corr_vecs).sum(axis=1)).ravel()
        denom     = norms_c * norms_r
        cos_sims  = np.where(denom > 0, dots / denom, 0.0)
    except Exception:
        cos_sims = np.zeros(len(candidates))
    char_matches = np.array([character_match_score(c, a) for c, a in zip(candidates, correct_ans)])
    freq_counts  = np.array([passage_frequency_count(c, art) for c, art in zip(candidates, articles)])
    X = np.column_stack([cos_sims, char_matches, freq_counts])
    y = np.array(labels)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

def generate_distractors_ml(ranker, article, correct_answer, top_k=3):
    candidates = extract_candidates(article, correct_answer)
    
    if len(candidates) == 0:
        return []
    
    features = extract_features(candidates, correct_answer, article)
    scores = ranker.predict_proba(features)[:, 1]
    
    candidate_scores = list(zip(candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    distractors = [cand for cand, score in candidate_scores[:top_k]]
    return distractors

def generate_distractors_word2vec(correct_answer, article, top_k=3):
    try:
        from gensim.models import KeyedVectors
        word2vec = KeyedVectors.load_word2vec_format('word2vec-google-news-300', binary=True)
    except:
        return []
    
    correct_words = correct_answer.lower().split()
    article_words = set(article.lower().split())
    
    all_neighbours = []
    
    for word in correct_words:
        if word in word2vec:
            try:
                neighbours = word2vec.most_similar(word, topn=20)
                for neighbour, _ in neighbours:
                    if neighbour.lower() not in article_words and len(neighbour) > 2:
                        all_neighbours.append(neighbour)
            except:
                continue
    
    return list(set(all_neighbours))[:top_k]

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

def evaluate_accuracy(df, ranker):
    correct, total = 0, 0
    for idx, row in df.iterrows():
        if row.get('answer') not in ['A', 'B', 'C', 'D']:
            continue
        article        = str(row['article']) if not pd.isna(row['article']) else ''
        correct_answer = str(row[row['answer']])
        if not article or not correct_answer:
            continue
        dists = generate_distractors_ml(ranker, article, correct_answer, top_k=1)
        if dists and dists[0].lower() != correct_answer.lower():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


print("Creating training data (5000 row sample)...")
sample_df = train_df.sample(n=5000, random_state=42)
training_data, labels = create_training_data(sample_df)
print(f"Training samples: {len(training_data)}")

print("Training ML ranker (batch mode)...")
ranker = train_ranker(training_data, labels)
print("Ranker trained.")

print("Evaluating on 500 val samples...")
results = []
eval_count = 0
for idx, row in val_df.head(500).iterrows():
    if row.get('answer') not in ['A', 'B', 'C', 'D']:
        continue
    article        = str(row['article']) if not pd.isna(row['article']) else ''
    correct_answer = str(row[row['answer']])
    wrong_options  = [str(row[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != row['answer']]
    if not article or not correct_answer:
        continue
    eval_count += 1
    if eval_count % 10 == 0:
        print(f"  Progress: {eval_count}/500 samples done...")
    ml_d   = generate_distractors_ml(ranker, article, correct_answer)
    w2v_d  = generate_distractors_word2vec(correct_answer, article)
    ml_p,  ml_r,  ml_f  = evaluate_distractors(ml_d,  wrong_options)
    w2v_p, w2v_r, w2v_f = evaluate_distractors(w2v_d, wrong_options)
    results.append({'question_id': row['id'],
                    'ml_precision': ml_p,   'ml_recall': ml_r,   'ml_f1': ml_f,
                    'w2v_precision': w2v_p, 'w2v_recall': w2v_r, 'w2v_f1': w2v_f,
                    'num_ml_distractors': len(ml_d), 'num_w2v_distractors': len(w2v_d)})

results_df = pd.DataFrame(results)
print("\nResults Summary (token-level F1):")
print(results_df[['ml_precision', 'ml_recall', 'ml_f1']].mean())
print(f"Ranker Accuracy: {evaluate_accuracy(val_df.head(500), ranker):.4f}")

os.makedirs('/content/drive/MyDrive/models', exist_ok=True)
results_df.to_csv('/content/drive/MyDrive/models/distractor_results.csv', index=False)
with open('/content/drive/MyDrive/models/distractor_model.pkl', 'wb') as f:
    pickle.dump(ranker, f)
print("Done. Files saved to Drive.")
