"""
PHASE 4 - MODEL B: DISTRACTOR + HINT GENERATOR
==============================================

Tasks:
1. Distractor Generation
   - Generate three wrong options (A/B/C) that are plausible, incorrect, and diverse
2. Hint Generation
   - Extract relevant article sentences
   - Rank hints using cosine similarity, frequency-based scoring, and ML ranking

Outputs:
  - model_b_distractors_hints.csv
  - model_b_hint_candidates.csv
  - model_b_summary.csv
  - model_b_artifacts.pkl
"""

import argparse
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "there", "they", "this", "to", "was", "were", "will",
    "with", "what", "which", "who", "whom", "where", "when", "why", "how",
    "into", "than", "then", "them", "his", "her", "she", "you", "your", "we",
    "our", "us", "i", "me", "my", "mine", "do", "does", "did", "can", "could",
    "should", "would", "may", "might", "must", "also", "about", "over", "under",
    "after", "before", "during", "up", "down", "out", "if", "because", "so",
    "but", "not", "no", "yes", "all", "any", "each", "few", "more", "most"
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    text = normalize_text(text)
    return re.findall(r"[a-z0-9]+", text)


def content_tokens(text):
    return [t for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(article):
    if not isinstance(article, str):
        return []
    normalized = re.sub(r"\s+", " ", article).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [p.strip() for p in parts if len(content_tokens(p)) >= 4]

    # Processed datasets may not have punctuation. Create pseudo-sentences by chunking.
    if len(sentences) <= 1:
        tokens = normalized.split()
        if len(tokens) >= 24:
            chunk_size = 28
            stride = 20
            chunked = []
            for start in range(0, len(tokens), stride):
                chunk_tokens = tokens[start:start + chunk_size]
                if len(chunk_tokens) < 8:
                    continue
                chunked.append(" ".join(chunk_tokens))
            if chunked:
                sentences = chunked

    return sentences


def rowwise_cosine_sparse(matrix_a, matrix_b):
    """
    Cosine similarity row-by-row for two sparse matrices of same shape.
    """
    numerator = np.asarray(matrix_a.multiply(matrix_b).sum(axis=1)).ravel()
    norm_a = np.sqrt(np.asarray(matrix_a.multiply(matrix_a).sum(axis=1)).ravel()) + 1e-12
    norm_b = np.sqrt(np.asarray(matrix_b.multiply(matrix_b).sum(axis=1)).ravel()) + 1e-12
    return numerator / (norm_a * norm_b)


def tile_sparse_row(row_matrix, n_rows):
    """
    Repeat a single sparse row matrix n_rows times.
    """
    return sp.vstack([row_matrix] * n_rows)


def frequency_overlap_score(source_text, candidate_text, top_k=12):
    """
    Frequency-based overlap score between source and candidate.
    """
    source = content_tokens(source_text)
    candidate = set(content_tokens(candidate_text))
    if not source or not candidate:
        return 0.0

    counts = Counter(source)
    top_terms = [term for term, _ in counts.most_common(top_k)]
    if not top_terms:
        return 0.0

    matched_weight = sum(counts[t] for t in top_terms if t in candidate)
    total_weight = sum(counts[t] for t in top_terms) + 1e-12
    return matched_weight / total_weight


def jaccard_similarity(text_a, text_b):
    tokens_a = set(content_tokens(text_a))
    tokens_b = set(content_tokens(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))


def load_clean_splits(processed_dir):
    train_path = os.path.join(processed_dir, "train_clean.csv")
    val_path = os.path.join(processed_dir, "val_clean.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for frame in [train_df, val_df]:
        if "Unnamed: 0" in frame.columns:
            frame.drop(columns=["Unnamed: 0"], inplace=True)

    return train_df, val_df


def sample_dataframe(df, max_rows, random_state):
    if max_rows is None or max_rows <= 0 or max_rows >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def option_text_for_answer(row):
    answer = row.get("answer", "")
    if answer in ["A", "B", "C", "D"]:
        return str(row.get(answer, ""))
    return str(row.get("A", ""))


def build_distractor_training_frame(df):
    records = []
    for _, row in df.iterrows():
        correct_label = str(row.get("answer", ""))
        correct_option = option_text_for_answer(row)

        for label in ["A", "B", "C", "D"]:
            option_text = str(row.get(label, ""))
            records.append(
                {
                    "article": str(row.get("article", "")),
                    "question": str(row.get("question", "")),
                    "correct_option": correct_option,
                    "candidate_option": option_text,
                    "label_is_distractor": 1 if label != correct_label else 0,
                }
            )
    return pd.DataFrame(records)


def build_distractor_features(frame, vectorizer):
    question_vec = vectorizer.transform(frame["question"].astype(str).tolist())
    article_vec = vectorizer.transform(frame["article"].astype(str).tolist())
    correct_vec = vectorizer.transform(frame["correct_option"].astype(str).tolist())
    option_vec = vectorizer.transform(frame["candidate_option"].astype(str).tolist())

    sim_q_opt = rowwise_cosine_sparse(question_vec, option_vec)
    sim_a_opt = rowwise_cosine_sparse(article_vec, option_vec)
    sim_c_opt = rowwise_cosine_sparse(correct_vec, option_vec)

    freq_overlap = np.array(
        [
            frequency_overlap_score(q, opt)
            for q, opt in zip(frame["question"].astype(str), frame["candidate_option"].astype(str))
        ]
    )

    option_len = np.array([len(content_tokens(x)) for x in frame["candidate_option"].astype(str)])
    unique_ratio = np.array(
        [
            (
                len(set(content_tokens(text))) / (len(content_tokens(text)) + 1e-12)
                if len(content_tokens(text)) > 0
                else 0.0
            )
            for text in frame["candidate_option"].astype(str)
        ]
    )

    return np.column_stack(
        [sim_q_opt, sim_a_opt, sim_c_opt, freq_overlap, option_len, unique_ratio]
    )


def train_distractor_ranker(train_df, rank_train_rows=3000, random_state=42):
    sampled = sample_dataframe(train_df, rank_train_rows, random_state)
    train_frame = build_distractor_training_frame(sampled)

    corpus = (
        train_frame["article"].astype(str).tolist()
        + train_frame["question"].astype(str).tolist()
        + train_frame["correct_option"].astype(str).tolist()
        + train_frame["candidate_option"].astype(str).tolist()
    )

    vectorizer = TfidfVectorizer(
        max_features=25000, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    vectorizer.fit(corpus)

    X = build_distractor_features(train_frame, vectorizer)
    y = train_frame["label_is_distractor"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print("\n[DISTRACTOR RANKER]")
    print("=" * 80)
    print(f"Training samples: {len(train_frame)}")
    print(f"Validation accuracy: {acc:.4f}")

    return {
        "model": model,
        "vectorizer": vectorizer,
        "feature_names": [
            "sim_question_option",
            "sim_article_option",
            "sim_correct_option",
            "freq_overlap",
            "option_len",
            "unique_ratio",
        ],
        "val_accuracy": acc,
    }


def build_option_pool(train_df, val_df, pool_rows=6000, random_state=42):
    train_sample = sample_dataframe(train_df, pool_rows, random_state)
    val_sample = sample_dataframe(val_df, pool_rows, random_state + 7)
    combined = pd.concat([train_sample, val_sample], ignore_index=True)

    rows = []
    for _, row in combined.iterrows():
        for label in ["A", "B", "C", "D"]:
            text = str(row.get(label, "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "source_id": row.get("id", ""),
                    "option_label": label,
                    "option_text": text,
                    "option_text_norm": normalize_text(text),
                }
            )

    pool_df = pd.DataFrame(rows).drop_duplicates(subset=["option_text_norm"]).reset_index(drop=True)
    return pool_df


def train_option_retriever(pool_df):
    vectorizer = TfidfVectorizer(
        max_features=30000, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    pool_matrix = vectorizer.fit_transform(pool_df["option_text"].astype(str).tolist())
    return vectorizer, pool_matrix


def choose_diverse_distractors(candidate_records, n=3):
    chosen = []
    used = set()

    while len(chosen) < n and candidate_records:
        best_idx = None
        best_score = -1e9

        for idx, rec in enumerate(candidate_records):
            if rec["option_text_norm"] in used:
                continue

            if not chosen:
                diversity_bonus = 1.0
            else:
                max_sim = max(
                    jaccard_similarity(rec["option_text"], prev["option_text"]) for prev in chosen
                )
                diversity_bonus = 1.0 - max_sim

            score = rec["base_score"] + (0.18 * diversity_bonus)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        picked = candidate_records.pop(best_idx)
        used.add(picked["option_text_norm"])
        chosen.append(picked)

    return chosen


def generate_distractors_for_row(
    row,
    pool_df,
    retrieval_vectorizer,
    pool_matrix,
    distractor_ranker,
    top_k=200,
):
    current_options = {normalize_text(str(row.get(label, ""))) for label in ["A", "B", "C", "D"]}
    correct_answer_text = option_text_for_answer(row)
    query_text = f"{row.get('question', '')} {correct_answer_text}"

    query_vec = retrieval_vectorizer.transform([query_text])
    similarities = (query_vec @ pool_matrix.T).toarray().ravel()

    if len(similarities) == 0:
        return [], []

    top_k = min(top_k, len(similarities))
    top_idx_unsorted = np.argpartition(-similarities, top_k - 1)[:top_k]
    top_idx = top_idx_unsorted[np.argsort(-similarities[top_idx_unsorted])]

    candidate_texts = []
    retrieval_scores = []
    pool_records = []

    for idx in top_idx:
        rec = pool_df.iloc[int(idx)]
        text = str(rec["option_text"])
        text_norm = str(rec["option_text_norm"])

        if not text_norm or text_norm in current_options:
            continue

        tokens = content_tokens(text)
        if len(tokens) < 1 or len(tokens) > 20:
            continue

        candidate_texts.append(text)
        retrieval_scores.append(float(similarities[int(idx)]))
        pool_records.append(rec)

    if not candidate_texts:
        return [], []

    feature_frame = pd.DataFrame(
        {
            "article": [str(row.get("article", ""))] * len(candidate_texts),
            "question": [str(row.get("question", ""))] * len(candidate_texts),
            "correct_option": [correct_answer_text] * len(candidate_texts),
            "candidate_option": candidate_texts,
        }
    )

    X_candidates = build_distractor_features(feature_frame, distractor_ranker["vectorizer"])
    ml_proba = distractor_ranker["model"].predict_proba(X_candidates)[:, 1]

    candidate_records = []
    for idx, (text, sim_val, ml_val) in enumerate(zip(candidate_texts, retrieval_scores, ml_proba)):
        base_score = (0.62 * sim_val) + (0.38 * float(ml_val))
        candidate_records.append(
            {
                "option_text": text,
                "option_text_norm": normalize_text(text),
                "retrieval_similarity": float(sim_val),
                "ml_distractor_probability": float(ml_val),
                "base_score": float(base_score),
                "source_id": pool_records[idx]["source_id"],
            }
        )

    candidate_records.sort(key=lambda x: x["base_score"], reverse=True)
    selected = choose_diverse_distractors(candidate_records, n=3)

    # fallback: use original incorrect options if retrieval didn't produce enough
    if len(selected) < 3:
        for label in ["A", "B", "C", "D"]:
            if label == row.get("answer", ""):
                continue
            text = str(row.get(label, "")).strip()
            text_norm = normalize_text(text)
            if not text or any(text_norm == s["option_text_norm"] for s in selected):
                continue
            selected.append(
                {
                    "option_text": text,
                    "option_text_norm": text_norm,
                    "retrieval_similarity": 0.0,
                    "ml_distractor_probability": 0.0,
                    "base_score": 0.0,
                    "source_id": "fallback_original_option",
                }
            )
            if len(selected) == 3:
                break

    return selected[:3], candidate_records[:20]


def fit_hint_ranker(train_df, hint_train_rows=1500, random_state=42):
    sampled = sample_dataframe(train_df, hint_train_rows, random_state)

    corpus = []
    for _, row in sampled.iterrows():
        corpus.append(str(row.get("question", "")))
        corpus.append(option_text_for_answer(row))
        corpus.extend(split_sentences(str(row.get("article", ""))))

    vectorizer = TfidfVectorizer(
        max_features=25000, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    vectorizer.fit(corpus)

    feature_rows = []
    labels = []

    for _, row in sampled.iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        answer_text = option_text_for_answer(row)
        qa_text = f"{question} {answer_text}"

        sentences = split_sentences(article)
        if not sentences:
            continue

        sent_vec = vectorizer.transform(sentences)
        q_vec = tile_sparse_row(vectorizer.transform([question]), sent_vec.shape[0])
        a_vec = tile_sparse_row(vectorizer.transform([answer_text]), sent_vec.shape[0])
        qa_vec = tile_sparse_row(vectorizer.transform([qa_text]), sent_vec.shape[0])

        sim_q = rowwise_cosine_sparse(sent_vec, q_vec)
        sim_a = rowwise_cosine_sparse(sent_vec, a_vec)
        sim_qa = rowwise_cosine_sparse(sent_vec, qa_vec)

        freq_scores = np.array([frequency_overlap_score(qa_text, s) for s in sentences])
        answer_tokens = set(content_tokens(answer_text))
        best_idx = int(np.argmax(sim_qa))
        sim_threshold = float(np.median(sim_qa))

        for i, sentence in enumerate(sentences):
            sent_tokens = set(content_tokens(sentence))
            has_answer_token = len(answer_tokens.intersection(sent_tokens)) > 0
            label = 1 if (i == best_idx or (has_answer_token and sim_qa[i] >= sim_threshold)) else 0

            position = 1.0 - (i / max(len(sentences) - 1, 1))

            feature_rows.append([sim_q[i], sim_a[i], sim_qa[i], freq_scores[i], position])
            labels.append(label)

    X = np.array(feature_rows, dtype=float)
    y = np.array(labels, dtype=int)

    # Safety fallback for weak-label edge cases (all-0 or all-1)
    if len(np.unique(y)) < 2 and len(X) > 0:
        sim_col = X[:, 2]
        threshold = np.quantile(sim_col, 0.75)
        y = (sim_col >= threshold).astype(int)
        if len(np.unique(y)) < 2:
            threshold = np.quantile(sim_col, 0.50)
            y = (sim_col >= threshold).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = LogisticRegression(
        max_iter=1200, random_state=random_state, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print("\n[HINT RANKER]")
    print("=" * 80)
    print(f"Training candidates: {len(y)}")
    print(f"Validation accuracy (weak labels): {acc:.4f}")

    return {
        "model": model,
        "vectorizer": vectorizer,
        "feature_names": ["sim_question", "sim_answer", "sim_qa", "freq_overlap", "position"],
        "val_accuracy": acc,
    }


def generate_hints_for_row(row, hint_ranker, top_n=3):
    article = str(row.get("article", ""))
    question = str(row.get("question", ""))
    answer_text = option_text_for_answer(row)
    qa_text = f"{question} {answer_text}"

    sentences = split_sentences(article)
    if not sentences:
        return [], []

    vectorizer = hint_ranker["vectorizer"]
    sent_vec = vectorizer.transform(sentences)
    q_vec = tile_sparse_row(vectorizer.transform([question]), sent_vec.shape[0])
    a_vec = tile_sparse_row(vectorizer.transform([answer_text]), sent_vec.shape[0])
    qa_vec = tile_sparse_row(vectorizer.transform([qa_text]), sent_vec.shape[0])

    sim_q = rowwise_cosine_sparse(sent_vec, q_vec)
    sim_a = rowwise_cosine_sparse(sent_vec, a_vec)
    sim_qa = rowwise_cosine_sparse(sent_vec, qa_vec)
    freq_scores = np.array([frequency_overlap_score(qa_text, s) for s in sentences], dtype=float)
    positions = np.array(
        [1.0 - (i / max(len(sentences) - 1, 1)) for i in range(len(sentences))], dtype=float
    )

    X = np.column_stack([sim_q, sim_a, sim_qa, freq_scores, positions])
    ml_scores = hint_ranker["model"].predict_proba(X)[:, 1]

    final_scores = (0.45 * sim_qa) + (0.20 * freq_scores) + (0.35 * ml_scores)
    order = np.argsort(-final_scores)

    hint_candidates = []
    for idx in order:
        hint_candidates.append(
            {
                "sentence": sentences[int(idx)],
                "sim_qa": float(sim_qa[int(idx)]),
                "freq_score": float(freq_scores[int(idx)]),
                "ml_score": float(ml_scores[int(idx)]),
                "final_score": float(final_scores[int(idx)]),
            }
        )

    selected = []
    for cand in hint_candidates:
        if any(jaccard_similarity(cand["sentence"], existing) > 0.85 for existing in selected):
            continue
        selected.append(cand["sentence"])
        if len(selected) == top_n:
            break

    return selected, hint_candidates[:10]


def run_model_b(
    processed_dir="data/processed",
    output_dir="models",
    max_questions=200,
    distractor_pool_rows=6000,
    distractor_train_rows=3000,
    hint_train_rows=1500,
    random_state=42,
):
    print("=" * 80)
    print("PHASE 4 - MODEL B: DISTRACTOR + HINT GENERATOR")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[LOADING DATA]")
    print("=" * 80)
    train_df, val_df = load_clean_splits(processed_dir)
    print(f"[OK] Train rows: {len(train_df)}")
    print(f"[OK] Validation rows: {len(val_df)}")

    distractor_ranker = train_distractor_ranker(
        train_df, rank_train_rows=distractor_train_rows, random_state=random_state
    )
    hint_ranker = fit_hint_ranker(
        train_df, hint_train_rows=hint_train_rows, random_state=random_state
    )

    print("\n[OPTION POOL + RETRIEVER]")
    print("=" * 80)
    pool_df = build_option_pool(
        train_df, val_df, pool_rows=distractor_pool_rows, random_state=random_state
    )
    retrieval_vectorizer, pool_matrix = train_option_retriever(pool_df)
    print(f"[OK] Option pool size: {len(pool_df)}")
    print(f"[OK] Retriever matrix shape: {pool_matrix.shape}")

    val_subset = sample_dataframe(val_df, max_questions, random_state + 101)
    print(f"\n[GENERATION] Running on {len(val_subset)} validation questions...")

    output_rows = []
    hint_candidate_rows = []
    distractor_from_fallback = 0

    for idx, row in val_subset.iterrows():
        distractors, distractor_candidates = generate_distractors_for_row(
            row=row,
            pool_df=pool_df,
            retrieval_vectorizer=retrieval_vectorizer,
            pool_matrix=pool_matrix,
            distractor_ranker=distractor_ranker,
        )

        if any(d["source_id"] == "fallback_original_option" for d in distractors):
            distractor_from_fallback += 1

        hints, hint_candidates = generate_hints_for_row(row, hint_ranker, top_n=3)

        while len(distractors) < 3:
            distractors.append(
                {
                    "option_text": "",
                    "option_text_norm": "",
                    "retrieval_similarity": 0.0,
                    "ml_distractor_probability": 0.0,
                    "base_score": 0.0,
                    "source_id": "empty_fill",
                }
            )
        while len(hints) < 3:
            hints.append("")

        correct_label = str(row.get("answer", ""))
        correct_text = option_text_for_answer(row)

        output_rows.append(
            {
                "id": row.get("id", idx),
                "question": row.get("question", ""),
                "correct_label": correct_label,
                "correct_answer_text": correct_text,
                "distractor_A": distractors[0]["option_text"],
                "distractor_B": distractors[1]["option_text"],
                "distractor_C": distractors[2]["option_text"],
                "hint_1": hints[0],
                "hint_2": hints[1],
                "hint_3": hints[2],
            }
        )

        for cand in hint_candidates:
            hint_candidate_rows.append(
                {
                    "id": row.get("id", idx),
                    "question": row.get("question", ""),
                    "candidate_hint": cand["sentence"],
                    "sim_qa": cand["sim_qa"],
                    "freq_score": cand["freq_score"],
                    "ml_score": cand["ml_score"],
                    "final_score": cand["final_score"],
                }
            )

        if (idx + 1) % 50 == 0 or (idx + 1) == len(val_subset):
            print(f"  Processed {idx + 1}/{len(val_subset)}")

    output_df = pd.DataFrame(output_rows)
    hint_candidates_df = pd.DataFrame(hint_candidate_rows)

    outputs_path = os.path.join(output_dir, "model_b_distractors_hints.csv")
    hints_path = os.path.join(output_dir, "model_b_hint_candidates.csv")
    summary_path = os.path.join(output_dir, "model_b_summary.csv")
    artifacts_path = os.path.join(output_dir, "model_b_artifacts.pkl")

    output_df.to_csv(outputs_path, index=False)
    hint_candidates_df.to_csv(hints_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_questions": len(val_subset),
                "distractor_ranker_val_acc": distractor_ranker["val_accuracy"],
                "hint_ranker_val_acc": hint_ranker["val_accuracy"],
                "fallback_distractor_rows": distractor_from_fallback,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    with open(artifacts_path, "wb") as f:
        pickle.dump(
            {
                "distractor_ranker": distractor_ranker,
                "hint_ranker": hint_ranker,
                "retrieval_vectorizer": retrieval_vectorizer,
            },
            f,
        )

    print("\n[SAVED OUTPUTS]")
    print("=" * 80)
    print(f"Saved: {outputs_path}")
    print(f"Saved: {hints_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {artifacts_path}")

    print("\n[PHASE 4 COMPLETE]")
    print("=" * 80)
    print(f"Generated rows: {len(output_df)}")
    print(f"Rows needing fallback distractors: {distractor_from_fallback}")

    return {
        "outputs": output_df,
        "hint_candidates": hint_candidates_df,
        "summary": summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 - Model B Distractor + Hint Generator")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--max_questions", type=int, default=200)
    parser.add_argument("--distractor_pool_rows", type=int, default=6000)
    parser.add_argument("--distractor_train_rows", type=int, default=3000)
    parser.add_argument("--hint_train_rows", type=int, default=1500)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_model_b(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        distractor_pool_rows=args.distractor_pool_rows,
        distractor_train_rows=args.distractor_train_rows,
        hint_train_rows=args.hint_train_rows,
        random_state=args.random_state,
    )
