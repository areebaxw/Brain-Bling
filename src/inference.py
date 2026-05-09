"""
INFERENCE.PY — Unified Inference API
======================================
Provides a single programmatic interface to:
  - generate_quiz(article)         → question, options, correct_answer
  - verify_answer(article, q, opt) → is_correct (bool), confidence (float)
  - generate_distractors(article, answer) → [distractor1, distractor2, distractor3]
  - generate_hints(article, question, answer) → [hint1, hint2, hint3]

All inference logic lives in api/app.py (FastAPI endpoints).
This module calls those endpoints locally or directly via the loaded models.
"""

import os
import sys
import pickle
import numpy as np
import re
from typing import List, Tuple, Optional

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_A_DIR = os.path.join(MODELS_DIR, "model_a", "traditional")
MODEL_B_DIR = os.path.join(MODELS_DIR, "model_b", "traditional")
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','is','are','was','were','be','been','have','has','had','do','does',
    'did','will','would','could','should','may','might','it','its','this','that'
}


def _load_pkl(path, label):
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[OK] {label} loaded")
        return obj
    print(f"[WARN] {label} not found at {path}")
    return None


def load_all_models():
    models = {
        "trained_models":    _load_pkl(f"{MODEL_A_DIR}/trained_models.pkl",    "trained_models"),
        "ensemble_models":   _load_pkl(f"{MODEL_A_DIR}/ensemble_models.pkl",   "ensemble_models"),
        "distractor_model":  _load_pkl(f"{MODEL_B_DIR}/distractor_model.pkl",  "distractor_model"),
        "hint_model_bundle": _load_pkl(f"{MODEL_B_DIR}/hint_model.pkl",        "hint_model"),
        "question_ranker":   _load_pkl(f"{MODEL_B_DIR}/question_ranker.pkl",   "question_ranker"),
        "onehot_vectorizer": _load_pkl(f"{DATA_DIR}/onehot_vectorizer.pkl",    "onehot_vectorizer"),
    }
    return models


def verify_answer(models: dict, article: str, question: str,
                  selected_option: str, options: List[str]) -> Tuple[bool, float]:
    """
    Use the best trained Model A classifier to verify if selected_option is correct.
    Returns (is_correct: bool, confidence: float).
    """
    trained = models.get("trained_models")
    onehot  = models.get("onehot_vectorizer")
    if trained is None or onehot is None:
        return False, 0.0

    clf = trained.get("Random Forest") or trained.get("Logistic Regression")
    if clf is None:
        return False, 0.0

    combined = f"{article} {question} {selected_option}"
    X = onehot.transform([combined])
    proba = clf.predict_proba(X)[0][1]
    return proba >= 0.5, float(proba)


def generate_hints(models: dict, article: str, question: str,
                   answer: str, top_k: int = 3) -> List[str]:
    """
    Extract top-K hint sentences from article using the trained hint LR model.
    Falls back to keyword overlap heuristic if model unavailable.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', article) if len(s.strip()) > 10]
    if not sentences:
        return []

    bundle = models.get("hint_model_bundle")
    if bundle and "hint_model" in bundle:
        hint_model = bundle["hint_model"]
        q_words = set(question.lower().split()) - STOPWORDS
        a_words = set(answer.lower().split()) - STOPWORDS
        total   = len(sentences)
        feats   = []
        for i, sent in enumerate(sentences):
            s_words   = set(sent.lower().split()) - STOPWORDS
            q_overlap = len(s_words & q_words) / len(q_words) if q_words else 0.0
            a_overlap = len(s_words & a_words) / len(a_words) if a_words else 0.0
            pos_norm  = i / max(total - 1, 1)
            length    = len(sent.split())
            len_norm  = min(length / 30.0, 1.0)
            is_first  = 1 if i == 0 else 0
            is_last   = 1 if i == total - 1 else 0
            feats.append([q_overlap, a_overlap, pos_norm, length, len_norm, is_first, is_last])
        probs    = hint_model.predict_proba(np.array(feats))[:, 1]
        top_idx  = np.argsort(probs)[::-1][:top_k]
        return [sentences[i] for i in sorted(top_idx)]

    # Fallback: keyword overlap heuristic
    q_words = set(question.lower().split()) - STOPWORDS
    scored  = [(len(set(s.lower().split()) & q_words), s) for s in sentences]
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_k]]


if __name__ == "__main__":
    print("=" * 80)
    print("INFERENCE MODULE — loading models")
    print("=" * 80)
    models = load_all_models()

    article = (
        "Marie Curie was a physicist and chemist who conducted pioneering research "
        "on radioactivity. She was the first woman to win a Nobel Prize and the only "
        "person to win Nobel Prizes in two different sciences."
    )
    question = "What field did Marie Curie conduct research in?"
    answer   = "radioactivity"

    print("\n[HINTS]")
    hints = generate_hints(models, article, question, answer)
    for i, h in enumerate(hints, 1):
        print(f"  Hint {i}: {h}")

    print("\n[VERIFY ANSWER]")
    is_correct, conf = verify_answer(models, article, question, answer, [answer])
    print(f"  is_correct={is_correct}  confidence={conf:.4f}")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
