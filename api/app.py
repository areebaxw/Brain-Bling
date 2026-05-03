
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import os
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import re
from collections import Counter
import random

app = FastAPI(title="Brain Bling API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory path
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

MODEL_A_TRAD = os.path.join(MODELS_DIR, "model_a", "traditional")
MODEL_A_NEURAL = os.path.join(MODELS_DIR, "model_a", "neural")
MODEL_B_TRAD = os.path.join(MODELS_DIR, "model_b", "traditional")
MODEL_B_NEURAL = os.path.join(MODELS_DIR, "model_b", "neural")

# Load models on startup
trained_models = None
ensemble_models = None
distractor_model = None
hint_model_bundle = None
onehot_vectorizer = None
bert_model = None
bert_tokenizer = None
sentence_model = None

def load_models():
    global trained_models, ensemble_models, distractor_model, hint_model_bundle, onehot_vectorizer
    global bert_model, bert_tokenizer, sentence_model
    try:
        if os.path.exists(f"{MODEL_A_TRAD}/trained_models.pkl"):
            with open(f"{MODEL_A_TRAD}/trained_models.pkl", "rb") as f:
                trained_models = pickle.load(f)
            print("[OK] trained_models loaded")

        if os.path.exists(f"{MODEL_A_TRAD}/ensemble_models.pkl"):
            with open(f"{MODEL_A_TRAD}/ensemble_models.pkl", "rb") as f:
                ensemble_models = pickle.load(f)
            print("[OK] ensemble_models loaded")

        if os.path.exists(f"{MODEL_B_TRAD}/distractor_model.pkl"):
            with open(f"{MODEL_B_TRAD}/distractor_model.pkl", "rb") as f:
                distractor_model = pickle.load(f)
            print(f"[OK] distractor_model loaded — type: {type(distractor_model)}")

        if os.path.exists(f"{MODEL_B_TRAD}/hint_model.pkl"):
            with open(f"{MODEL_B_TRAD}/hint_model.pkl", "rb") as f:
                hint_model_bundle = pickle.load(f)
            print(f"[OK] hint_model loaded — type: {type(hint_model_bundle)}")

        if os.path.exists(f"{DATA_DIR}/processed/onehot_vectorizer.pkl"):
            with open(f"{DATA_DIR}/processed/onehot_vectorizer.pkl", "rb") as f:
                onehot_vectorizer = pickle.load(f)
            print("[OK] onehot_vectorizer loaded")

        # BERT for answer verification
        if os.path.exists(MODEL_A_NEURAL):
            try:
                from transformers import AutoTokenizer, AutoModelForMultipleChoice
                import torch
                bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_A_NEURAL)
                bert_model = AutoModelForMultipleChoice.from_pretrained(MODEL_A_NEURAL)
                bert_model.eval()
                print("[OK] BERT model loaded")
            except Exception as e:
                print(f"[WARN] BERT not loaded: {e}")

        # Sentence Transformer for distractor + hint
        if os.path.exists(MODEL_B_NEURAL):
            try:
                from sentence_transformers import SentenceTransformer
                sentence_model = SentenceTransformer(MODEL_B_NEURAL)
                print("[OK] SentenceTransformer loaded")
            except Exception as e:
                print(f"[WARN] SentenceTransformer not loaded: {e}")

        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()

load_models()


class ArticleInput(BaseModel):
    article: str
    model_type: Optional[str] = "traditional"

class QuestionInput(BaseModel):
    article: str
    question: str
    options: List[str]
    correct_answer: Optional[str] = None
    model_type: Optional[str] = "traditional"

class AnswerCheck(BaseModel):
    article: str
    question: str
    selected_option: str
    options: List[str]
    correct_answer: Optional[str] = None
    model_type: Optional[str] = "both"  # "bert", "traditional", "both"


STOPWORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','is','are','was','were','be','been','have','has','had','do','does',
    'did','will','would','could','should','may','might','can','this','that',
    'these','those','i','you','he','she','it','we','they','as','into','from'
])


def clean(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def generate_wh_question(sentence):
    """
    Rule-based Wh-word template question generation.
    Transforms a declarative sentence into a question + answer pair.
    Prioritizes proper nouns (Who/Where) for best quality questions.
    Returns (question_str, answer_str) or (None, None).
    """
    words = sentence.split()
    if len(words) < 5:
        return None, None

    # Skip sentences starting with conjunctions or labels — they make bad questions
    skip_starts = ['but', 'and', 'or', 'so', 'however', 'also', 'then', 'yet', 'rates:', 'price:', 'cost:', 'from', 'rooms:', 'the', 'a', 'an']
    if words[0].lower() in skip_starts:
        return None, None

    # Strategy 1 (best): Capitalized proper noun (not sentence start) → Who/Where
    for i, w in enumerate(words):
        if i > 0 and len(w) > 2 and w[0].isupper() and w.lower() not in STOPWORDS:
            answer = w
            j = i + 1
            while j < len(words) and len(words[j]) > 1 and words[j][0].isupper():
                answer += ' ' + words[j]
                j += 1
            remaining = words[:i] + ['_______'] + words[j:]
            if i > 0 and words[i-1].lower() in ['in', 'at', 'near', 'from', 'to', 'around']:
                answer = words[i-1] + ' ' + answer
                remaining = words[:i-1] + ['_______'] + words[j:]
                question = "According to the passage, where " + ' '.join(remaining).lstrip(',').strip() + "?"
            else:
                question = "According to the passage, who " + ' '.join(remaining).lstrip(',').strip() + "?"
            return question, re.sub(r'[,;:]$', '', answer).strip()

    # Strategy 2: Key content word → fill-in-the-blank with "What"
    content_indices = [
        (i, w) for i, w in enumerate(words)
        if len(w) >= 5 and w.lower() not in STOPWORDS and w.isalpha() and i > 0
    ]
    if content_indices:
        mid = len(content_indices) // 2
        idx, answer = content_indices[mid]
        remaining = words[:idx] + ['_______'] + words[idx+1:]
        question = "According to the passage, " + ' '.join(remaining) + "?"
        return question, answer

    return None, None


def rank_questions(questions_with_answers, article):
    """
    Rank candidate questions by fluency and relevance features.
    Uses keyword overlap with article as a simple relevance score.
    """
    article_words = set(clean(article).split()) - STOPWORDS
    scored = []
    for question, answer in questions_with_answers:
        q_words = set(clean(question).split()) - STOPWORDS
        # Relevance: keyword overlap with article
        overlap = len(q_words & article_words) / max(len(q_words), 1)
        # Fluency: prefer questions with 8-15 words
        q_len = len(question.split())
        length_score = 1.0 if 8 <= q_len <= 15 else 0.5
        # Answer quality: prefer 1-3 word answers
        a_len = len(answer.split())
        answer_score = 1.0 if 1 <= a_len <= 3 else 0.5
        score = overlap * length_score * answer_score
        scored.append((question, answer, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def extract_candidates_from_sentence(sentence, correct_answer):
    words = sentence.split()
    candidates = []
    for i in range(len(words)):
        for length in range(1, 4):
            if i + length <= len(words):
                candidate = re.sub(r'[^a-zA-Z0-9 ]', ' ', ' '.join(words[i:i+length])).strip()
                candidate = re.sub(r'\s+', ' ', candidate).strip()
                cand_words = candidate.split()
                stopword_ratio = sum(1 for w in cand_words if w.lower() in STOPWORDS) / max(len(cand_words), 1)
                has_fragment = any(len(w) <= 1 for w in cand_words)
                if (len(candidate) >= 4 and
                    candidate.lower() != correct_answer.lower() and
                    candidate.lower() not in STOPWORDS and
                    not candidate.isdigit() and
                    any(c.isalpha() for c in candidate) and
                    stopword_ratio < 0.4 and
                    not has_fragment):
                    candidates.append(candidate)
    word_freq = Counter(words)
    freq_candidates = [
        re.sub(r'[^a-zA-Z0-9 ]', ' ', w).strip() for w, count in word_freq.items()
        if count >= 2 and len(w) >= 4 and w.lower() not in STOPWORDS
        and w.lower() != correct_answer.lower() and any(c.isalpha() for c in w)
    ]
    candidates.extend(freq_candidates)
    return list(set(c for c in candidates if c))


def extract_candidates(article, correct_answer):
    sentences = split_sentences(article)
    all_candidates = []
    for sentence in sentences:
        all_candidates.extend(extract_candidates_from_sentence(sentence, correct_answer))
    return list(set(all_candidates))


def augment_candidates(correct_answer, article):
    extra = []
    words = correct_answer.split()
    # Only apply negation/morphological variants for short answers (phrase-level)
    if len(words) <= 4:
        extra.append('not ' + correct_answer)
        if any(c.isdigit() for c in correct_answer):
            for n in range(1, 10):
                variant = re.sub(r'\d+', str(n), correct_answer)
                if variant != correct_answer:
                    extra.append(variant)
                    break
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
    # Top frequent content words from article always added
    article_words = article.lower().split()
    word_freq = Counter(article_words)
    top_words = [
        w for w, c in word_freq.most_common(30)
        if len(w) >= 4 and w not in STOPWORDS and w != correct_answer.lower()
    ]
    extra.extend(top_words[:15])
    return list(set(c for c in extra if c and c.lower() != correct_answer.lower()))


def diversity_penalty(candidate, selected):
    if not selected:
        return 0.0
    cand_tokens = set(candidate.lower().split())
    max_overlap = 0.0
    for s in selected:
        s_tokens = set(s.lower().split())
        union = cand_tokens | s_tokens
        if union:
            overlap = len(cand_tokens & s_tokens) / len(union)
            max_overlap = max(max_overlap, overlap)
    return max_overlap


def char_match(a, b):
    candidate_chars = set(a.lower())
    correct_chars = set(b.lower())
    
    intersection = len(candidate_chars & correct_chars)
    union = len(candidate_chars | correct_chars)
    
    return intersection / union if union > 0 else 0


def passage_freq(candidate, article):
    words = article.lower().split()
    candidate_lower = candidate.lower()
    
    count = sum(1 for word in words if candidate_lower in word)
    return count


def get_char_ngram_overlap(text1, text2, n=2):
    def get_ngrams(text, n):
        text = text.lower().replace(' ', '')
        return set(text[i:i+n] for i in range(len(text)-n+1))
    ng1, ng2 = get_ngrams(text1, n), get_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def extract_features(candidates, correct_answer, article):
    # Article-context TF-IDF for better IDF weights (matches training)
    cos_sims = np.zeros(len(candidates))
    try:
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1, token_pattern=r"(?u)\b\w+\b")
        context = [s for s in split_sentences(article) if s.strip()]
        fit_texts = context + candidates + [correct_answer]
        if len(fit_texts) >= 2:
            vectorizer.fit(fit_texts)
            cand_vecs = vectorizer.transform(candidates)
            corr_vec  = vectorizer.transform([correct_answer])
            cos_sims  = cosine_similarity(cand_vecs, corr_vec).ravel()
    except Exception:
        cos_sims = np.array([char_match(c, correct_answer) for c in candidates])
    char_matches  = np.array([char_match(c, correct_answer) for c in candidates])
    freq_counts   = np.array([passage_freq(c, article) for c in candidates])
    word_lengths  = np.array([len(c.split()) for c in candidates])
    char_lengths  = np.array([len(c) for c in candidates])
    char_bigrams  = np.array([get_char_ngram_overlap(c, correct_answer, 2) for c in candidates])
    char_trigrams = np.array([get_char_ngram_overlap(c, correct_answer, 3) for c in candidates])
    ans_wc = max(len(correct_answer.split()), 1)
    wc_match  = np.array([min(len(c.split()), ans_wc) / max(len(c.split()), ans_wc) for c in candidates])
    ans_cl = max(len(correct_answer), 1)
    len_ratio = np.array([min(len(c), ans_cl) / max(len(c), ans_cl) for c in candidates])
    return np.column_stack([cos_sims, char_matches, freq_counts, word_lengths, char_lengths, char_bigrams, char_trigrams, wc_match, len_ratio])


def run_distractor_generation(article, correct_answer, model_type="traditional"):
    candidates = list(set(extract_candidates(article, correct_answer) + augment_candidates(correct_answer, article)))
    if not candidates:
        return []

    ans_wc = len(correct_answer.split())

    # ── Neural: SentenceTransformer semantic similarity ──────────────────
    if model_type == "bert" and sentence_model is not None:
        try:
            print(f"[Distractor NEURAL] scoring {len(candidates)} candidates with SentenceTransformer")
            all_texts   = candidates + [correct_answer]
            embeddings  = sentence_model.encode(all_texts, convert_to_numpy=True)
            cand_embs   = embeddings[:-1]
            correct_emb = embeddings[-1:]
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            sims = cos_sim(cand_embs, correct_emb).ravel()
            # Good distractors: similar but NOT identical (0.3–0.85 range)
            scored = [(c, float(s)) for c, s in zip(candidates, sims)
                      if 0.25 < float(s) < 0.88]
            scored.sort(key=lambda x: x[1], reverse=True)
            pool = scored if len(scored) >= 3 else list(zip(candidates, sims))
            selected = []
            for cand, score in pool:
                if len(selected) >= 3: break
                if diversity_penalty(cand, selected) < 0.35:
                    selected.append(cand)
            if len(selected) < 3:
                for cand, score in pool:
                    if cand not in selected: selected.append(cand)
                    if len(selected) >= 3: break
            print(f"[Distractor NEURAL] result={selected}")
            return selected
        except Exception as e:
            print(f"[Distractor NEURAL ERROR] {e} — falling back to traditional")

    # ── Traditional: Random Forest ranker ────────────────────────────────
    if distractor_model is None:
        return candidates[:3]
    features = extract_features(candidates, correct_answer, article)
    try:
        proba  = distractor_model.predict_proba(features)
        scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        wc_filtered = [(c, s) for c, s in ranked if abs(len(c.split()) - ans_wc) <= 1]
        pool = wc_filtered if len(wc_filtered) >= 3 else ranked
        selected = []
        for cand, score in pool:
            if len(selected) >= 3: break
            if diversity_penalty(cand, selected) < 0.35:
                selected.append(cand)
        if len(selected) < 3:
            for cand, score in pool:
                if cand not in selected: selected.append(cand)
                if len(selected) >= 3: break
        print(f"[Distractor TRAD] result={selected}")
        return selected
    except Exception as e:
        print(f"[Distractor ERROR] {e}")
        return candidates[:3]


def run_hint_generation(article, question, model_type="traditional"):
    sentences = split_sentences(article)
    if not sentences:
        return []

    # ── Neural: SentenceTransformer semantic similarity ──────────────────
    if model_type == "bert" and sentence_model is not None:
        try:
            print("[Hint NEURAL] using SentenceTransformer")
            q_emb    = sentence_model.encode([question], convert_to_numpy=True)
            s_embs   = sentence_model.encode(sentences, convert_to_numpy=True)
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            scores   = cos_sim(s_embs, q_emb).ravel()
            ranked   = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
            n = len(ranked)
            if n >= 3:
                return [ranked[0][0], ranked[n // 2][0], ranked[-1][0]]
            return [r[0] for r in ranked] + [ranked[-1][0]] * (3 - n)
        except Exception as e:
            print(f"[Hint NEURAL ERROR] {e} — falling back to traditional")

    # ── Traditional: Logistic Regression ─────────────────────────────────
    model = None
    if hint_model_bundle is not None:
        model = hint_model_bundle.get('hint_model') if isinstance(hint_model_bundle, dict) else hint_model_bundle
    if model is None:
        return sentences[:3]

    q_words  = set(clean(question).split()) - STOPWORDS
    total    = len(sentences)
    features = []
    for i, sent in enumerate(sentences):
        s_words  = set(clean(sent).split()) - STOPWORDS
        overlap  = len(s_words & q_words) / len(q_words) if q_words else 0
        position = i / max(total - 1, 1)
        length   = len(sent.split())
        features.append([overlap, position, length])
    try:
        scores = model.predict_proba(features)[:, 1]
        ranked = sorted(zip(sentences, scores), key=lambda x: x[1])
        n      = len(ranked)
        if n >= 3:   return [ranked[0][0], ranked[n // 2][0], ranked[-1][0]]
        elif n == 2: return [ranked[0][0], ranked[1][0], ranked[1][0]]
        elif n == 1: return [ranked[0][0], ranked[0][0], ranked[0][0]]
    except Exception as e:
        print(f"[Hint ERROR] {e}")
        return sentences[:3]
    return []


def extract_live_lexical_features(article, question, option_text):
    art_clean = clean(article)
    q_words   = set(clean(question).split())
    opt_words = set(clean(option_text).split())
    art_words = set(art_clean.split())
    union         = q_words | opt_words
    jaccard       = len(q_words & opt_words) / len(union) if union else 0.0
    word_density  = len(q_words & opt_words) / len(opt_words) if opt_words else 0.0
    art_len       = max(len(art_clean.split()), 1)
    length_ratio  = len(opt_words) / art_len
    cap_density   = sum(1 for c in option_text if c.isupper()) / max(len(option_text), 1)
    has_numbers   = 1 if any(c.isdigit() for c in option_text) else 0
    common_words  = len(opt_words & art_words)
    avg_word_len  = float(np.mean([len(w) for w in opt_words])) if opt_words else 0.0
    return [jaccard, word_density, length_ratio, cap_density, has_numbers, common_words, avg_word_len]


def run_traditional_ml_verification(article, question, chosen_option):
    if onehot_vectorizer is None:
        return None
    try:
        combined = clean(f"{article} {question} {chosen_option}")
        onehot   = onehot_vectorizer.transform([combined])
        lex      = extract_live_lexical_features(article, question, chosen_option)
        lex_sp   = sp.csr_matrix(np.array(lex).reshape(1, -1))
        X        = sp.hstack([onehot, lex_sp], format='csr')

        model = None
        if ensemble_models is not None:
            if isinstance(ensemble_models, dict):
                model = (ensemble_models.get('soft_voting')
                         or ensemble_models.get('Soft Voting')
                         or ensemble_models.get('hard_voting')
                         or ensemble_models.get('Hard Voting')
                         or list(ensemble_models.values())[0])
            else:
                model = ensemble_models
        elif trained_models is not None:
            if isinstance(trained_models, dict):
                model = (trained_models.get('Logistic Regression')
                         or trained_models.get('logistic_regression')
                         or list(trained_models.values())[0])
            else:
                model = trained_models

        if model is None:
            return None

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            conf  = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        else:
            conf = float(model.predict(X)[0])

        print(f"[TRAD ML verify] confidence={conf:.3f}")
        return float(np.clip(conf, 0.05, 0.95))
    except Exception as e:
        print(f"[TRAD ML verify ERROR] {e}")
        return None


def run_answer_verification(article, question, chosen_option):
    # BERT-based answer verification (primary)
    if bert_model is not None and bert_tokenizer is not None:
        try:
            import torch
            options = [chosen_option, "none of the above", "cannot determine", "not mentioned"]
            inputs = bert_tokenizer(
                [[article[:300] + " " + question, opt] for opt in options],
                return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
            with torch.no_grad():
                logits = bert_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            confidence = float(probs[0].item())
            print(f"[BERT verify] confidence={confidence:.3f}")
            return float(np.clip(confidence, 0.05, 0.95))
        except Exception as e:
            print(f"[BERT verify ERROR] {e}")

    # Fallback: trained traditional ML (LR/Ensemble)
    conf = run_traditional_ml_verification(article, question, chosen_option)
    if conf is not None:
        return conf

    # Final fallback: cosine similarity
    if onehot_vectorizer is not None:
        try:
            ctx = clean(f"{article} {question}")
            opt = clean(chosen_option)
            if ctx.strip() and opt.strip():
                vecs = onehot_vectorizer.transform([ctx, opt])
                sim  = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
                return float(np.clip(sim * 5.0, 0.05, 0.95))
        except Exception as e:
            print(f"[Verify cosine ERROR] {e}")

    return 0.5


@app.get("/")
async def root():
    return {"message": "Brain Bling API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "trained_models":      trained_models is not None,
            "ensemble_models":     ensemble_models is not None,
            "distractor_model":    distractor_model is not None,
            "hint_model":          hint_model_bundle is not None,
            "onehot_vectorizer":   onehot_vectorizer is not None,
            "bert_model":          bert_model is not None,
            "sentence_transformer": sentence_model is not None
        }
    }


@app.get("/metrics")
async def get_metrics():
    try:
        binary   = pd.read_csv(f"{MODEL_A_TRAD}/binary_classification_metrics.csv")
        ensemble = pd.read_csv(f"{MODEL_A_TRAD}/ensemble_binary_metrics.csv")

        # BERT row — from evaluated CSV if available, otherwise show loaded status
        bert_rows = []
        bert_csv  = f"{MODEL_A_TRAD}/bert_vs_traditional.csv"
        if os.path.exists(bert_csv):
            bert_df = pd.read_csv(bert_csv)
            bert_rows = bert_df[bert_df["Type"] == "Neural"].to_dict(orient="records")
        elif bert_model is not None:
            bert_rows = [{
                "Model":     "BERT (RoBERTa-RACE)",
                "Accuracy":  "Run eval",
                "Precision": "Run eval",
                "Recall":    "Run eval",
                "Macro F1":  "Run eval",
                "ROC-AUC":   "-",
                "Type":      "Neural",
                "Note":      "Model loaded. Run evaluate_bert_vs_traditional.py on Colab for scores."
            }]

        return {
            "binary_metrics":   binary.to_dict(orient="records"),
            "ensemble_metrics": ensemble.to_dict(orient="records"),
            "neural_metrics":   bert_rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-article")
async def get_sample_article():
    try:
        df  = pd.read_csv(f"{DATA_DIR}/raw/dev.csv")
        row = df.sample(1).iloc[0]
        return {
            "article":        str(row["article"]),
            "question":       str(row["question"]),
            "options": [
                {"id": "A", "text": str(row["A"])},
                {"id": "B", "text": str(row["B"])},
                {"id": "C", "text": str(row["C"])},
                {"id": "D", "text": str(row["D"])},
            ],
            "correct_answer": str(row["answer"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-race-quiz")
async def generate_race_quiz(input: QuestionInput):
    try:
        correct_letter = input.correct_answer if input.correct_answer else "A"
        correct_idx    = ord(correct_letter) - ord('A')
        correct_text   = input.options[correct_idx] if correct_idx < len(input.options) else input.options[0]

        print(f"\n[RACE QUIZ] article={input.article[:80]} correct={correct_text}")

        distractors = run_distractor_generation(input.article, correct_text, model_type=input.model_type or "traditional")
        while len(distractors) < 3:
            distractors.append(f"option {len(distractors)+1}")

        all_options = [correct_text] + distractors[:3]
        random.shuffle(all_options)

        correct_index  = all_options.index(correct_text)
        correct_answer = chr(65 + correct_index)

        return {
            "question":            input.question,
            "options":             all_options,
            "correct_answer":      correct_answer,
            "correct_option_text": correct_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-quiz")
async def generate_quiz(input: ArticleInput):
    try:
        sentences = split_sentences(input.article)
        
        # Step 1 & 2: Generate Wh-questions from each sentence using templates
        candidates_qa = []
        for sent in sentences:
            q, a = generate_wh_question(sent)
            if q and a:
                candidates_qa.append((q, a))
        
        # Step 3: Rank questions by fluency and relevance
        if candidates_qa:
            ranked = rank_questions(candidates_qa, input.article)
            question = ranked[0][0]
            correct_text = ranked[0][1]
        else:
            # Fallback if no questions could be generated
            question = "What is the main topic discussed in the passage?"
            correct_text = sentences[0].split()[0] if sentences else "topic"

        print(f"\n[QUIZ] question={question[:60]} correct={correct_text}")

        # Generate distractors for the correct answer
        distractors = run_distractor_generation(input.article, correct_text, model_type=input.model_type or "traditional")
        while len(distractors) < 3:
            distractors.append(f"option {len(distractors)+1}")

        all_options = [correct_text] + distractors[:3]
        random.shuffle(all_options)

        correct_index  = all_options.index(correct_text)
        correct_answer = chr(65 + correct_index)

        return {
            "question":            question,
            "options":             all_options,
            "correct_answer":      correct_answer,
            "correct_option_text": correct_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-distractors")
async def generate_distractors_endpoint(input: QuestionInput):
    try:
        correct_letter = input.correct_answer if input.correct_answer else "A"
        correct_idx    = ord(correct_letter) - ord('A')
        correct_text   = input.options[correct_idx] if correct_idx < len(input.options) else input.options[0]

        distractors = run_distractor_generation(input.article, correct_text)
        while len(distractors) < 3:
            distractors.append(f"option {len(distractors)+1}")

        return {"distractors": distractors[:3]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-hints")
async def generate_hints_endpoint(input: QuestionInput):
    try:
        hints = run_hint_generation(input.article, input.question, model_type=input.model_type or "traditional")
        while len(hints) < 3:
            hints.append("Look carefully at the passage for more context.")
        return {"hints": hints[:3]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/check-answer")
async def check_answer(input: AnswerCheck):
    try:
        if not input.correct_answer:
            raise HTTPException(status_code=400, detail="correct_answer not provided")

        correct_idx  = ord(input.correct_answer) - ord('A')
        correct_text = input.options[correct_idx] if correct_idx < len(input.options) else input.options[0]
        selected_idx  = ord(input.selected_option) - ord('A')
        selected_text = input.options[selected_idx] if selected_idx < len(input.options) else input.options[0]
        is_correct = selected_text == correct_text
        model_type = (input.model_type or "both").lower()

        result = {
            "is_correct":           is_correct,
            "correct_answer":       input.correct_answer,
            "correct_option_text":  correct_text,
            "selected_option_text": selected_text,
        }

        # BERT confidence
        if model_type in ("bert", "both"):
            bert_conf = None
            if bert_model is not None and bert_tokenizer is not None:
                try:
                    import torch
                    options = [selected_text, "none of the above", "cannot determine", "not mentioned"]
                    enc = bert_tokenizer(
                        [[input.article[:300] + " " + input.question, opt] for opt in options],
                        return_tensors="pt", padding=True, truncation=True, max_length=512
                    )
                    enc = {k: v.unsqueeze(0) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = bert_model(**enc).logits
                    probs = torch.softmax(logits, dim=-1)[0]
                    bert_conf = round(float(np.clip(probs[0].item(), 0.05, 0.95)), 4)
                except Exception as e:
                    print(f"[BERT ERROR] {e}")
            result["bert_confidence"]    = bert_conf
            result["bert_explanation"]   = f"BERT confidence: {bert_conf:.2%}" if bert_conf is not None else "BERT not loaded"

        # Traditional (trained LR/Ensemble ML models)
        if model_type in ("traditional", "both"):
            trad_conf = run_traditional_ml_verification(input.article, input.question, selected_text)
            if trad_conf is None and onehot_vectorizer is not None:
                try:
                    ctx  = clean(f"{input.article} {input.question}")
                    opt  = clean(selected_text)
                    vecs = onehot_vectorizer.transform([ctx, opt])
                    sim  = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
                    trad_conf = round(float(np.clip(sim * 5.0, 0.05, 0.95)), 4)
                except Exception as e:
                    print(f"[TRAD ERROR] {e}")
            if trad_conf is not None:
                trad_conf = round(trad_conf, 4)
            result["trad_confidence"]    = trad_conf
            result["trad_explanation"]   = f"Traditional ML confidence: {trad_conf:.2%}" if trad_conf is not None else "Traditional not loaded"

        # Primary confidence for backward compat
        if model_type == "bert":
            result["confidence"]  = result.get("bert_confidence", 0.5)
            result["explanation"] = result.get("bert_explanation", "")
        elif model_type == "traditional":
            result["confidence"]  = result.get("trad_confidence", 0.5)
            result["explanation"] = result.get("trad_explanation", "")
        else:
            bert_c = result.get("bert_confidence")
            trad_c = result.get("trad_confidence")
            result["confidence"]  = bert_c if bert_c is not None else (trad_c or 0.5)
            result["explanation"] = f"BERT: {bert_c:.2%} | Traditional: {trad_c:.2%}" if (bert_c and trad_c) else "Partial models loaded"

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)