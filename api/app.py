from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import os
import re
import random
from typing import List, Optional
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app = FastAPI(title="Brain Bling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

trained_models    = None
ensemble_models   = None
distractor_bundle = None
hint_bundle       = None
phase4_models     = None


def load_models():
    global trained_models, ensemble_models, distractor_bundle, hint_bundle, phase4_models

    try:
        with open(f"{MODELS_DIR}/trained_models.pkl", "rb") as f:
            trained_models = pickle.load(f)
        print("[OK] trained_models loaded")
        print(f"[DEBUG] trained_models keys: {list(trained_models.keys()) if isinstance(trained_models, dict) else type(trained_models)}")
    except Exception as e:
        print(f"[WARN] trained_models: {e}")

    try:
        with open(f"{MODELS_DIR}/ensemble_models.pkl", "rb") as f:
            ensemble_models = pickle.load(f)
        print("[OK] ensemble_models loaded")
    except Exception as e:
        print(f"[WARN] ensemble_models: {e}")

    try:
        with open(f"{MODELS_DIR}/distractor_model.pkl", "rb") as f:
            distractor_bundle = pickle.load(f)
        print(f"[OK] distractor_model loaded — type: {type(distractor_bundle)}")
    except Exception as e:
        print(f"[WARN] distractor_model: {e}")

    try:
        with open(f"{MODELS_DIR}/hint_model.pkl", "rb") as f:
            hint_bundle = pickle.load(f)
        print(f"[OK] hint_model loaded — type: {type(hint_bundle)}")
    except Exception as e:
        print(f"[WARN] hint_model: {e}")

    try:
        with open(f"{MODELS_DIR}/phase4_models.pkl", "rb") as f:
            phase4_models = pickle.load(f)
        print(f"[OK] phase4_models loaded — type: {type(phase4_models)}")
        print(f"[DEBUG] phase4_models keys: {list(phase4_models.keys()) if isinstance(phase4_models, dict) else type(phase4_models)}")
    except Exception as e:
        print(f"[WARN] phase4_models: {e}")


load_models()


class ArticleInput(BaseModel):
    article: str

class QuestionInput(BaseModel):
    article: str
    question: str
    options: List[str]
    correct_answer: Optional[str] = None

class AnswerCheck(BaseModel):
    article: str
    question: str
    selected_option: str
    options: List[str]
    correct_answer: Optional[str] = None


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

def extract_candidates(article, correct_answer, top_n=30):
    sentences = split_sentences(article)
    all_candidates = []
    
    for sentence in sentences:
        candidates = extract_candidates_from_sentence(sentence, correct_answer)
        all_candidates.extend(candidates)
    
    return list(set(all_candidates))[:top_n]


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


def extract_features(candidates, correct_answer, article):
    try:
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
        vectorizer.fit(candidates + [correct_answer])
        cand_vecs = vectorizer.transform(candidates)
        corr_vec  = vectorizer.transform([correct_answer])
        cos_sims  = cosine_similarity(cand_vecs, corr_vec).ravel()
    except Exception:
        cos_sims = np.zeros(len(candidates))
    char_matches = np.array([char_match(c, correct_answer) for c in candidates])
    freq_counts  = np.array([passage_freq(c, article) for c in candidates])
    return np.column_stack([cos_sims, char_matches, freq_counts])


def get_ranker():
    if distractor_bundle is None:
        return None
    if isinstance(distractor_bundle, dict):
        return distractor_bundle.get('ranker')
    return distractor_bundle


def get_hint_model():
    if hint_bundle is None:
        return None
    if isinstance(hint_bundle, dict):
        return hint_bundle.get('hint_model')
    return hint_bundle


def run_distractor_generation(article, correct_answer):
    ranker = get_ranker()
    if ranker is None:
        return []

    candidates = extract_candidates(article, correct_answer)
    if not candidates:
        return []

    features = extract_features(candidates, correct_answer, article)

    try:
        scores = ranker.predict_proba(features)[:, 1]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = [c for c, s in ranked if c.lower() != correct_answer.lower()][:3]
        print(f"[Distractor] candidates={len(candidates)} result={result}")
        return result
    except Exception as e:
        print(f"[Distractor ERROR] {e}")
        return candidates[:3]


def run_hint_generation(article, question):
    sentences = split_sentences(article)
    if not sentences:
        return []

    model = get_hint_model()
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
        if n >= 3:
            return [ranked[0][0], ranked[n // 2][0], ranked[-1][0]]
        elif n == 2:
            return [ranked[0][0], ranked[1][0], ranked[1][0]]
        elif n == 1:
            return [ranked[0][0], ranked[0][0], ranked[0][0]]
    except Exception as e:
        print(f"[Hint ERROR] {e}")
        return sentences[:3]

    return []


def run_answer_verification(article, question, chosen_option):
    if not ensemble_models:
        return 0.5

    meta_clf = ensemble_models.get('stacking_meta_clf') if isinstance(ensemble_models, dict) else None
    if not meta_clf:
        return 0.5

    combined = clean(f"{article} {question} {chosen_option}")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec  = TfidfVectorizer(max_features=1000)
        X    = vec.fit_transform([combined])
        prob = meta_clf.predict_proba(X)
        return float(prob[0][1]) if prob.shape[1] > 1 else 0.5
    except Exception as e:
        print(f"[Verify ERROR] {e}")
        return 0.5


@app.get("/")
async def root():
    return {"message": "Brain Bling API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "trained_models":   trained_models is not None,
            "ensemble_models":  ensemble_models is not None,
            "distractor_model": distractor_bundle is not None,
            "hint_model":       hint_bundle is not None,
        }
    }


@app.get("/metrics")
async def get_metrics():
    try:
        binary   = pd.read_csv(f"{MODELS_DIR}/binary_classification_metrics.csv")
        ensemble = pd.read_csv(f"{MODELS_DIR}/ensemble_binary_metrics.csv")
        return {
            "binary_metrics":   binary.to_dict(orient="records"),
            "ensemble_metrics": ensemble.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-article")
async def get_sample_article():
    try:
        df  = pd.read_csv(f"{DATA_DIR}/dev.csv")
        row = df.sample(1).iloc[0]
        return {
            "article":        str(row["article"]),
            "question":       str(row["question"]),
            "A":              {"text": str(row["A"])},
            "B":              {"text": str(row["B"])},
            "C":              {"text": str(row["C"])},
            "D":              {"text": str(row["D"])},
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

        distractors = run_distractor_generation(input.article, correct_text)
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
        
        # Varied question patterns to avoid repetition
        question_patterns = [
            "What is the main idea of the passage?",
            "What does the passage mainly discuss?",
            "According to the passage, what is the primary focus?",
            "What is the key message conveyed in the text?",
            "What is the central theme of the passage?",
            "What topic is primarily addressed in the text?",
            "What is the main subject of the passage?",
            "What concept does the passage primarily explain?"
        ]
        
        # Randomly select a question pattern
        import random
        question = random.choice(question_patterns)
        
        correct_text = sentences[0] if sentences else "The passage describes a key concept."

        print(f"\n[QUIZ] article={input.article[:80]} correct={correct_text}")

        distractors = run_distractor_generation(input.article, correct_text)
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
        hints = run_hint_generation(input.article, input.question)
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
        confidence = run_answer_verification(input.article, input.question, selected_text)

        return {
            "is_correct":           is_correct,
            "confidence":           round(confidence, 4),
            "correct_answer":       input.correct_answer,
            "correct_option_text":  correct_text,
            "selected_option_text": selected_text,
            "explanation":          f"Model A confidence: {confidence:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)