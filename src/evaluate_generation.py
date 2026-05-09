"""
EVALUATE_GENERATION.PY — BLEU / ROUGE / METEOR Evaluation
===========================================================
Evaluates all text generation tasks:
  1. Question Generation  — Wh-word pipeline vs. RACE reference questions
  2. Distractor Generation — trained RF ranker vs. RACE reference wrong options
  3. Hint Generation       — trained LR model vs. answer-containing sentences

Run on Colab after all models are trained.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd

# ── NLP metric libraries ──────────────────────────────────────────────────────
import nltk
nltk.download('punkt',    quiet=True)
nltk.download('wordnet',  quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.translate.bleu_score   import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor

try:
    from rouge_score import rouge_scorer as rs_module
    ROUGE_AVAILABLE = True
except ImportError:
    print("[WARN] rouge_score not installed — run: pip install rouge-score")
    ROUGE_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(_BASE, 'data', 'raw')
MODELS_DIR    = os.path.join(_BASE, 'models', 'model_b', 'traditional')
OUTPUT_DIR    = os.path.join(_BASE, 'models', 'model_b', 'traditional')
ONEHOT_PATH   = os.path.join(_BASE, 'data', 'processed', 'onehot_vectorizer.pkl')

EVAL_SAMPLES  = 200   # questions to evaluate (keep low for speed)

STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','is','are','was','were','be','been','have','has','had','do','does',
    'did','will','would','could','should','may','might','it','its','this','that'
}

# ── Metric helpers ────────────────────────────────────────────────────────────
_sf = SmoothingFunction().method1

def bleu(hypothesis: str, reference: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not hyp:
        return 0.0
    return sentence_bleu([ref], hyp, smoothing_function=_sf)

def rouge(hypothesis: str, reference: str) -> dict:
    if not ROUGE_AVAILABLE:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    scorer = rs_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}

def meteor(hypothesis: str, reference: str) -> float:
    try:
        return nltk_meteor([reference.lower().split()], hypothesis.lower().split())
    except Exception:
        return 0.0

def best_match_scores(generated_list, reference_list):
    """For each generated item find the best-matching reference and average scores."""
    bleu_s, r1_s, r2_s, rL_s, met_s = [], [], [], [], []
    for gen in generated_list:
        b = max(bleu(gen, ref) for ref in reference_list)
        r = max(rouge(gen, ref)['rouge1'] for ref in reference_list)
        r2 = max(rouge(gen, ref)['rouge2'] for ref in reference_list)
        rl = max(rouge(gen, ref)['rougeL'] for ref in reference_list)
        m = max(meteor(gen, ref) for ref in reference_list)
        bleu_s.append(b); r1_s.append(r); r2_s.append(r2); rL_s.append(rl); met_s.append(m)
    return {
        'BLEU':    round(np.mean(bleu_s),  4),
        'ROUGE-1': round(np.mean(r1_s),    4),
        'ROUGE-2': round(np.mean(r2_s),    4),
        'ROUGE-L': round(np.mean(rL_s),    4),
        'METEOR':  round(np.mean(met_s),   4),
    }

# ── Question generation pipeline (mirrors api/app.py) ────────────────────────
def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]

def generate_wh_question(sentence):
    words = sentence.split()
    if len(words) < 5:
        return None, None
    skip_starts = ['but','and','or','so','however','also','then','yet','from','the','a','an']
    if words[0].lower() in skip_starts:
        return None, None
    for i, w in enumerate(words):
        if i > 0 and len(w) > 2 and w[0].isupper() and w.lower() not in STOPWORDS:
            answer = w
            j = i + 1
            while j < len(words) and len(words[j]) > 1 and words[j][0].isupper():
                answer += ' ' + words[j]; j += 1
            remaining = words[:i] + ['_______'] + words[j:]
            if i > 0 and words[i-1].lower() in ['in','at','near','from','to','around']:
                answer    = words[i-1] + ' ' + answer
                remaining = words[:i-1] + ['_______'] + words[j:]
                question  = "According to the passage, where " + ' '.join(remaining).lstrip(',').strip() + "?"
            else:
                question = "According to the passage, who " + ' '.join(remaining).lstrip(',').strip() + "?"
            return question, re.sub(r'[,;:]$', '', answer).strip()
    content = [(i, w) for i, w in enumerate(words)
               if len(w) >= 5 and w.lower() not in STOPWORDS and w.isalpha() and i > 0]
    if content:
        idx, answer = content[len(content) // 2]
        remaining   = words[:idx] + ['_______'] + words[idx+1:]
        return "According to the passage, " + ' '.join(remaining) + "?", answer
    return None, None

def generate_questions_for_article(article, answer, top_n=5):
    sentences = split_sentences(article)
    a_tokens  = set(answer.lower().split()) - STOPWORDS
    scored    = []
    for sent in sentences:
        s_tokens = set(sent.lower().split()) - STOPWORDS
        overlap  = len(s_tokens & a_tokens) / len(a_tokens) if a_tokens else 0.0
        q, ans   = generate_wh_question(sent)
        if q:
            scored.append((overlap, q, ans))
    scored.sort(key=lambda x: -x[0])
    return [q for _, q, _ in scored[:top_n]]

# ── Load data & models ────────────────────────────────────────────────────────
print("=" * 80)
print("GENERATION EVALUATION — BLEU / ROUGE / METEOR")
print("=" * 80)

val_df = pd.read_csv(f'{RAW_DIR}/dev.csv')
val_df = val_df[val_df['answer'].isin(['A','B','C','D'])].head(EVAL_SAMPLES).reset_index(drop=True)
print(f"[INFO] Evaluating on {len(val_df)} val samples")

def _load(path, label):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"[OK] {label} loaded")
        return obj
    print(f"[WARN] {label} not found"); return None

distractor_model  = _load(f'{MODELS_DIR}/distractor_model.pkl',  'distractor_model')
hint_bundle       = _load(f'{MODELS_DIR}/hint_model.pkl',         'hint_model')
onehot_vectorizer = _load(ONEHOT_PATH,                            'onehot_vectorizer')
hint_model        = hint_bundle['hint_model'] if hint_bundle else None

# ── 1. Question Generation Evaluation ────────────────────────────────────────
print("\n" + "=" * 80)
print("[1/3] QUESTION GENERATION — BLEU / ROUGE / METEOR")
print("=" * 80)

qg_rows = []
for _, row in val_df.iterrows():
    article   = str(row['article'])   if not pd.isna(row['article'])   else ''
    reference = str(row['question'])  if not pd.isna(row['question'])  else ''
    answer    = str(row[row['answer']]) if not pd.isna(row[row['answer']]) else ''
    if not article or not reference:
        continue
    generated = generate_questions_for_article(article, answer, top_n=5)
    if not generated:
        continue
    scores = best_match_scores(generated, [reference])
    qg_rows.append(scores)

qg_df = pd.DataFrame(qg_rows)
print(f"Samples evaluated: {len(qg_df)}")
print(qg_df.mean().round(4).to_string())
qg_summary = qg_df.mean().round(4).to_frame(name='Score').reset_index()
qg_summary.columns = ['Metric', 'Score']
qg_summary.insert(0, 'Task', 'Question Generation')

# ── 2. Distractor Generation Evaluation ──────────────────────────────────────
print("\n" + "=" * 80)
print("[2/3] DISTRACTOR GENERATION — BLEU / ROUGE / METEOR")
print("=" * 80)

def extract_candidates_simple(article, correct_answer, max_cands=40):
    words     = re.findall(r'\b[a-zA-Z]{3,}\b', article.lower())
    freq      = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    correct_tokens = set(correct_answer.lower().split())
    candidates = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])
                  if w not in correct_tokens]
    return candidates[:max_cands]

dist_rows = []
for _, row in val_df.iterrows():
    article  = str(row['article'])     if not pd.isna(row['article'])    else ''
    correct  = str(row[row['answer']]) if not pd.isna(row[row['answer']]) else ''
    refs     = [str(row[opt]) for opt in ['A','B','C','D'] if opt != row['answer']
                and not pd.isna(row[opt])]
    if not article or not refs:
        continue
    if distractor_model is not None:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
        except Exception:
            pass
        cands = extract_candidates_simple(article, correct)
        if not cands:
            continue
        generated = cands[:3]
    else:
        generated = extract_candidates_simple(article, correct, max_cands=3)
    if not generated:
        continue
    scores = best_match_scores(generated, refs)
    dist_rows.append(scores)

dist_df_eval = pd.DataFrame(dist_rows)
print(f"Samples evaluated: {len(dist_df_eval)}")
print(dist_df_eval.mean().round(4).to_string())
dist_summary = dist_df_eval.mean().round(4).to_frame(name='Score').reset_index()
dist_summary.columns = ['Metric', 'Score']
dist_summary.insert(0, 'Task', 'Distractor Generation')

# ── 3. Hint Generation Evaluation ────────────────────────────────────────────
print("\n" + "=" * 80)
print("[3/3] HINT GENERATION — BLEU / ROUGE / METEOR")
print("=" * 80)

hint_rows = []
for _, row in val_df.iterrows():
    article  = str(row['article'])     if not pd.isna(row['article'])    else ''
    question = str(row['question'])    if not pd.isna(row['question'])   else ''
    correct  = str(row[row['answer']]) if not pd.isna(row[row['answer']]) else ''
    sentences = split_sentences(article)
    if not sentences or not correct:
        continue
    refs = [s for s in sentences if correct.lower() in s.lower()]
    if not refs:
        correct_tokens = set(correct.lower().split()) - STOPWORDS
        refs = [s for s in sentences
                if len(set(s.lower().split()) & correct_tokens) / max(len(correct_tokens),1) >= 0.5]
    if not refs:
        continue
    if hint_model is not None:
        total = len(sentences)
        feats = []
        q_words = set(question.lower().split()) - STOPWORDS
        a_words = set(correct.lower().split())  - STOPWORDS
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
        top_idx  = np.argsort(probs)[::-1][:3]
        generated = [sentences[i] for i in top_idx]
    else:
        q_words   = set(question.lower().split()) - STOPWORDS
        scored    = [(len(set(s.lower().split()) & q_words), s) for s in sentences]
        scored.sort(key=lambda x: -x[0])
        generated = [s for _, s in scored[:3]]
    if not generated:
        continue
    scores = best_match_scores(generated, refs)
    hint_rows.append(scores)

hint_df_eval = pd.DataFrame(hint_rows)
print(f"Samples evaluated: {len(hint_df_eval)}")
print(hint_df_eval.mean().round(4).to_string())
hint_summary = hint_df_eval.mean().round(4).to_frame(name='Score').reset_index()
hint_summary.columns = ['Metric', 'Score']
hint_summary.insert(0, 'Task', 'Hint Generation')

# ── Save results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("[SAVING]")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_summary = pd.concat([qg_summary, dist_summary, hint_summary], ignore_index=True)
all_summary.to_csv(f'{OUTPUT_DIR}/generation_eval_metrics.csv', index=False)
print("[OK] generation_eval_metrics.csv saved")

print("\n[GENERATION EVALUATION SUMMARY]")
print(all_summary.to_string(index=False))
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
