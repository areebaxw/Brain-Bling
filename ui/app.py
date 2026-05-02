import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="QuizMind AI", page_icon="Q", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
:root {
    --bg-0: #06030c;
    --bg-1: #0d0718;
    --bg-2: #1b1130;
    --panel: rgba(20, 13, 36, 0.92);
    --panel-soft: rgba(27, 17, 48, 0.86);
    --text: #f4ecff;
    --muted: #c9b6e9;
    --accent: #a855f7;
    --accent-2: #7c3aed;
    --accent-3: #d8b4fe;
    --glow: rgba(168, 85, 247, 0.58);
}
body { font-family: 'Poppins', sans-serif; color: var(--text); }
.stApp { background: radial-gradient(circle at 12% 10%, #2a1450 0%, var(--bg-1) 34%, var(--bg-0) 75%, #030106 100%); }
.main-header { background: linear-gradient(90deg, var(--accent-3), var(--accent), var(--accent-2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 2rem; text-shadow: 0 0 24px rgba(168, 85, 247, 0.35); }
.glow-card { background: var(--panel); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; padding: 2rem; box-shadow: 0 0 24px var(--glow); border: 1px solid rgba(216, 180, 254, 0.26); margin-bottom: 1rem; color: var(--text); }
.purple-glow { box-shadow: 0 0 22px var(--glow) !important; }
.pink-glow { box-shadow: 0 0 22px rgba(167, 98, 255, 0.52) !important; }
.mint-glow { box-shadow: 0 0 22px rgba(123, 58, 237, 0.5) !important; }
.blue-glow { box-shadow: 0 0 22px rgba(139, 92, 246, 0.52) !important; }
.gold-glow { box-shadow: 0 0 22px rgba(216, 180, 254, 0.5) !important; }
.option-card { background: var(--panel-soft); backdrop-filter: blur(10px); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border: 2px solid rgba(216, 180, 254, 0.24); transition: all 0.3s ease; cursor: pointer; color: var(--text); }
.option-card.correct { border-color: #d8b4fe; background: rgba(124, 58, 237, 0.33); box-shadow: 0 0 20px rgba(168, 85, 247, 0.58); color: var(--text); }
.option-card.incorrect { border-color: #a855f7; background: rgba(91, 33, 182, 0.36); box-shadow: 0 0 20px rgba(124, 58, 237, 0.55); color: var(--text); }
.metric-box { background: var(--panel-soft); backdrop-filter: blur(10px); border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 0 22px rgba(124, 58, 237, 0.45); border: 1px solid rgba(216, 180, 254, 0.24); color: var(--text); }
.result-banner { padding: 1.5rem; border-radius: 12px; text-align: center; font-weight: 600; margin: 1rem 0; }
.result-banner.correct { background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: var(--text); box-shadow: 0 0 22px rgba(168, 85, 247, 0.6); }
.result-banner.incorrect { background: linear-gradient(135deg, #6d28d9, #4c1d95); color: var(--text); box-shadow: 0 0 22px rgba(124, 58, 237, 0.56); }
.hint-card { background: var(--panel-soft); backdrop-filter: blur(10px); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid var(--accent); color: var(--text); }
.hint-card.hint-1 { border-left-color: #a855f7; box-shadow: 0 0 16px rgba(168, 85, 247, 0.52); }
.hint-card.hint-2 { border-left-color: #9333ea; box-shadow: 0 0 16px rgba(147, 51, 234, 0.5); }
.hint-card.hint-3 { border-left-color: #d8b4fe; box-shadow: 0 0 16px rgba(216, 180, 254, 0.52); }
.sidebar-gradient { background: linear-gradient(180deg, #150c27, #0b0614); padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; color: var(--text); border: 1px solid rgba(216, 180, 254, 0.2); }
.status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
.status-loaded { background-color: #d8b4fe; box-shadow: 0 0 10px rgba(216, 180, 254, 0.9); }
.status-not-loaded { background-color: #5b4a7d; box-shadow: 0 0 10px rgba(91, 74, 125, 0.75); }
.question-card { background: linear-gradient(135deg, #7c3aed, #4c1d95); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 0 16px rgba(124, 58, 237, 0.62); color: var(--text); }
.scroll-card { max-height: 400px; overflow-y: auto; padding-right: 1rem; color: var(--text); }
.data-table { background: var(--panel-soft); backdrop-filter: blur(10px); border-radius: 12px; overflow: hidden; box-shadow: 0 0 20px rgba(124, 58, 237, 0.5); color: var(--text); }
.divider-gradient { height: 2px; background: linear-gradient(90deg, transparent, #6d28d9, #a855f7, #d8b4fe, transparent); margin: 2rem 0; }
.tab-header { background: linear-gradient(135deg, #1a1130, #0f0a1d); padding: 1rem; border-radius: 12px; text-align: center; font-weight: 600; margin-bottom: 1rem; color: var(--text); border: 1px solid rgba(216, 180, 254, 0.2); }
div[data-testid="stTabs"] button { color: var(--muted) !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent-3) !important; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #7c3aed, #a855f7) !important; color: var(--text) !important; border: 1px solid rgba(216, 180, 254, 0.34) !important; border-radius: 12px !important; box-shadow: 0 0 14px rgba(124, 58, 237, 0.42); font-family: 'Poppins', sans-serif !important; }
div[data-testid="stButton"] > button:hover { border-color: rgba(216, 180, 254, 0.6) !important; box-shadow: 0 0 20px rgba(168, 85, 247, 0.58); transform: translateY(-1px); }
div[data-testid="stButton"] > button:disabled { background: linear-gradient(135deg, #39215f, #53308f) !important; color: #d9c8f0 !important; border-color: rgba(124, 58, 237, 0.32) !important; box-shadow: none; }
div[data-testid="stTextArea"] textarea { font-family: 'Poppins', sans-serif !important; background: rgba(29, 20, 49, 0.92) !important; color: var(--text) !important; border: 1px solid rgba(216, 180, 254, 0.3) !important; }
div[data-testid="stTextInput"] input { font-family: 'Poppins', sans-serif !important; background: rgba(29, 20, 49, 0.92) !important; color: var(--text) !important; border: 1px solid rgba(216, 180, 254, 0.3) !important; }
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background: rgba(29, 20, 49, 0.92) !important; color: var(--text) !important; border: 1px solid rgba(216, 180, 254, 0.3) !important; }
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
p, span, div { color: var(--text) !important; }
label { color: var(--muted) !important; }
@keyframes neonDrift {
    0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
    50% { transform: translate3d(15px, -10px, 0) scale(1.08); opacity: 0.7; }
    100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
}
@keyframes railScan {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
.aurora-wrap {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: -1;
    overflow: hidden;
}
.aurora-orb {
    position: absolute;
    border-radius: 999px;
    filter: blur(60px);
    animation: neonDrift 10s ease-in-out infinite;
}
.orb-a { width: 260px; height: 260px; top: 8%; left: 3%; background: rgba(124, 58, 237, 0.32); }
.orb-b { width: 320px; height: 320px; top: 40%; right: -40px; background: rgba(168, 85, 247, 0.24); animation-delay: 2s; }
.orb-c { width: 280px; height: 280px; bottom: -90px; left: 35%; background: rgba(216, 180, 254, 0.16); animation-delay: 4s; }
.command-shell {
    background: linear-gradient(140deg, rgba(31, 20, 56, 0.92), rgba(12, 7, 22, 0.94));
    border: 1px solid rgba(216, 180, 254, 0.28);
    border-radius: 24px;
    padding: 1.2rem 1.25rem;
    box-shadow: 0 0 28px rgba(124, 58, 237, 0.45);
    margin: 0.2rem 0 1.1rem 0;
}
.command-kicker {
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: #d8b4fe;
    margin: 0;
}
.command-sub {
    margin: 0.25rem 0 0.95rem 0;
    color: #c9b6e9;
    font-size: 0.95rem;
}
.pulse-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(160px, 1fr));
    gap: 0.65rem;
}
.pulse-card {
    background: rgba(28, 18, 49, 0.86);
    border: 1px solid rgba(216, 180, 254, 0.2);
    border-radius: 14px;
    padding: 0.7rem 0.85rem;
    box-shadow: inset 0 0 22px rgba(124, 58, 237, 0.16);
}
.pulse-label {
    margin: 0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #bfa9e2;
}
.pulse-value {
    margin: 0.3rem 0 0 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f4ecff;
}
.pulse-rail {
    height: 3px;
    margin-top: 0.55rem;
    border-radius: 99px;
    background: linear-gradient(90deg, #4c1d95, #a855f7, #d8b4fe, #4c1d95);
    background-size: 200% 100%;
    animation: railScan 2.2s linear infinite;
}
.timeline-wrap {
    background: rgba(20, 13, 36, 0.92);
    border: 1px solid rgba(216, 180, 254, 0.22);
    border-radius: 14px;
    padding: 1rem;
    margin-top: 0.7rem;
}
.timeline-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #d8b4fe;
    margin: 0 0 0.55rem 0;
}
.timeline-item {
    display: grid;
    grid-template-columns: 90px 1fr;
    gap: 0.65rem;
    margin-bottom: 0.55rem;
}
.timeline-item:last-child {
    margin-bottom: 0;
}
.timeline-chip {
    text-align: center;
    border-radius: 999px;
    padding: 0.2rem 0.4rem;
    background: rgba(124, 58, 237, 0.26);
    border: 1px solid rgba(216, 180, 254, 0.22);
    font-size: 0.72rem;
}
.timeline-bar {
    position: relative;
    background: rgba(91, 33, 182, 0.22);
    border-radius: 999px;
    height: 26px;
    border: 1px solid rgba(216, 180, 254, 0.2);
    overflow: hidden;
}
.timeline-fill {
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    background: linear-gradient(90deg, #7c3aed, #d8b4fe);
}
.timeline-text {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    color: #f4ecff;
}
.dock-card {
    background: linear-gradient(150deg, rgba(26, 17, 46, 0.92), rgba(11, 6, 20, 0.94));
    border: 1px solid rgba(216, 180, 254, 0.22);
    border-radius: 18px;
    box-shadow: 0 0 22px rgba(124, 58, 237, 0.34);
    padding: 0.9rem 1rem 1rem 1rem;
    margin-bottom: 1rem;
}
.dock-title {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    color: #d8b4fe;
    margin: 0.1rem 0 0.55rem 0;
}
.chart-shell {
    background: rgba(20, 13, 36, 0.88);
    border: 1px solid rgba(216, 180, 254, 0.2);
    border-radius: 14px;
    padding: 0.6rem 0.7rem 0.4rem 0.7rem;
    margin-top: 0.7rem;
}
@media (max-width: 900px) {
    .pulse-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
    .command-shell { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        with open('models/trained_models.pkl', 'rb') as f:
            trained_models = pickle.load(f)
    except:
        trained_models = {}
    try:
        with open('models/ensemble_models.pkl', 'rb') as f:
            ensemble_models = pickle.load(f)
    except:
        ensemble_models = {}
    try:
        with open('models/distractor_model.pkl', 'rb') as f:
            distractor_model = pickle.load(f)
    except:
        distractor_model = {}
    try:
        with open('models/hint_model.pkl', 'rb') as f:
            hint_model = pickle.load(f)
    except:
        hint_model = {}
    try:
        with open('data/processed/onehot_vectorizer.pkl', 'rb') as f:
            onehot_vectorizer = pickle.load(f)
    except:
        onehot_vectorizer = None
    return trained_models, ensemble_models, distractor_model, hint_model, onehot_vectorizer


@st.cache_data
def load_val_data():
    try:
        df = pd.read_csv('data/raw/dev.csv')
        return df
    except:
        return pd.DataFrame()


@st.cache_data
def load_metrics():
    try:
        binary_metrics = pd.read_csv('models/binary_classification_metrics.csv')
    except:
        binary_metrics = pd.DataFrame()
    try:
        ensemble_metrics = pd.read_csv('models/ensemble_binary_metrics.csv')
    except:
        ensemble_metrics = pd.DataFrame()
    try:
        distractor_results = pd.read_csv('models/distractor_results.csv')
    except:
        distractor_results = pd.DataFrame()
    try:
        hint_results = pd.read_csv('models/hint_results.csv')
    except:
        hint_results = pd.DataFrame()
    return binary_metrics, ensemble_metrics, distractor_results, hint_results


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()


def split_into_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def manual_stopwords():
    return set(['the','a','an','and','or','but','in','on','at','to','for','of','with',
                'by','is','are','was','were','be','been','have','has','had','do','does',
                'did','will','would','could','should','may','might','can','this','that',
                'these','those','i','you','he','she','it','we','they'])


def extract_candidates(article, correct_answer):
    stopwords = manual_stopwords()
    sentences = split_into_sentences(article)
    candidates = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            for length in range(1, 4):
                if i + length <= len(words):
                    candidate = ' '.join(words[i:i + length]).lower().strip()
                    if len(candidate) > 2 and candidate != correct_answer.lower():
                        candidates.append(candidate)
        word_freq = Counter(words)
        for word, count in word_freq.items():
            if count > 1 and len(word) > 2 and word.lower() != correct_answer.lower():
                candidates.append(word.lower())
    return list(set(candidates))


def character_match_score(candidate, correct_answer):
    c1 = set(candidate.lower())
    c2 = set(correct_answer.lower())
    intersection = len(c1 & c2)
    union = len(c1 | c2)
    return intersection / union if union > 0 else 0


def passage_frequency_count(candidate, article):
    words = article.lower().split()
    return sum(1 for word in words if candidate.lower() in word)


def cosine_sim_candidate(candidate, correct_answer, vectorizer):
    try:
        X = vectorizer.transform([candidate, correct_answer])
        return float(cosine_similarity(X[0:1], X[1:2])[0][0])
    except:
        return 0.0


def generate_distractors(article, correct_answer, distractor_model):
    if isinstance(distractor_model, dict):
        ranker = distractor_model.get('ranker')
        if ranker is None:
            return []
    elif hasattr(distractor_model, 'predict_proba'):
        ranker = distractor_model
    else:
        return []
    candidates = extract_candidates(article, correct_answer)
    if not candidates:
        return []
    features = []
    for c in candidates:
        try:
            _v = TfidfVectorizer()
            X = _v.fit_transform([c, correct_answer])
            cosim = float(cosine_similarity(X[0:1], X[1:2])[0][0])
        except Exception:
            cosim = 0.0
        features.append([
            cosim,
            character_match_score(c, correct_answer),
            passage_frequency_count(c, article)
        ])
    scores = ranker.predict_proba(features)[:, 1]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if c.lower() != correct_answer.lower()][:3]


def generate_hints(article, question, hint_bundle):
    sentences = split_into_sentences(article)
    if not sentences:
        return []
    if 'hint_model' not in hint_bundle:
        return sentences[:3]
    model = hint_bundle['hint_model']
    features = []
    total = len(sentences)
    stopwords = manual_stopwords()
    q_words = set(question.lower().split()) - stopwords
    for i, sent in enumerate(sentences):
        s_words = set(sent.lower().split()) - stopwords
        overlap = len(s_words & q_words) / len(q_words) if q_words else 0
        position = i / max(total - 1, 1)
        length = len(sent.split())
        features.append([overlap, position, length])
    scores = model.predict_proba(features)[:, 1]
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1])
    n = len(ranked)
    hints = []
    if n >= 3:
        hints = [ranked[0][0], ranked[n // 2][0], ranked[-1][0]]
    elif n == 2:
        hints = [ranked[0][0], ranked[1][0], ranked[1][0]]
    elif n == 1:
        hints = [ranked[0][0], ranked[0][0], ranked[0][0]]
    return hints


def verify_answer(article, question, chosen_option, ensemble_models, onehot_vectorizer=None):
    if onehot_vectorizer is not None:
        try:
            ctx = clean_text(f"{article} {question}")
            opt = clean_text(chosen_option)
            if ctx.strip() and opt.strip():
                vecs = onehot_vectorizer.transform([ctx, opt])
                sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
                return float(np.clip(sim * 5.0, 0.05, 0.95))
        except Exception:
            pass
    return 0.5


def init_session_state():
    defaults = {
        'article': '',
        'question': '',
        'opt_a': '',
        'opt_b': '',
        'opt_c': '',
        'opt_d': '',
        'correct_answer': 'A',
        'selected_option': None,
        'answer_checked': False,
        'confidence': 0.0,
        'hints': [],
        'hint_revealed': [False, False, False],
        'inference_times': {},
        'quiz_generated': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

trained_models, ensemble_models, distractor_model, hint_model, onehot_vectorizer = load_models()
val_df = load_val_data()
binary_metrics, ensemble_metrics, distractor_results, hint_results = load_metrics()


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-gradient">
        <h2 style="color: #E9D5FF; text-align: center;">QuizMind AI</h2>
        <p style="text-align: center; color: #C9B6E9;">Intelligent Reading Comprehension and Quiz Generation</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-gradient">
        <h4 style="color: #D8B4FE;">Instructions</h4>
        <ol style="color: #C9B6E9; font-size: 0.9rem;">
            <li>Load a RACE sample or paste your own</li>
            <li>Click Generate Distractors</li>
            <li>Click Generate Quiz</li>
            <li>Use Mission Quiz to answer</li>
            <li>Use Hint Reactor if stuck</li>
            <li>Track metrics in Telemetry</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-gradient"><h4 style="color: #D8B4FE;">Model Status</h4>', unsafe_allow_html=True)
    for name, obj in [("Trained Models", trained_models), ("Ensemble Models", ensemble_models),
                       ("Distractor Model", distractor_model), ("Hint Model", hint_model),
                       ("Feature Vectorizer", onehot_vectorizer)]:
        cls = "status-loaded" if obj else "status-not-loaded"
        st.markdown(f'<p style="color:#C9B6E9;font-size:0.9rem;"><span class="status-indicator {cls}"></span>{name}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# TITLE
models_online = sum(1 for obj in [trained_models, ensemble_models, distractor_model, hint_model] if obj)
sample_pool = len(val_df) if not val_df.empty else 0
hint_progress = f"{sum(st.session_state.hint_revealed)}/3"
confidence_text = f"{st.session_state.confidence:.1%}" if st.session_state.answer_checked else "--"
if st.session_state.inference_times:
    avg_latency = float(np.mean(list(st.session_state.inference_times.values())))
    latency_text = f"{avg_latency:.3f}s"
else:
    latency_text = "--"

if not st.session_state.quiz_generated:
    quiz_mode = "Input Mode"
elif st.session_state.answer_checked and st.session_state.selected_option == st.session_state.correct_answer:
    quiz_mode = "Solved"
elif st.session_state.answer_checked:
    quiz_mode = "Reviewing"
else:
    quiz_mode = "Quiz Live"

st.markdown("""
<div class="aurora-wrap">
    <div class="aurora-orb orb-a"></div>
    <div class="aurora-orb orb-b"></div>
    <div class="aurora-orb orb-c"></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="command-shell">
    <p class="command-kicker">Neural Command Deck</p>
    <p class="command-sub">A cinematic learning cockpit with real-time quiz telemetry.</p>
    <div class="pulse-grid">
        <div class="pulse-card">
            <p class="pulse-label">Mode</p>
            <p class="pulse-value">{quiz_mode}</p>
            <div class="pulse-rail"></div>
        </div>
        <div class="pulse-card">
            <p class="pulse-label">Models Online</p>
            <p class="pulse-value">{models_online}/4</p>
            <div class="pulse-rail"></div>
        </div>
        <div class="pulse-card">
            <p class="pulse-label">Sample Pool</p>
            <p class="pulse-value">{sample_pool}</p>
            <div class="pulse-rail"></div>
        </div>
        <div class="pulse-card">
            <p class="pulse-label">Confidence</p>
            <p class="pulse-value">{confidence_text}</p>
            <div class="pulse-rail"></div>
        </div>
        <div class="pulse-card">
            <p class="pulse-label">Avg Inference</p>
            <p class="pulse-value">{latency_text}</p>
            <div class="pulse-rail"></div>
        </div>
        <div class="pulse-card">
            <p class="pulse-label">Hints Opened</p>
            <p class="pulse-value">{hint_progress}</p>
            <div class="pulse-rail"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">QuizMind AI</h1>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Input Lab", "Mission Quiz", "Hint Reactor", "Telemetry"])


# INPUT LAB
with tab1:
    # Apply pending distractor values BEFORE any widgets render
    for _s in ['a', 'b', 'c', 'd']:
        _pk = f'_pending_opt_{_s}'
        if _pk in st.session_state:
            st.session_state[f'ti_{_s}'] = st.session_state[_pk]
            st.session_state[f'opt_{_s}'] = st.session_state[_pk]
            del st.session_state[_pk]

    st.markdown('<div class="tab-header">Input Lab</div>', unsafe_allow_html=True)

    if st.button("Load Random RACE Sample", key="load_sample"):
        if not val_df.empty:
            sample = val_df.sample(1).iloc[0]
            article_text = str(sample['article'])
            question_text = str(sample['question'])
            opt_a_text = str(sample['A'])
            opt_b_text = str(sample['B'])
            opt_c_text = str(sample['C'])
            opt_d_text = str(sample['D'])
            correct_answer = str(sample['answer']) if str(sample['answer']) in ["A", "B", "C", "D"] else "A"

            st.session_state.article = article_text
            st.session_state.question = question_text
            st.session_state.opt_a = opt_a_text
            st.session_state.opt_b = opt_b_text
            st.session_state.opt_c = opt_c_text
            st.session_state.opt_d = opt_d_text
            st.session_state.correct_answer = correct_answer

            st.session_state.ta_article = article_text
            st.session_state.ta_question = question_text
            st.session_state.ti_a = opt_a_text
            st.session_state.ti_b = opt_b_text
            st.session_state.ti_c = opt_c_text
            st.session_state.ti_d = opt_d_text
            st.session_state.sb_correct = correct_answer

            st.session_state.selected_option = None
            st.session_state.answer_checked = False
            st.session_state.hint_revealed = [False, False, False]
            st.session_state.quiz_generated = False
            st.success("RACE sample loaded.")
        else:
            st.error("dev.csv not found or empty.")

    st.markdown('<div class="glow-card purple-glow">', unsafe_allow_html=True)
    st.session_state.article = st.text_area(
        "Paste your article here:", value=st.session_state.article, height=220, key="ta_article"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-card purple-glow">', unsafe_allow_html=True)
    st.session_state.question = st.text_area(
        "Enter your question:", value=st.session_state.question, height=120, key="ta_question"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        st.markdown('<div class="glow-card purple-glow">', unsafe_allow_html=True)
        st.session_state.opt_a = st.text_input("Option A:", value=st.session_state.opt_a, key="ti_a")
        st.session_state.opt_b = st.text_input("Option B:", value=st.session_state.opt_b, key="ti_b")
        st.markdown('</div>', unsafe_allow_html=True)
    with opt_col2:
        st.markdown('<div class="glow-card purple-glow">', unsafe_allow_html=True)
        st.session_state.opt_c = st.text_input("Option C:", value=st.session_state.opt_c, key="ti_c")
        st.session_state.opt_d = st.text_input("Option D:", value=st.session_state.opt_d, key="ti_d")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-card purple-glow">', unsafe_allow_html=True)
    idx = ["A", "B", "C", "D"].index(st.session_state.correct_answer) if st.session_state.correct_answer in ["A", "B", "C", "D"] else 0
    st.session_state.correct_answer = st.selectbox("Correct Answer:", ["A", "B", "C", "D"], index=idx, key="sb_correct")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Generate Distractors", key="gen_distractors"):
        correct_slot = st.session_state.correct_answer
        correct_text_map = {
            "A": st.session_state.opt_a,
            "B": st.session_state.opt_b,
            "C": st.session_state.opt_c,
            "D": st.session_state.opt_d,
        }
        correct_text = correct_text_map.get(correct_slot, "")
        if not st.session_state.article.strip():
            st.error("Please paste an article first.")
        elif not correct_text.strip():
            st.error(f"Please enter the correct answer text in Option {correct_slot} first.")
        else:
            with st.spinner("Generating distractors..."):
                start = time.time()
                distractors = generate_distractors(st.session_state.article, correct_text, distractor_model)
                st.session_state.inference_times['distractor_generation'] = time.time() - start
            if distractors:
                other_slots = [s for s in ["A", "B", "C", "D"] if s != correct_slot]
                for slot, dist in zip(other_slots, distractors):
                    st.session_state[f'_pending_opt_{slot.lower()}'] = dist
                st.success(f"Generated {len(distractors)} distractor(s). Review them in the option fields below, then click Generate Quiz.")
                st.rerun()
            else:
                st.warning("Could not generate distractors. Please check the distractor model is loaded.")

    if st.button("Generate Quiz", key="generate_quiz"):
        opts = {
            "A": st.session_state.opt_a,
            "B": st.session_state.opt_b,
            "C": st.session_state.opt_c,
            "D": st.session_state.opt_d
        }
        if not st.session_state.article.strip() or not st.session_state.question.strip():
            st.error("Please fill in article and question.")
        elif not all(v.strip() for v in opts.values()):
            st.error("Please fill in all four options.")
        else:
            with st.spinner("Generating your quiz..."):
                start = time.time()
                st.session_state.options = opts
                st.session_state.hints = generate_hints(
                    st.session_state.article, st.session_state.question, hint_model
                )
                st.session_state.hint_revealed = [False, False, False]
                st.session_state.selected_option = None
                st.session_state.answer_checked = False
                st.session_state.quiz_generated = True
                st.session_state.inference_times['quiz_generation'] = time.time() - start
            st.success("Quiz generated. Continue in Mission Quiz.")


# MISSION QUIZ
with tab2:
    st.markdown('<div class="tab-header">Mission Quiz</div>', unsafe_allow_html=True)

    if not st.session_state.quiz_generated:
        st.warning("Generate a quiz first from Input Lab.")
    else:
        st.markdown('<div class="glow-card pink-glow">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#D8B4FE;">Article</h4>', unsafe_allow_html=True)
        st.markdown(f'<div class="scroll-card">{st.session_state.article}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="question-card"><h4>Question: {st.session_state.question}</h4></div>', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#D8B4FE; margin:1rem 0;">Choose your answer:</h4>', unsafe_allow_html=True)

        opts = st.session_state.get('options', {
            "A": st.session_state.opt_a,
            "B": st.session_state.opt_b,
            "C": st.session_state.opt_c,
            "D": st.session_state.opt_d
        })

        for key, text in opts.items():
            is_correct = key == st.session_state.correct_answer
            is_selected = st.session_state.selected_option == key
            is_checked = st.session_state.answer_checked

            button_label = f"{key}: {text}"
            if is_checked and is_correct:
                button_label = f"Correct {key}: {text}"
            elif is_checked and is_selected:
                button_label = f"Wrong {key}: {text}"

            if st.button(button_label, key=f"opt_btn_{key}", disabled=is_checked):
                st.session_state.selected_option = key
                st.session_state.answer_checked = False
                st.rerun()

        quiz_col1, quiz_col2 = st.columns([2, 1])
        with quiz_col1:
            check_disabled = st.session_state.selected_option is None
            if st.button("Check Answer", key="check_answer", disabled=check_disabled):
                with st.spinner("Checking your answer..."):
                    start = time.time()
                    chosen_text = opts[st.session_state.selected_option]
                    st.session_state.confidence = verify_answer(
                        st.session_state.article,
                        st.session_state.question,
                        chosen_text,
                        ensemble_models,
                        onehot_vectorizer
                    )
                    st.session_state.answer_checked = True
                    st.session_state.inference_times['answer_check'] = time.time() - start
                    st.rerun()

        with quiz_col2:
            if st.session_state.answer_checked:
                is_correct = st.session_state.selected_option == st.session_state.correct_answer
                banner_class = "correct" if is_correct else "incorrect"
                banner_text = "Correct. Well done." if is_correct else f"Incorrect. Correct answer: {st.session_state.correct_answer}"
                st.markdown(f'<div class="result-banner {banner_class}">{banner_text}</div>', unsafe_allow_html=True)
                st.markdown(f'''
                <div class="metric-box blue-glow">
                    <h5 style="color:#E9D5FF;">Model Confidence</h5>
                    <h3 style="color:#E9D5FF;">{st.session_state.confidence:.1%}</h3>
                </div>''', unsafe_allow_html=True)
                if 'answer_check' in st.session_state.inference_times:
                    st.markdown(f'''
                    <div class="metric-box blue-glow">
                        <h5 style="color:#E9D5FF;">Inference Time</h5>
                        <h3 style="color:#E9D5FF;">{st.session_state.inference_times["answer_check"]:.3f}s</h3>
                    </div>''', unsafe_allow_html=True)


# HINT REACTOR
with tab3:
    st.markdown('<div class="tab-header">Hint Reactor</div>', unsafe_allow_html=True)

    if not st.session_state.quiz_generated:
        st.warning("Generate a quiz first from Input Lab.")
    else:
        st.markdown('<h2 style="color:#D8B4FE; text-align:center;">Need a Hint?</h2>', unsafe_allow_html=True)
        hints = st.session_state.hints
        if hints:
            labels = ["Hint 1 - General Clue", "Hint 2 - More Specific", "Hint 3 - Near Explicit"]
            css_class = ["hint-1", "hint-2", "hint-3"]

            for i in range(3):
                if i < len(hints):
                    if st.session_state.hint_revealed[i]:
                        st.markdown(f'''
                        <div class="hint-card {css_class[i]}">
                            <h5>{labels[i]}</h5>
                            <p>{hints[i]}</p>
                        </div>''', unsafe_allow_html=True)
                    else:
                        if st.button(f"Show Hint {i + 1}", key=f"hint_btn_{i}"):
                            st.session_state.hint_revealed[i] = True
                            st.rerun()

            if all(st.session_state.hint_revealed[:len(hints)]):
                if st.button("Reveal Answer", key="reveal_answer"):
                    ans_key = st.session_state.correct_answer
                    ans_opts = st.session_state.get('options', {
                        "A": st.session_state.opt_a,
                        "B": st.session_state.opt_b,
                        "C": st.session_state.opt_c,
                        "D": st.session_state.opt_d
                    })
                    ans_text = ans_opts.get(ans_key, '')
                    st.markdown(f'''
                    <div class="result-banner correct gold-glow">
                        <h3>The correct answer is: {ans_key}</h3>
                        <p>{ans_text}</p>
                    </div>''', unsafe_allow_html=True)
        else:
            st.warning("No hints available.")


# TELEMETRY
with tab4:
    st.markdown('<div class="tab-header">Telemetry Dashboard</div>', unsafe_allow_html=True)

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.markdown('<h4 style="color:#C4B5FD;">Model A Performance</h4>', unsafe_allow_html=True)
        if not binary_metrics.empty:
            st.dataframe(binary_metrics, use_container_width=True)
        else:
            st.warning("Metrics not available.")

    with metrics_col2:
        st.markdown('<h4 style="color:#C4B5FD;">Ensemble Performance</h4>', unsafe_allow_html=True)
        if not ensemble_metrics.empty:
            st.dataframe(ensemble_metrics, use_container_width=True)
        else:
            st.warning("Metrics not available.")

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)

    data_col1, data_col2 = st.columns(2)
    with data_col1:
        st.markdown('<h4 style="color:#C4B5FD;">Distractor Results</h4>', unsafe_allow_html=True)
        if not distractor_results.empty:
            st.dataframe(distractor_results, use_container_width=True)
        else:
            st.warning("Not available.")
    with data_col2:
        st.markdown('<h4 style="color:#C4B5FD;">Hint Results</h4>', unsafe_allow_html=True)
        if not hint_results.empty:
            st.dataframe(hint_results, use_container_width=True)
        else:
            st.warning("Not available.")

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#C4B5FD;">Performance KPIs</h4>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    best_acc = binary_metrics['Accuracy'].max() if not binary_metrics.empty and 'Accuracy' in binary_metrics.columns else 0
    best_f1 = binary_metrics['Macro F1'].max() if not binary_metrics.empty and 'Macro F1' in binary_metrics.columns else 0
    total_inf = sum(st.session_state.inference_times.values())

    with kpi1:
        st.markdown(f'<div class="metric-box blue-glow"><h5 style="color:#E9D5FF;">Best Accuracy</h5><h3 style="color:#E9D5FF;">{best_acc:.3f}</h3></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="metric-box blue-glow"><h5 style="color:#E9D5FF;">Best F1</h5><h3 style="color:#E9D5FF;">{best_f1:.3f}</h3></div>', unsafe_allow_html=True)
    with kpi3:
        distractor_f1 = distractor_results['F1'].iloc[0] if not distractor_results.empty and 'F1' in distractor_results.columns else 0
        st.markdown(f'<div class="metric-box blue-glow"><h5 style="color:#E9D5FF;">Distractor F1</h5><h3 style="color:#E9D5FF;">{distractor_f1:.3f}</h3></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="metric-box blue-glow"><h5 style="color:#E9D5FF;">Total Inference</h5><h3 style="color:#E9D5FF;">{total_inf:.3f}s</h3></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#C4B5FD;">Inference Latency Tracker</h4>', unsafe_allow_html=True)

    if st.session_state.inference_times:
        max_latency = max(st.session_state.inference_times.values())
        pulse_rows = []
        for op, latency in st.session_state.inference_times.items():
            op_label = op.replace("_", " ").title()
            width_pct = int((latency / max_latency) * 100) if max_latency else 0
            width_pct = min(max(width_pct, 8), 100)
            pulse_rows.append(f'''
            <div class="timeline-item">
                <div class="timeline-chip">{op_label}</div>
                <div class="timeline-bar">
                    <div class="timeline-fill" style="width:{width_pct}%"></div>
                    <div class="timeline-text">{latency:.3f}s</div>
                </div>
            </div>''')

        st.markdown(f'''
        <div class="timeline-wrap">
            <p class="timeline-title">Inference Pulse Map</p>
            {"".join(pulse_rows)}
        </div>''', unsafe_allow_html=True)

        lat_df = pd.DataFrame([
            {"Operation": k.replace("_", " ").title(), "Time (s)": f"{v:.3f}"}
            for k, v in st.session_state.inference_times.items()
        ])
        st.dataframe(lat_df, use_container_width=True)
    else:
        st.info("No inference operations yet.")

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    if st.button("Export Session Log", key="export_session"):
        opts = st.session_state.get('options', {})
        session_df = pd.DataFrame([{
            "article": st.session_state.article[:200],
            "question": st.session_state.question,
            "correct_answer": st.session_state.correct_answer,
            "selected_option": st.session_state.selected_option,
            "confidence": st.session_state.confidence,
            "hints_revealed": sum(st.session_state.hint_revealed)
        }])
        st.download_button(
            label="Download Session Log",
            data=session_df.to_csv(index=False),
            file_name="quizmind_session_log.csv",
            mime="text/csv"
        )
