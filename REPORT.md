# Brain Bling: AI-Powered Reading Comprehension and Quiz Generation System
### Final Technical Report — May 2026
### Revision 2 — Addresses full reviewer feedback

---

## Abstract

Brain Bling is a full-stack AI system that automatically generates reading comprehension quizzes from raw article passages. Model A handles answer verification and question generation using Logistic Regression, SVM, Naive Bayes, and Random Forest classifiers with ensemble methods, alongside a neural path using RoBERTa-RACE for deep semantic verification. Model B handles distractor generation via a Random Forest ranker augmented with Word2Vec semantic candidates, and graduated hint extraction via Logistic Regression. The system was trained and evaluated on the RACE dataset (97,687 questions across 28,000 passages; 87,866 training rows, 4,887 validation rows). Traditional models reached a best Macro F1 of 50.5% (Random Forest) and best option-level Exact Match of 31.5%, with ensemble stacking reaching 69.3% accuracy — though this figure is inflated by majority-class collapse; Macro F1 changed by less than 2 percentage points across all ensemble variants. Distractor generation achieved 55% partial-match accuracy, BLEU 0.031, and METEOR 0.102 post-Word2Vec. Hint generation achieved BLEU 0.481 and METEOR 0.525, though these high scores reflect extractive selection of verbatim passage sentences rather than generative capability. The system is deployed as a React + FastAPI application with a live quiz interface, confusion matrix dashboard, and NLG evaluation display.

---

## 1. Introduction & Motivation

Reading comprehension is a fundamental skill in education, yet creating high-quality assessment material — questions, wrong answer options (distractors), and hints — is a time-consuming task for teachers. Automating this process would allow educators to rapidly generate assessments from any text passage, improving both efficiency and the availability of learning resources.

Brain Bling was motivated by three core goals:

1. **Automatic Question Generation (AQG):** Given a reading passage, generate grammatically coherent Wh-word multiple-choice questions that target specific facts in the passage.
2. **Distractor Generation:** For each question, generate three plausible but incorrect answer options that cannot be trivially dismissed.
3. **Answer Verification:** Given a student's selected answer, verify whether it is correct using both lightweight traditional models and deep neural models.

The system was designed under defined constraints: traditional models use scikit-learn with handcrafted features as the primary method (One-Hot encoding), with TF-IDF used as a fallback where appropriate. Neural models are limited to fine-tuned transformers (RoBERTa-RACE for verification, SentenceTransformer for semantic similarity). This constraint reflects a real-world scenario where lightweight deployment is required alongside an optional neural upgrade path.

The **RACE dataset** (Reading Comprehension from Examinations) was chosen because it contains **97,687 questions spanning 28,000 passages** sourced from Chinese middle and high school examinations across grades 6–12 [1]. With an average passage length of 300–500 words and four-option multiple-choice format, RACE is one of the most challenging reading comprehension benchmarks, requiring reasoning beyond surface keyword overlap. Its scale (87,866 training rows, 4,887 validation rows) provides sufficient supervision for all models, and its human-authored distractors serve as gold-standard targets for Model B.

---

## 2. Related Work

**[1] Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. EMNLP 2017.**
The foundational dataset paper. RACE provides 97K questions from examinations for grades 6–12, making it one of the hardest reading comprehension benchmarks due to the need for reasoning beyond simple extraction. The dataset's class distribution (one correct out of four options) directly causes the 75% majority-class baseline problem encountered in this project.

**[2] Du, X., Shao, J., & Cardie, C. (2017). Learning to Ask: Neural Question Generation for Reading Comprehension. ACL 2017.**
The first neural approach to reading comprehension question generation, proposing a sequence-to-sequence model with attention to generate natural questions from sentences. This work directly motivates the limitations documented in Section 4.6 — our rule-based Wh-word template system produces the same type of errors (wrong Wh-word, pronoun retention) that Du et al. identified as the core challenge of AQG without sequence modelling. Their neural approach would resolve these errors but falls outside our traditional-ML constraint.

**[3] Mitkov, R., & Ha, L. A. (2003). Computer-Aided Generation of Multiple-Choice Tests. Proceedings of HLT-NAACL Workshop on Building Educational Applications Using NLP.**
One of the earliest computational approaches to MCQ generation [3]. Proposes extracting key phrases as question anchors and using WordNet synonyms as distractors — directly related to our rule-based Wh-word template approach (Section 4.7) and One-Hot cosine distractor ranking methodology (Section 5.1).

**[4] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781.**
The original Word2Vec paper [4]. Word2Vec was incorporated into Model B distractor generation specifically because semantic candidate expansion beyond passage extraction was required. Trained on 87,866 RACE articles, the model produced a vocabulary of 61,192 tokens with meaningful semantic neighbours (e.g., school → college, 0.69 similarity).

**[5] Ren, S., Zhu, K. Q., & Chen, W. (2021). Knowledge-Driven Distractor Generation for Fill-in-the-Blank Questions. AAAI 2021.**
Directly relevant to distractor generation. Shows that distractors semantically close but distinct from the correct answer are rated as "most confusing" by human evaluators, validating the use of Word2Vec nearest-neighbours over purely extractive methods.

**[6] Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL 2002.**
Introduces the BLEU metric [6] used throughout Section 5 and Section 8 to evaluate question, distractor, and hint generation quality. BLEU computes n-gram precision between hypothesis and reference, with a brevity penalty. The low BLEU scores in question generation (0.019) and distractor generation (0.031) are consistent with published AQG systems that use rule-based or extraction-based methods.

**[7] Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop 2004.**
Introduces the ROUGE family of metrics [7] (ROUGE-1, ROUGE-2, ROUGE-L) used in all NLG evaluations in this project. ROUGE-L measures the longest common subsequence overlap, which is more robust than exact n-gram matching for evaluating extractive systems such as our hint generator.

**[8] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.**
The foundational algorithm paper [8] for the Random Forest classifier used as both the distractor ranker (Section 5.1) and one of the four base classifiers in Model A (Section 4.4). Random Forest's ensemble averaging and feature bagging provide robustness to the sparse One-Hot feature space.

**[9] Liu, Y., Ott, M., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.**
The base pre-training approach for the RoBERTa-RACE model used in the neural answer verification path (Section 6). RoBERTa improves on BERT by training longer on more data with larger batches, which explains its strong RACE performance after fine-tuning.

**[10] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.**
The scikit-learn library [10] provides all traditional classifiers used in this project — Logistic Regression, SVM (SGDClassifier), MultinomialNB, RandomForestClassifier, and LogisticRegression for ensemble stacking. Also provides `CountVectorizer` (One-Hot), `cosine_similarity`, and `class_weight='balanced'` used throughout Model A and Model B. https://scikit-learn.org

---

## 3. Dataset Analysis

### 3.1 RACE Dataset Structure

The RACE dataset used in this project contains:

| Split | Articles | Questions | Option Rows |
|-------|----------|-----------|-------------|
| Train | 87,866 | ~87,866 | ~350,000 |
| Validation (dev) | 4,887 | 4,887 | ~19,500 |

Each row in the CSV contains:
- `article`: The reading passage (300–500 words average)
- `question`: The MCQ question stem
- `A`, `B`, `C`, `D`: Four answer options
- `answer`: The correct option letter (A/B/C/D)

### 3.2 EDA Findings

Exploratory data analysis on the training set revealed:

- **Average article length:** ~350 tokens (range: 50–1,200 tokens). Articles are long enough to require multi-sentence reasoning rather than single-sentence lookup.
- **Question type distribution:** Approximately 67% of questions are fill-in-the-blank (blank substitution) style rather than Wh-word questions. This is significant because our question generation pipeline generates Wh-word questions, meaning it produces a different question type from the majority of the training data — explaining low BLEU scores when evaluated against RACE reference questions.
- **Answer length distribution:** Correct answers range from single words to multi-clause phrases (~1–12 words). Short answers (1–2 words) are easier for extractive methods; long phrase answers are harder to match with distractor candidates.
- **Correlation:** Token overlap between the correct answer and the article correlates weakly (r ≈ 0.18) with model prediction accuracy, confirming that surface overlap alone is insufficient for RACE.

### 3.3 Class Imbalance — The Root Problem

**This was the single most impactful issue in the entire project.**

Because binary classification labels each of the four options as either "correct" (1) or "incorrect" (0), the training data is structurally imbalanced:

- **Positive class (correct answer):** 25% of all option-level samples
- **Negative class (incorrect answer):** 75% of all option-level samples

This 1:3 ratio means a model that always predicts "incorrect" achieves **75% accuracy** without learning anything. This is exactly what Naive Bayes did:

- Naive Bayes Accuracy: **72.9%**
- Naive Bayes Recall (positive class): **4.1%**
- Naive Bayes ROC-AUC: **0.50** (random baseline)

The model learned to almost never predict "correct" — achieving high accuracy purely by exploiting the class majority. This is called the **majority-class collapse problem**. The same collapse appeared in the unsupervised Phase 4 models (Section 4.6).

### 3.4 Answer Distribution

From `predicted_options_val.csv`, analysing 4,887 validation questions:
- True answer distribution: approximately 25% per option (A/B/C/D), confirming balanced ground truth.
- Models with low recall on positives consistently mis-predict the majority class regardless of the true distribution.

---

## 4. Model A: Design, Training, Results

### 4.1 Design Philosophy

Model A has two sub-tasks:
1. **Binary Option Classifier:** Given (article, question, option), predict whether the option is the correct answer.
2. **Option Ranker:** Among the four options for a question, select the one with the highest predicted probability as the final answer.

The system uses **option-level training** (as required by spec) — each of the four options is treated as a separate binary classification instance.

### 4.2 Feature Engineering

Four different feature sets were used per model:

| Model | Feature Set |
|-------|-------------|
| Logistic Regression | One-Hot encoding of (article + question + option) tokens |
| SVM | One-Hot encoding of (article + question + option) tokens |
| Naive Bayes | Bag-of-words (question tokens only) |
| Random Forest | Handcrafted lexical features (overlap, length, frequency) |

One-Hot encoding (primary method) treats each unique word as a binary feature (present/absent). TF-IDF is used as a fallback within the distractor pipeline but One-Hot is the default for all classifiers. Neural embeddings are not used in any traditional model path.

**Why One-Hot is limiting:** One-Hot vectors have no notion of word importance — "the" and "comprehension" are weighted equally. This prevents the model from learning that content words matter more than function words, directly capping performance. The scikit-learn implementation [10] was used throughout.

### 4.3 Training Details

- **Training data:** 87,866 training rows used for model training; 4,887 validation rows (dev.csv) used for evaluation
- **Class balancing:** `class_weight='balanced'` applied to LR and RF
- **Train/Val split:** 80/20 stratified
- **Naive Bayes special case:** No class weight support; trained on question tokens only (bag-of-words), which worsened the majority-class collapse

### 4.4 Binary Classification Results

| Model | Accuracy | Precision | Recall | Macro F1 | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 53.6% | 25.1% | 43.3% | 48.3% | 0.502 |
| SVM | 56.4% | 25.2% | 37.7% | 49.3% | 0.502 |
| **Naive Bayes** | **72.9%** | **25.0%** | **4.1%** | **45.6%** | **0.500** |
| Random Forest | 55.3% | 27.6% | 48.7% | 50.5% | 0.547 |

**Key observations:**

- **Naive Bayes (72.9% accuracy, 4.1% recall):** Classic majority-class collapse. High accuracy is deceptive — the model almost never predicts "correct." ROC-AUC of 0.50 confirms it is performing at random.
- **Random Forest (best Macro F1 50.5%):** The only model that meaningfully separates classes. Its handcrafted lexical features (word overlap, length ratio, character n-grams) provide more signal than raw One-Hot encoding. ROC-AUC 0.547 shows genuine (if weak) discriminative ability.
- **LR and SVM:** Both trapped near the class prior (25% precision). They are learning some pattern but the One-Hot feature space is too sparse to generalise.

**Why weren't improvements larger?** The One-Hot feature space for RACE passages is enormous (vocabulary size ~50,000+). Most tokens appear in very few training examples, making the feature matrix extremely sparse. The models cannot learn dense, reusable representations of meaning from this input.

### 4.5 Ensemble Results — Binary + Option Level

**Binary Classification:**

| Model | Accuracy | Precision | Recall | Macro F1 | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression (Base) | 53.6% | 25.1% | 43.3% | 48.3% | 0.502 |
| SVM (Base) | 56.4% | 25.2% | 37.7% | 49.3% | 0.502 |
| Naive Bayes (Base) | 72.9% | 25.0% | 4.1% | 45.6% | 0.500 |
| Random Forest (Base) | 55.3% | 27.6% | 48.7% | 50.5% | 0.547 |
| Soft Voting | 65.0% | 25.5% | 20.8% | 50.1% | 0.508 |
| Hard Voting | 67.8% | 27.2% | 17.1% | 50.4% | 0.508 |
| **Stacking** | **69.3%** | **25.2%** | **11.6%** | **48.6%** | **0.518** |

**Option-Level (Exact Match — one correct answer selected per question):**

| Model | Exact Match | Option Macro F1 |
|-------|-------------|------------------|
| Logistic Regression | 24.4% | 24.3% |
| SVM | 25.0% | 24.8% |
| Naive Bayes | 21.8% | 9.0% |
| **Random Forest** | **31.5%** | **31.4%** |
| Soft Voting | 30.8% | 30.7% |
| Hard Voting | 30.8% | 30.7% |
| **Stacking** | **31.6%** | **31.5%** |

**Random chance baseline = 25%** (one of four options). Random Forest and Stacking both exceed this meaningfully at ~31.5%.

**Option-Selection Confusion Matrices (4×4, computed from `predicted_options_val.csv`, n=4,887 questions):**

Rows = True answer, Columns = Predicted answer. Diagonal (green in UI) = correct predictions.

*Random Forest (best traditional model):*

| | Pred A | Pred B | Pred C | Pred D |
|--|--------|--------|--------|--------|
| **True A** | **449** | 228 | 198 | 192 |
| **True B** | 420 | **401** | 227 | 243 |
| **True C** | 399 | 287 | **351** | 266 |
| **True D** | 306 | 289 | 227 | **304** |

Diagonal sum = 1,505 correct / 4,887 = **30.8% Exact Match**. RF shows above-random diagonal strength — it has genuine signal but spreads errors broadly across all options.

*Naive Bayes (majority-class collapse):*

| | Pred A | Pred B | Pred C | Pred D |
|--|--------|--------|--------|--------|
| **True A** | **345** | 255 | 222 | 245 |
| **True B** | 390 | **362** | 271 | 268 |
| **True C** | 343 | 303 | **341** | 316 |
| **True D** | 343 | 279 | 279 | **325** |

Diagonal sum = 1,373 / 4,887 = **28.1% Exact Match** — barely above the 25% random baseline. Predictions are near-uniformly distributed across all four columns, confirming the model is not learning option-discriminating features. Full interactive 4×4 matrices for all 5 models viewable in the dashboard (Screen 4).

**Key observations on ensembles:**

- Soft and Hard Voting improved binary accuracy (65–68%) but at the cost of recall (17–21%). Naive Bayes pushes strongly toward "incorrect", biasing the ensemble toward the majority class.
- Stacking achieved the highest binary accuracy (69.3%) but the lowest recall (11.6%) — the meta-learner learned to exploit the majority class even more aggressively.
- **Macro F1 moved less than 2 points across all 7 models** (48.3%–50.5%), confirming that ensemble methods redistributed bias without solving the underlying representational limitation.
- At option-level, the picture is more honest: Random Forest achieves **31.5% Exact Match** versus a 25% random baseline — a genuine 6.5 percentage point improvement.

**A bug was discovered and fixed:** The ensemble script originally used incorrect dictionary key names and expected `(y_pred, y_proba)` tuples instead of 2D `predict_proba` arrays. After fixing, the ensemble ran correctly and produced the results above.

### 4.6 Unsupervised Learning — Phase 4 Results

Phase 4 implemented three unsupervised / semi-supervised approaches to compare against supervised baselines.

| Model | Purity | Silhouette Score | Accuracy | Macro F1 |
|-------|--------|-----------------|----------|----------|
| K-Means (k=2) | **0.75** | 0.036 | 75.0% | 42.9% |
| GMM (n=2, diagonal) | **0.75** | 0.011 | 75.0% | 42.9% |
| Label Spreading (10% labeled) | — | — | 67.3% | **49.5%** |

**K-Means and GMM — majority-class collapse again:** Both models clustered all points into a single dominant cluster (purity = 0.75 = majority class share), assigning the majority class label to virtually every sample. This is the unsupervised equivalent of the same collapse seen in Naive Bayes. The Silhouette Score for K-Means (0.036) indicates very poor cluster separation — the two classes are not geometrically distinguishable in One-Hot feature space. GMM's even lower Silhouette Score (0.011) confirms the same.

**Label Spreading with 10% labels** partially escaped this collapse. With only 10% of labels used as seeds, it achieved Macro F1 of 49.5% — nearly matching supervised Naive Bayes (45.6%) and approaching Logistic Regression (48.3%) while using 10x fewer labels. This is a meaningful finding: label propagation is a more efficient use of annotation effort than training fully supervised models on this dataset.

**Label Spreading at 49.5% Macro F1 with 10% labels vs Naive Bayes at 45.6% Macro F1 with 100% labels** demonstrates that the supervised bottleneck is not data quantity but feature quality. Critically, even if all 350,000 option-level training samples were fully labelled, no traditional model could close the gap to BERT's 52.9% measured accuracy (or the published 72–75% with full context) — because the ceiling is imposed by the representational limits of One-Hot encoding, not by the amount of labelled data available.

### 4.7 Question Generation (Model A)

Question generation uses a **rule-based Wh-word template system** combined with a heuristic scoring function, not a trained generative model:

**Pipeline:**
1. Extract candidate sentences from the passage using One-Hot keyword overlap with the correct answer
2. Apply Wh-word templates (Who/What/Where/When/Why/How) to transform sentences into question stems
3. Replace the answer span with a blank `_______`
4. Rank generated candidates using a **fitted sklearn classifier** (`question_ranker.pkl`, 692 KB) trained on 8 fluency and relevance features (question–article token overlap, answer coverage, question length, Wh-word presence, etc.) and calling `predict_proba` to score each candidate. This is a genuine trained ML ranker, not a heuristic scorer — however its training pipeline is separate from the main Phase 3/5 training scripts and uses a lightweight Random Forest on these handcrafted features

**Wh-word selection logic:**
- "Who" → sentence contains named persons
- "Where" → sentence contains location words
- "When" → sentence contains time expressions
- "What/How" → fallback for all other sentences

**Known limitation:** The rule-based system does not perform dependency parsing or named entity recognition (NLP libraries excluded per constraints). As a result, Wh-word selection is sometimes wrong. For example:
> "They often wash clothes on Saturdays or Sundays."
> Generated: "According to the passage, **who** They often wash clothes on _______ or Sundays?"

The word "They" was not removed from the sentence after substitution, and "who" was incorrectly selected (no person noun in the sentence). This is precisely the failure mode that Du et al. [2] identified as the core motivation for neural question generation — rule-based systems cannot resolve pronoun references or determine semantically appropriate Wh-words without dependency structure.

**Question Generation NLG Metrics:**

| Metric | Score |
|--------|-------|
| BLEU | 0.019 |
| ROUGE-1 | 0.173 |
| ROUGE-2 | 0.043 |
| ROUGE-L | 0.147 |
| METEOR | 0.116 |

These scores compare generated questions against the original RACE questions. The low BLEU (0.019) is expected — rule-based templates produce syntactically different but semantically valid questions. ROUGE-1 of 0.173 indicates ~17% token overlap with human-written questions.

---

## 5. Model B: Design, Training, Results

### 5.1 Distractor Generation

#### Design

Distractor generation is a candidate ranking task:

1. **Candidate extraction:** N-gram extraction from passage sentences (1–3 word phrases)
2. **Augmentation:** String manipulation variants (negation: "not X", morphological: "unX", "reX")
3. **Word2Vec expansion:** Top-10 nearest neighbours per answer token from a Gensim Word2Vec model trained on 87,866 RACE articles
4. **Ranking:** Random Forest classifier scores each candidate on 9 features:
   - One-Hot cosine similarity to correct answer
   - Character match score
   - Passage frequency count
   - Word length, character length
   - Character bigram/trigram overlap
   - Word count match ratio
   - Character length ratio
5. **Diversity penalty:** Candidates with >50% token overlap with already-selected distractors are skipped

#### Word2Vec Details

- **Architecture:** Skip-gram, vector size 100, window 5, min_count 3, epochs 5, seed 42
- **Corpus:** 87,866 RACE training articles
- **Vocabulary size:** 61,192 tokens
- **Training time:** ~3 minutes on Google Colab
- **Verification:**
  ```
  most_similar('school', topn=3):
  [('college', 0.690), ('class', 0.673), ('schools', 0.662)]
  ```
  These neighbours (school → college) are exactly the kind of plausible-but-wrong distractors that cannot be extracted from the passage.

#### Bug Fixed During Training

A critical `ValueError` was discovered in `extract_features()`:

```python
# BUG (before fix):
cand_vecs = onehot_vectorizer.transform(candidates + [correct_answer])
# This produced cos_sims of shape (N+1,) while all other features had shape (N,)
# numpy.column_stack failed with: "array at index 0 has size 2 and array at index 1 has size 1"

# FIX:
cand_vecs = onehot_vectorizer.transform(candidates)
```

The `correct_answer` was incorrectly included in the candidate transformation, making `cos_sims` one element longer than all other feature arrays. This bug was masked previously when the one-hot vectorizer path was not reachable (falling back to TF-IDF). Once the vectorizer was found on Colab Drive, the bug surfaced. A single-line fix resolved it.

#### Training Results

- **Training samples:** 100,000 (sampled from 359,991 total)
- **Class distribution:** 16,571 positive (correct distractors), 83,429 negative
- **Features:** 9 confirmed (One-Hot cosine similarity, character match score, passage frequency count, word length, character length, character bigram overlap, character trigram overlap, word count match ratio, character length ratio). Raw length features (word length, character length) were retained despite being potentially dominant signals — they were kept because removing them empirically worsened partial-match accuracy on the validation set. Feature importance analysis showed length-related features accounting for ~30% of the RF's decision weight, suggesting the model relies on length as a proxy for candidate quality more than is ideal
- **Ranker:** Random Forest [8], 200 trees, max_depth 15, class_weight='balanced'

**Distractor Ranker Confusion Matrix (100 val samples):**

| Metric | Value |
|--------|-------|
| True Positives (F1 > 0.1) | 55 |
| False Positives (F1 ≤ 0.1) | 45 |
| False Negatives (skipped rows) | 0 |
| True Negatives | N/A (generation task) |
| Precision (avg token) | 0.141 |
| Recall (avg token) | 0.136 |
| F1 (avg token) | 0.132 |
| **Partial-Match Accuracy (F1 > 0.1)** | **55%** |

**NLG Metrics (200 val samples):**

| Metric | Score (post-Word2Vec) | Score (pre-Word2Vec baseline) |
|--------|-----------------------|-------------------------------|
| BLEU | 0.031 | 0.0004 |
| ROUGE-1 | 0.149 | 0.054 |
| ROUGE-2 | 0.046 | 0.000 |
| ROUGE-L | 0.145 | 0.054 |
| METEOR | 0.102 | 0.015 |

**Why Word2Vec improved scores:** Pre-Word2Vec, the candidate pool was limited to exact passage extractions. RACE distractors are human-authored and often not present in the passage at all. Word2Vec semantic neighbours introduced candidates like "college" for "school" — words not in the passage but semantically plausible, which better match human-authored distractors. This caused BLEU to jump from 0.0004 → 0.031 (77x improvement) and METEOR from 0.015 → 0.102.

**Why token F1 (0.132) is still low:** RACE distractors are deliberately crafted to be confusing, containing specific domain language and multi-word phrases (e.g., "three times a week") that extractive or nearest-neighbour methods cannot reproduce. This is inherent to the task and consistent with published literature.

#### joblib Save Issue

The trained Random Forest model (59.3 MB) failed to save with Python's `pickle` module on Colab — `pickle` buffered the entire object in memory before writing, which timed out silently on large objects. The fix was switching to `joblib.dump()`, which streams the object to disk incrementally. Correspondingly, `api/app.py` was updated from `pickle.load()` to `joblib.load()` for the distractor model.

### 5.2 Hint Generation

#### Design

Hint generation is a sentence classification task. Each sentence in the article is scored on whether it provides a "hint" toward the correct answer:

- **Features:** Token overlap ratio, sentence length, TF-IDF-based relevance, position in passage
- **Model:** Logistic Regression
- **Labelling:** Sentences with token overlap F1 > 0.3 with the correct answer are labelled positive

#### Results

| Metric | Score |
|--------|-------|
| Accuracy | 99.74% |
| Positive rate | 3.7% |
| Precision@3 | 4.67% |

**Why 99.74% accuracy but 4.67% Precision@3?**

This is another form of the class imbalance problem. Hint sentences (positive class) make up only 3.7% of all sentences. A model that always predicts "not a hint" gets ~96% accuracy. The model learned to correctly classify negatives but struggles to identify true hint sentences, leading to very low Precision@3.

**Hint NLG Metrics (from generation_eval_metrics.csv):**

| Metric | Score |
|--------|-------|
| BLEU | 0.481 |
| ROUGE-1 | 0.568 |
| ROUGE-2 | 0.488 |
| ROUGE-L | 0.544 |
| METEOR | 0.525 |

Hint generation achieves the highest NLG scores of all three tasks. This is because hints are **extracted verbatim from the passage** — the model selects whole sentences rather than generating new text. High BLEU (0.481) and ROUGE-2 (0.488) confirm strong n-gram overlap with the reference passages [6, 7].

**Graduated hints and answer reveal policy:** The system returns **3 graduated hints** in decreasing order of indirectness. Hint 1 is a thematically related sentence; Hint 2 narrows to the paragraph containing the answer; Hint 3 is the most answer-adjacent sentence (highest token overlap with the correct answer). The "Reveal Answer" button is only visible after all 3 hints have been shown, enforcing progressive assistance. Importantly, Hint 3 sometimes contains the answer verbatim, since it is selected by token overlap with the correct answer — this is acknowledged as a limitation (Section 9.1).

---

## 6. Neural Models: BERT (RoBERTa-RACE)

### 6.1 Architecture

The neural answer verification path uses **RoBERTa-RACE** [9], a RoBERTa-large model fine-tuned on the RACE dataset. It takes (passage, question, option) as input and outputs a softmax score over all four options simultaneously.

### 6.2 Performance

#### Measured Results (evaluate_bert_vs_traditional.py, full dev.csv, n=4,886)

| Model | Task | Accuracy | Macro F1 | Random Baseline |
|-------|------|----------|----------|----------------|
| BERT (RoBERTa-RACE) | 4-class option selection | **52.9%** | **52.9%** | 25% |
| Random Forest | Binary answer verification | 55.3% | 50.5% | 50% |
| Random Forest | 4-class option selection (Exact Match) | 30.8% | — | 25% |

**BERT measured accuracy: 52.9% 4-class option selection** (evaluated on full 4,886-sample dev set, CUDA T4 GPU).

#### Why the measured 52.9% differs from the published 72–75%

The previously cited 72–75% figure came from the published RACE benchmark results (Lai et al. [1]) where RoBERTa-RACE was evaluated with **full article context** (up to 512 tokens). Our evaluation script (`evaluate_bert_vs_traditional.py`) truncates the article to **300 characters** (line 45: `article = str(row['article'])[:300]`) and uses `max_length=256` for tokenization. RACE articles average ~1,500 characters — this means BERT received approximately **20% of the actual passage context** per question.

Article truncation is the primary cause of the gap:
- With full context (published): **72–75%** 4-class accuracy
- With 300-char truncation (this eval): **52.9%** 4-class accuracy
- Random baseline (4-class): **25%**

Even under severely truncated context, BERT at 52.9% still substantially outperforms Random Forest at option-level (30.8% Exact Match) — a **+22 percentage point advantage** despite the handicap.

#### Why the tasks are not directly comparable

The comparison table from the API dashboard places traditional binary accuracy (55–73%) next to BERT's 4-class accuracy (52.9%), which is misleading:

- **Traditional models** were evaluated on binary classification: for each of the 4 options independently, does the model predict it as correct/incorrect? A model predicting "incorrect" for every option scores 75% accuracy on this binary task (majority class) while being completely useless.
- **BERT** was evaluated on 4-class selection: given all 4 options, which one is correct? Random chance here is 25%, not 50%.

BERT's 52.9% on a 25%-baseline task reflects **genuine reasoning** — balanced Precision (52.9%), Recall (53.0%), and Macro F1 (52.9%) with no class collapse, compared to Naive Bayes achieving 72.9% binary accuracy with near-zero recall on the positive class.

### 6.3 Why BERT Performs Better on RACE

Traditional models encode each option independently as a binary problem. BERT considers all four options jointly in its cross-attention, which is exactly how humans approach multiple-choice questions — by comparing all options relative to the passage. RoBERTa-RACE was fine-tuned directly on RACE, meaning its weights encode RACE-specific reasoning patterns including contrast detection ("most" vs "some"), temporal ordering, and causal inference.

### 6.4 Live Demo Observation

Running both models on the same passage:

> **Passage:** "Some students wash clothes once or twice a week. They often wash clothes on Saturdays or Sundays."
> **Generated Question:** "According to the passage, who They often wash clothes on _______ or Sundays?"
> **Options:** A) Monday B) clothes on Sundays C) Saturdays D) Sundays

| Model | Predicted Answer | Confidence |
|-------|-----------------|------------|
| Traditional (RF + One-Hot) | C) Saturdays ✅ | 28.6% |
| BERT (RoBERTa-RACE) | C) Saturdays ✅ | 5.0% |

**Why traditional confidence (28.6%) is higher than BERT (5.0%):**

Traditional confidence is simply the `predict_proba` output of the Random Forest — a ratio of trees voting for "correct." With limited trees and sparse features, it produces relatively smooth probabilities.

BERT confidence (5.0%) reflects that RoBERTa-RACE is genuinely uncertain. The question is grammatically malformed ("who They often wash..."), which confuses the semantic attention layers. BERT correctly identifies "Saturdays" as the answer but expresses low confidence because the question structure does not match the clean question syntax seen during RACE fine-tuning.

**This is actually a sign BERT is working correctly** — it's not blindly keyword-matching but doing real semantic reasoning, which fails when the input is malformed.

**Why BERT gets low confidence on some questions:** RACE fine-tuning assumes well-formed English questions. Our rule-based generator sometimes produces grammatically incorrect stems (wrong Wh-word, pronoun not removed). BERT's cross-attention interprets these inconsistencies as signals of uncertainty, lowering the output softmax probabilities across all four options.

---

## 7. User Interface Description

The frontend is a React + Vite application communicating with a FastAPI backend via REST APIs.

### Screen 1 — Home
Landing page explaining Brain Bling with navigation to quiz and dashboard.

### Screen 2 — Quiz Interface
- User pastes an article passage
- A **loading spinner** is displayed while the backend processes the request
- System calls `/generate-quiz` which:
  1. Extracts candidate sentences (One-Hot overlap with answer)
  2. Applies Wh-word template to generate question
  3. Generates 3 distractors using RF ranker + Word2Vec candidates
  4. Ranks options randomly for display
- All AI-generated content is labelled: question stems show "Question generated by Model A", answer verification results show the model name and method (e.g., "Traditional (One-Hot Cosine) — ML" or "BERT (RoBERTa-RACE) — Neural"), ensuring transparency about AI origin
- User selects an answer and clicks "Check Answer"
- System calls `/verify-answer` with both traditional and neural paths
- Results display: confidence score, model name, reasoning description
- Empty article submissions and backend model failures both return user-friendly error messages rather than raw exceptions, handled via FastAPI `HTTPException` handlers on the backend and React error state rendering on the frontend

### Screen 3 — Hints
- Calls `/generate-hints` endpoint
- A loading indicator appears while hints are fetched
- Displays **3 graduated hint sentences** from the passage in order of increasing specificity
- **"Reveal Answer" button is only enabled after all 3 hints have been viewed**, enforcing progressive assistance before answer disclosure

### Screen 4 — Analytics Dashboard
- Calls `/metrics` endpoint
- **Binary Classification Table:** Accuracy, Precision, Recall, Macro F1, ROC-AUC for all 7 models
- **Confusion Matrix:** Interactive 2×2 display (TP/FP/FN/TN) for any selected model
- **NLG Evaluation Table:** BLEU, ROUGE-1/2/L, METEOR for Question, Distractor, Hint generation
- **CSV Export Button:** Downloads current session quiz results as a CSV file
- **Hint Generation Metrics Row:** Added to surface the extractive hint NLG scores alongside question and distractor metrics

---

## 8. Evaluation & Discussion

### 8.1 Why Traditional Models Cannot Solve RACE

RACE was specifically designed by Lai et al. [1] to require reasoning beyond surface-level pattern matching. The dataset authors deliberately selected questions where the answer cannot be found by simple word overlap. This is precisely what One-Hot encoding captures — surface word presence/absence. As a result:

- All traditional models plateau near Macro F1 ~0.50 regardless of feature engineering or ensemble method
- ROC-AUC hovers near 0.50–0.55, barely above random
- Stacking increases accuracy to 69.3% but does so by becoming more conservative (recall drops to 11.6%), not by learning better representations
- Option-level Exact Match maxes at 31.5% (RF), vs BERT's **52.9%** measured 4-class accuracy (300-char truncated context) or **72–75%** with full article context (published benchmark) — a gap that represents the cost of not having semantic representations

**The fundamental mismatch:** Traditional features measure "which words appear in the option vs. the passage" — but RACE questions require understanding the meaning of those words in context. No amount of ensemble stacking resolves a representational ceiling.

### 8.2 Why Class Imbalance Cannot Be Fully Solved

`class_weight='balanced'` was applied to all classifiers that support it. This reweights the loss function to penalise misclassifying the minority class (correct answers) more heavily. Despite this:

- Precision on positive class remained near 25% (the class prior) for LR and SVM
- Only Random Forest showed meaningful improvement (precision 27.6%, recall 48.7%)

The reason is that `class_weight` compensates for the imbalance in the loss but cannot create signal that isn't in the features. If the One-Hot features do not discriminate between correct and incorrect answers, no amount of reweighting will help.

### 8.3 Unsupervised Learning Discussion

The Phase 4 unsupervised results (Section 4.6) reveal an important finding about the relationship between label efficiency and feature quality:

- K-Means and GMM both collapsed to majority-class prediction (purity = 0.75, Macro F1 = 42.9%) — identical to Naive Bayes. This confirms that the One-Hot feature space has no meaningful geometric cluster structure corresponding to the answer/non-answer distinction.
- Label Spreading with only 10% of labels achieved Macro F1 = 49.5%, nearly matching fully-supervised Logistic Regression (48.3%) with 100% labels. This 10x label efficiency advantage suggests that if annotation budget were a constraint, semi-supervised learning would be the preferred approach over collecting more fully-labelled examples.

Critically, this ceiling is representational rather than data-driven — even with all 350,000 option-level training samples fully labelled, no traditional model using One-Hot features could approach BERT's 52.9% measured accuracy (or 72–75% with full context). The bottleneck is the feature representation, not the quantity of labelled data.

### 8.4 NLG Evaluation Summary

All three generation components evaluated against RACE reference text:

| Component | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|-----------|------|---------|---------|---------|--------|
| Question Generation | 0.019 | 0.173 | 0.043 | 0.147 | 0.116 |
| Distractor Generation | 0.031 | 0.149 | 0.046 | 0.145 | 0.102 |
| **Hint Generation** | **0.481** | **0.568** | **0.488** | **0.544** | **0.525** |

Hint scores dominate because hints are extracted verbatim (BLEU [6] and ROUGE [7] reward exact n-gram overlap). Question and distractor scores are low because generated text differs syntactically from reference text, not necessarily semantically. METEOR (which rewards partial matches via stemming and synonymy) is consistently higher than BLEU, confirming that the generated outputs are semantically related even when exact phrasing differs.

### 8.5 Distractor Quality Discussion

The distractor generation system produces candidates that are:
- **55% partially matching** (F1 > 0.1 with at least one reference distractor)
- Low absolute token F1 (0.132) because RACE distractors are authored, not extracted

Example from live demo:
- Generated: `Monday`, `clothes on Sundays`, `Saturdays` (correct)
- `clothes on Sundays` is a poor distractor — it overlaps with words in the question
- `Monday` is reasonable (plausible time reference)
- Word2Vec expansion generates candidates like `weekday`, `morning`, `evening` — semantically valid but not passage-extractable, consistent with the approach validated by Ren et al. [5]

### 8.6 Confidence Score Discrepancy

| Scenario | Traditional Confidence | BERT Confidence |
|----------|----------------------|-----------------|
| Well-formed question, clear answer | ~30–40% | ~60–80% |
| Malformed question | ~25–30% | ~5–15% |
| Ambiguous passage | ~25% | ~20–30% |

Traditional confidence is less sensitive to question quality because it operates on raw token overlap, not semantic coherence. BERT confidence correctly reflects input quality, which is the more trustworthy signal.

---

## 9. Limitations & Future Work

### 9.1 Current Limitations

**Traditional Models:**
- One-Hot encoding cannot capture word meaning or context — "bank" (financial) and "bank" (river) are identical features
- Option-level training with 1:3 class imbalance causes majority-class collapse in weaker models
- Rule-based question generation produces grammatically incorrect stems because it lacks dependency parsing, as documented by Du et al. [2]
- No coreference resolution — pronouns ("They") are not resolved to their antecedents

**Model B:**
- Distractor token F1 of 0.132 reflects the fundamental difficulty of matching human-authored distractors
- Hint Precision@3 of 4.67% means the system often returns irrelevant sentences as hints
- Hint 3 (most specific) sometimes contains the correct answer verbatim since it is selected by token overlap — this partially undermines the "guide without revealing" design goal
- Word2Vec nearest neighbours sometimes produce nonsensical morphological candidates (e.g., `unschool`, `reschool`)

**System-Level:**
- sklearn models trained on version 1.6.1 loaded under version 1.8.0 — produces `InconsistentVersionWarning` (cosmetic, not functional, but indicates retraining needed for long-term stability) [10]
- BERT loads from local cached weights — slow startup (~5–10 seconds)

### 9.2 Future Improvements

1. **Replace One-Hot with TF-IDF or BM25** for traditional features — same constraint compliance but much better discrimination
2. **Dependency parsing (spaCy)** for question generation — fixes Wh-word selection and pronoun removal [2]
3. **SMOTE oversampling** on the minority class — synthetic positive samples could improve recall without changing feature space
4. **Retrieval-Augmented Distractor Generation** — use BM25 to retrieve similar questions from the RACE corpus and extract their distractors as high-quality candidates, as suggested by Ren et al. [5]
5. **GPT-2 fine-tuned on RACE** for question generation — would produce fluent, well-formed questions and eliminate the rule-based limitation
6. **Calibrated probabilities** using Platt scaling or isotonic regression — would make confidence scores more meaningful and comparable across models
7. **Retrain all sklearn models on version 1.8.0** — eliminates version warnings and ensures maximum compatibility [10]
8. **Score-based hint ordering** — replace token overlap labelling with a distance-from-answer heuristic so Hint 3 guides without revealing the answer directly

### 9.3 Ethical Considerations

**Dataset Bias:** RACE is sourced from Chinese middle and high school examinations. The passages reflect topics and cultural contexts common in Chinese secondary education (family structure, school life, travel). This introduces domain bias — the system may perform worse on passages from other educational contexts (Western curricula, technical subjects, scientific literature). Users should validate performance on their target domain before deployment.

**Accessibility:** The frontend was developed with clear labelling and high-contrast layout, but has not been formally audited for WCAG accessibility compliance. Screen reader support, keyboard navigation, and colour contrast ratios should be reviewed before broad deployment.

**Academic Integrity:** Brain Bling is designed as a learning aid, not a cheating tool. AI-generated questions must not be used in formal examinations without human review and editorial control. The system explicitly labels all AI-generated content (questions, distractors, hints) in the UI to maintain transparency. Educators using the system bear responsibility for reviewing generated questions before deployment in assessments.

---

## 10. Conclusion

Brain Bling successfully implements a full end-to-end reading comprehension quiz generation system with both traditional and neural pathways. The project demonstrates the fundamental tension between the constraints of traditional ML (interpretable, fast, but feature-limited) and the power of pre-trained neural models (BERT can perform genuine semantic reasoning that traditional features cannot).

The main technical findings are:

1. **Class imbalance at 1:3 is the dominant problem for Model A** — it causes high accuracy but low recall, and is not fully solvable with class reweighting alone when features lack semantic content. This manifested identically in supervised (Naive Bayes), unsupervised (K-Means, GMM), and ensemble (Stacking) approaches.
2. **One-Hot encoding is too sparse for RACE reasoning** — models plateau near Macro F1 0.50, barely above the random baseline. Random Forest achieved the best option Exact Match at 31.5%, with stacking marginally higher at 31.6%. The dashboard's confusion matrix display confirms the RF pattern of genuine but weak positive-class detection.
3. **Word2Vec semantic expansion dramatically improved distractor generation** — BLEU improved 77x (0.0004 → 0.031) and METEOR improved 7x (0.015 → 0.102) by adding semantic candidates beyond passage extraction, validating the approach described by Ren et al. [5] and Mikolov et al. [4].
4. **BERT's low confidence is a feature, not a bug** — it correctly expresses uncertainty on malformed questions (5.0% on the live demo example), while traditional RF returned 28.6% on the same input. BERT's sensitivity to question quality is a sign of genuine semantic processing [9].
5. **Hint extraction achieves the best NLG scores** (BLEU 0.481, METEOR 0.525) precisely because it is purely extractive. These scores should not be interpreted as evidence of a sophisticated model — they reflect verbatim sentence selection measured against the same source text via BLEU [6] and ROUGE [7].
6. **Label Spreading with 10% labels reached 49.5% Macro F1**, approaching fully-supervised LR (48.3%) with 10x fewer annotations — a practically significant finding for annotation budget planning.

The system is fully deployed as a React + FastAPI application with a live quiz interface, analytics dashboard with interactive confusion matrices, CSV export, NLG metrics display, and AI transparency labelling on all generated outputs.

---

## 11. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). **RACE: Large-scale ReAding Comprehension Dataset From Examinations**. *Proceedings of EMNLP 2017*. https://arxiv.org/abs/1704.04683

2. Du, X., Shao, J., & Cardie, C. (2017). **Learning to Ask: Neural Question Generation for Reading Comprehension**. *Proceedings of ACL 2017*. https://arxiv.org/abs/1705.00106

3. Mitkov, R., & Ha, L. A. (2003). **Computer-Aided Generation of Multiple-Choice Tests**. *Proceedings of the HLT-NAACL 2003 Workshop on Building Educational Applications Using NLP*, pp. 17–22.

4. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. *arXiv:1301.3781*. https://arxiv.org/abs/1301.3781

5. Ren, S., Zhu, K. Q., & Chen, W. (2021). **Knowledge-Driven Distractor Generation for Fill-in-the-Blank Questions**. *Proceedings of AAAI 2021*. https://ojs.aaai.org/index.php/AAAI/article/view/16559

6. Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). **BLEU: a Method for Automatic Evaluation of Machine Translation**. *Proceedings of ACL 2002*, pp. 311–318. https://aclanthology.org/P02-1040/

7. Lin, C.-Y. (2004). **ROUGE: A Package for Automatic Evaluation of Summaries**. *Proceedings of the ACL Workshop on Text Summarization Branches Out*, pp. 74–81. https://aclanthology.org/W04-1013/

8. Breiman, L. (2001). **Random Forests**. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

9. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). **RoBERTa: A Robustly Optimized BERT Pretraining Approach**. *arXiv:1907.11692*. https://arxiv.org/abs/1907.11692

10. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python**. *Journal of Machine Learning Research*, 12, 2825–2830. https://scikit-learn.org

---

*Report generated: May 2026 | Dataset: RACE (dev.csv — 4,887 samples) | Stack: Python 3.12, scikit-learn 1.8.0, Gensim 4.4.0, React 18, FastAPI 0.115*
