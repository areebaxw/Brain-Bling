"""
TEST_INFERENCE.PY — Unit Tests for Inference Module
=====================================================
Tests the core inference functions: answer verification,
hint generation, and question generation pipeline.
Run with: python -m pytest tests/test_inference.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import generate_hints, load_all_models

ARTICLE = (
    "Marie Curie was a physicist and chemist who conducted pioneering research on "
    "radioactivity. She was born in Warsaw in 1867. She was the first woman to win a "
    "Nobel Prize and the only person to win Nobel Prizes in two different sciences. "
    "She received her first Nobel Prize in Physics in 1903 and her second in Chemistry in 1911."
)
QUESTION = "What field did Marie Curie conduct research in?"
ANSWER   = "radioactivity"


@pytest.fixture(scope="module")
def models():
    return load_all_models()


class TestHintGeneration:
    def test_returns_list(self, models):
        hints = generate_hints(models, ARTICLE, QUESTION, ANSWER)
        assert isinstance(hints, list)

    def test_returns_nonempty(self, models):
        hints = generate_hints(models, ARTICLE, QUESTION, ANSWER)
        assert len(hints) >= 1

    def test_returns_at_most_top_k(self, models):
        hints = generate_hints(models, ARTICLE, QUESTION, ANSWER, top_k=3)
        assert len(hints) <= 3

    def test_hints_are_strings(self, models):
        hints = generate_hints(models, ARTICLE, QUESTION, ANSWER)
        for h in hints:
            assert isinstance(h, str) and len(h) > 0

    def test_fallback_on_empty_article(self, models):
        hints = generate_hints(models, "", QUESTION, ANSWER)
        assert isinstance(hints, list)

    def test_custom_top_k(self, models):
        hints = generate_hints(models, ARTICLE, QUESTION, ANSWER, top_k=1)
        assert len(hints) <= 1


class TestVerifyAnswer:
    def test_returns_tuple(self, models):
        from src.inference import verify_answer
        result = verify_answer(models, ARTICLE, QUESTION, ANSWER, [ANSWER])
        assert isinstance(result, tuple) and len(result) == 2

    def test_confidence_in_range(self, models):
        from src.inference import verify_answer
        _, conf = verify_answer(models, ARTICLE, QUESTION, ANSWER, [ANSWER])
        assert 0.0 <= conf <= 1.0

    def test_fallback_no_models(self):
        from src.inference import verify_answer
        is_correct, conf = verify_answer({}, ARTICLE, QUESTION, ANSWER, [ANSWER])
        assert is_correct is False and conf == 0.0


class TestQuestionGenerationPipeline:
    def test_wh_question_returns_tuple(self):
        import re
        from api.app import generate_wh_question
        q, a = generate_wh_question("Marie Curie was born in Warsaw in 1867.")
        assert q is None or isinstance(q, str)
        assert a is None or isinstance(a, str)

    def test_short_sentence_returns_none(self):
        from api.app import generate_wh_question
        q, a = generate_wh_question("She ran.")
        assert q is None and a is None

    def test_question_ends_with_questionmark(self):
        from api.app import generate_wh_question
        for sent in [
            "Marie Curie was born in Warsaw in 1867.",
            "Albert Einstein developed the theory of relativity in Germany.",
            "The Eiffel Tower was built in Paris during the 19th century.",
        ]:
            q, _ = generate_wh_question(sent)
            if q is not None:
                assert q.endswith("?"), f"Question does not end with '?': {q}"
