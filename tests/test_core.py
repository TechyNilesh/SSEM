"""Tests for SSEM core module."""

import pytest
from SSEM.core import Citation, CITATIONS, MetricResult, EmbeddingEngine


class TestCitation:
    def test_creation(self):
        c = Citation(
            title="Test Paper",
            authors="Author A",
            year=2024,
            venue="Test Venue",
            url="https://example.com",
            description="A test citation.",
        )
        assert c.title == "Test Paper"
        assert c.year == 2024

    def test_str(self):
        c = Citation(
            title="Test",
            authors="A",
            year=2024,
            venue="V",
            url="https://x.com",
            description="D",
        )
        s = str(c)
        assert "A (2024)" in s
        assert "Test" in s

    def test_to_dict(self):
        c = Citation(
            title="T",
            authors="A",
            year=2024,
            venue="V",
            url="https://x.com",
            description="D",
        )
        d = c.to_dict()
        assert d["title"] == "T"
        assert isinstance(d, dict)

    def test_citations_registry(self):
        assert "bertscore" in CITATIONS
        assert "semantic_similarity" in CITATIONS
        assert "nli_faithfulness" in CITATIONS
        assert "pass_at_k" in CITATIONS
        assert "agent_eval" in CITATIONS


class TestMetricResult:
    def _make_result(self, score=0.75):
        return MetricResult(
            score=score,
            metric_name="Test Metric",
            score_range=(0.0, 1.0),
            interpretation="Good score.",
            method="Step 1. Did something.\nStep 2. Did more.",
            model_used="test-model",
            citations=[CITATIONS["bertscore"]],
            details={"key": "value"},
            elapsed_sec=0.5,
        )

    def test_repr(self):
        r = self._make_result()
        assert "Test Metric" in repr(r)
        assert "0.7500" in repr(r)

    def test_explain(self):
        r = self._make_result()
        explanation = r.explain()
        assert "Test Metric" in explanation
        assert "0.7500" in explanation
        assert "How This Score Was Computed" in explanation
        assert "Research Citations" in explanation
        assert "BERTScore" in explanation

    def test_to_dict(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["score"] == 0.75
        assert d["metric_name"] == "Test Metric"
        assert len(d["citations"]) == 1
        assert d["details"]["key"] == "value"


class TestEmbeddingEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        return EmbeddingEngine(
            model_name="bert-base-uncased",
            device="cpu",
        )

    def test_encode(self, engine):
        embs = engine.encode(["Hello world", "Test sentence"])
        assert embs.shape[0] == 2
        assert embs.shape[1] > 0  # hidden dim

    def test_encode_cls(self, engine):
        embs = engine.encode_cls(["Hello world"])
        assert embs.shape[0] == 1

    def test_encode_tokens(self, engine):
        token_embs = engine.encode_tokens(["Hello world"])
        assert len(token_embs) == 1
        assert token_embs[0].ndim == 2  # (n_tokens, hidden_dim)

    def test_repr(self, engine):
        assert "bert-base-uncased" in repr(engine)
