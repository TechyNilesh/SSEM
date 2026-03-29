"""Tests for all SSEM metrics."""

import pytest
from SSEM.core import EmbeddingEngine, MetricResult


@pytest.fixture(scope="module")
def engine():
    return EmbeddingEngine(model_name="bert-base-uncased", device="cpu")


# ------------------------------------------------------------------
# Semantic Similarity
# ------------------------------------------------------------------

class TestSemanticSimilarity:
    def test_cosine(self, engine):
        from SSEM.semantic import SemanticSimilarityMetric

        m = SemanticSimilarityMetric(engine=engine, metric="cosine")
        result = m.evaluate(
            ["The cat sat on the mat."],
            ["A cat was sitting on a mat."],
        )
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert "cosine" in result.metric_name.lower()
        assert len(result.citations) > 0

    def test_euclidean(self, engine):
        from SSEM.semantic import SemanticSimilarityMetric

        m = SemanticSimilarityMetric(engine=engine, metric="euclidean")
        result = m.evaluate(["Hello"], ["Hi"])
        assert 0.0 <= result.score <= 1.0

    def test_transparency(self, engine):
        from SSEM.semantic import SemanticSimilarityMetric

        m = SemanticSimilarityMetric(engine=engine)
        result = m.evaluate(["Test"], ["Test"])
        explanation = result.explain()
        assert "How This Score Was Computed" in explanation
        assert "Research Citations" in explanation


class TestBERTScore:
    def test_basic(self, engine):
        from SSEM.semantic import BERTScoreMetric

        m = BERTScoreMetric(engine=engine)
        result = m.evaluate(
            ["The cat sat on the mat."],
            ["A cat was sitting on a mat."],
        )
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert "precision" in result.details
        assert "recall" in result.details
        assert "f1" in result.details

    def test_identical(self, engine):
        from SSEM.semantic import BERTScoreMetric

        m = BERTScoreMetric(engine=engine)
        result = m.evaluate(["Hello world"], ["Hello world"])
        assert result.score >= 0.9  # identical should be very high

    def test_unequal_lengths(self, engine):
        from SSEM.semantic import BERTScoreMetric

        m = BERTScoreMetric(engine=engine)
        with pytest.raises(ValueError):
            m.evaluate(["A", "B"], ["C"])


# ------------------------------------------------------------------
# Faithfulness & Hallucination
# ------------------------------------------------------------------

class TestFaithfulness:
    def test_embedding_based(self, engine):
        from SSEM.faithfulness import FaithfulnessMetric

        m = FaithfulnessMetric(engine=engine, threshold=0.5)
        result = m.evaluate(
            output_text="The cat is on the mat. The dog is outside.",
            source_context="A cat was sitting on a mat in the living room.",
        )
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert "claims" in result.details

    def test_empty_output(self, engine):
        from SSEM.faithfulness import FaithfulnessMetric

        m = FaithfulnessMetric(engine=engine)
        result = m.evaluate(output_text="", source_context="Some context.")
        assert result.score == 1.0  # trivially faithful


class TestHallucination:
    def test_basic(self, engine):
        from SSEM.faithfulness import HallucinationMetric

        m = HallucinationMetric(engine=engine, threshold=0.5)
        result = m.evaluate(
            output_text="The cat is on the mat.",
            source_context="A cat was sitting on a mat.",
        )
        assert 0.0 <= result.score <= 1.0
        assert "faithfulness_score" in result.details


# ------------------------------------------------------------------
# Answer Relevancy
# ------------------------------------------------------------------

class TestAnswerRelevancy:
    def test_basic(self, engine):
        from SSEM.relevancy import AnswerRelevancyMetric

        m = AnswerRelevancyMetric(engine=engine)
        result = m.evaluate(
            questions=["What color is the sky?"],
            answers=["The sky is blue."],
        )
        assert 0.0 <= result.score <= 1.0
        assert "per_pair" in result.details


# ------------------------------------------------------------------
# Agentic Metrics
# ------------------------------------------------------------------

class TestReasoningCoherence:
    def test_coherent_chain(self, engine):
        from SSEM.agentic import ReasoningCoherenceMetric

        m = ReasoningCoherenceMetric(engine=engine)
        result = m.evaluate(
            reasoning_steps=[
                "First, I need to find the relevant data.",
                "Next, I will analyze the data patterns.",
                "Finally, I will summarize the findings.",
            ],
        )
        assert 0.0 <= result.score <= 1.0
        assert "sequential_similarities" in result.details

    def test_with_goal(self, engine):
        from SSEM.agentic import ReasoningCoherenceMetric

        m = ReasoningCoherenceMetric(engine=engine)
        result = m.evaluate(
            reasoning_steps=["Find data.", "Analyze data."],
            goal="Produce a data analysis report.",
        )
        assert result.details["goal_alignment"] is not None

    def test_single_step(self, engine):
        from SSEM.agentic import ReasoningCoherenceMetric

        m = ReasoningCoherenceMetric(engine=engine)
        result = m.evaluate(reasoning_steps=["Only one step."])
        assert result.score == 1.0


class TestToolCallAccuracy:
    def test_perfect_match(self):
        from SSEM.agentic import ToolCallAccuracyMetric

        m = ToolCallAccuracyMetric()
        result = m.evaluate(
            predicted_calls=[
                {"tool": "search", "params": {"query": "python"}},
                {"tool": "read", "params": {"file": "main.py"}},
            ],
            expected_calls=[
                {"tool": "search", "params": {"query": "python"}},
                {"tool": "read", "params": {"file": "main.py"}},
            ],
        )
        assert result.score == 1.0

    def test_wrong_tool(self):
        from SSEM.agentic import ToolCallAccuracyMetric

        m = ToolCallAccuracyMetric()
        result = m.evaluate(
            predicted_calls=[{"tool": "write", "params": {}}],
            expected_calls=[{"tool": "read", "params": {}}],
        )
        assert result.score < 1.0

    def test_empty_expected(self):
        from SSEM.agentic import ToolCallAccuracyMetric

        m = ToolCallAccuracyMetric()
        result = m.evaluate(predicted_calls=[], expected_calls=[])
        assert result.score == 1.0


class TestTaskCompletion:
    def test_checklist_mode(self, engine):
        from SSEM.agentic import TaskCompletionMetric

        m = TaskCompletionMetric(engine=engine)
        result = m.evaluate(
            agent_output="I found the data and created a summary report.",
            expected_criteria=["Find the data", "Create a summary"],
        )
        assert 0.0 <= result.score <= 1.0
        assert result.details["mode"] == "checklist"

    def test_reference_mode(self, engine):
        from SSEM.agentic import TaskCompletionMetric

        m = TaskCompletionMetric(engine=engine)
        result = m.evaluate(
            agent_output="The analysis is complete.",
            reference_output="The analysis has been completed successfully.",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.details["mode"] == "reference"


# ------------------------------------------------------------------
# Consistency
# ------------------------------------------------------------------

class TestMultiTurnConsistency:
    def test_consistent_turns(self, engine):
        from SSEM.consistency import MultiTurnConsistencyMetric

        m = MultiTurnConsistencyMetric(engine=engine)
        result = m.evaluate(
            responses=[
                "Python is a great programming language.",
                "I recommend Python for data science.",
                "Python has excellent libraries for ML.",
            ],
        )
        assert 0.0 <= result.score <= 1.0

    def test_single_turn(self, engine):
        from SSEM.consistency import MultiTurnConsistencyMetric

        m = MultiTurnConsistencyMetric(engine=engine)
        result = m.evaluate(responses=["Just one response."])
        assert result.score == 1.0


class TestSelfCheck:
    def test_consistent_samples(self, engine):
        from SSEM.consistency import SelfCheckConsistencyMetric

        m = SelfCheckConsistencyMetric(engine=engine)
        result = m.evaluate(
            main_response="Paris is the capital of France.",
            sampled_responses=[
                "The capital of France is Paris.",
                "Paris serves as France's capital city.",
            ],
        )
        assert 0.0 <= result.score <= 1.0
        assert "claims" in result.details


# ------------------------------------------------------------------
# Code Evaluation
# ------------------------------------------------------------------

class TestCodeCorrectness:
    def test_passing_code(self):
        from SSEM.code_eval import CodeCorrectnessMetric

        m = CodeCorrectnessMetric(timeout=5)
        result = m.evaluate(
            code_samples=["def add(a, b): return a + b"],
            test_code="assert add(1, 2) == 3\nassert add(0, 0) == 0",
        )
        assert result.score == 1.0
        assert result.details["pass_count"] == 1

    def test_failing_code(self):
        from SSEM.code_eval import CodeCorrectnessMetric

        m = CodeCorrectnessMetric(timeout=5)
        result = m.evaluate(
            code_samples=["def add(a, b): return a - b"],
            test_code="assert add(1, 2) == 3",
        )
        assert result.score == 0.0

    def test_multiple_samples(self):
        from SSEM.code_eval import CodeCorrectnessMetric

        m = CodeCorrectnessMetric(timeout=5, k_values=[1, 2])
        result = m.evaluate(
            code_samples=[
                "def add(a, b): return a + b",
                "def add(a, b): return a - b",
            ],
            test_code="assert add(1, 2) == 3",
        )
        assert 0.0 < result.score <= 1.0
        assert "pass@1" in result.details["pass_at_k"]
        assert "pass@2" in result.details["pass_at_k"]


# ------------------------------------------------------------------
# Unified SSEM API
# ------------------------------------------------------------------

class TestSSEMUnified:
    @pytest.fixture(scope="class")
    def evaluator(self):
        from SSEM.evaluator import SSEM

        return SSEM(model_name="bert-base-uncased", device="cpu")

    def test_semantic_similarity(self, evaluator):
        result = evaluator.semantic_similarity(["Hello"], ["Hi"])
        assert isinstance(result, MetricResult)

    def test_bertscore(self, evaluator):
        result = evaluator.bertscore(["Hello world"], ["Hello world"])
        assert result.score >= 0.9

    def test_faithfulness(self, evaluator):
        result = evaluator.faithfulness(
            "The cat is on the mat.", "A cat was sitting on a mat."
        )
        assert isinstance(result, MetricResult)

    def test_answer_relevancy(self, evaluator):
        result = evaluator.answer_relevancy(
            ["What is 2+2?"], ["The answer is 4."]
        )
        assert isinstance(result, MetricResult)

    def test_reasoning_coherence(self, evaluator):
        result = evaluator.reasoning_coherence(
            ["Step 1: Find data.", "Step 2: Analyze it."]
        )
        assert isinstance(result, MetricResult)

    def test_tool_accuracy(self, evaluator):
        result = evaluator.tool_accuracy(
            [{"tool": "search", "params": {"q": "test"}}],
            [{"tool": "search", "params": {"q": "test"}}],
        )
        assert result.score == 1.0

    def test_list_metrics(self, evaluator):
        metrics = evaluator.list_metrics()
        assert "bertscore" in metrics
        assert "faithfulness" in metrics
        assert "reasoning_coherence" in metrics

    def test_list_citations(self, evaluator):
        citations = evaluator.list_citations()
        assert "bertscore" in citations

    def test_evaluate_all(self, evaluator):
        report = evaluator.evaluate_all(
            output_sentences=["The cat sat."],
            reference_sentences=["A cat was sitting."],
            source_context="A cat was observed sitting on a mat.",
        )
        assert len(report.results) >= 2  # at least similarity + faithfulness
        summary = report.summary()
        assert "SSEM Evaluation Report" in summary

    def test_report_json(self, evaluator):
        import json

        report = evaluator.evaluate_all(
            output_sentences=["Hello"],
            reference_sentences=["Hi"],
        )
        j = report.to_json()
        parsed = json.loads(j)
        assert "metrics" in parsed
