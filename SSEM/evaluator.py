"""
SSEM Unified Evaluator
======================

The main entry point for all SSEM metrics.  One class, one model load,
all evaluation capabilities.

Usage:
    from ssem import SSEM

    evaluator = SSEM()
    result = evaluator.bertscore(output_sentences, reference_sentences)
    print(result.explain())  # Full transparency

    report = evaluator.evaluate_all(
        output_sentences=outputs,
        reference_sentences=references,
        source_context=context,
        questions=questions,
    )
    print(report.summary())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from SSEM.core import EmbeddingEngine, MetricResult, CITATIONS
from SSEM.report import EvaluationReport


class SSEM:
    """Unified evaluation interface for LLMs and AI agents.

    Wraps all SSEM metrics behind simple method calls.  The underlying
    embedding model is loaded once and shared across all metrics.

    Parameters:
        model_name: HuggingFace model for embeddings.
                    Default: "bert-base-multilingual-cased".
        device:     "cpu", "cuda", or "mps".  Auto-detected if None.

    All methods return ``MetricResult`` — a transparent, self-documenting
    object that includes the score, method description, model used,
    score range, interpretation, and research citations.

    Call ``.explain()`` on any result for a full human-readable report.
    Call ``.to_dict()`` for JSON-serializable output.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: Optional[str] = None,
    ) -> None:
        self.engine = EmbeddingEngine(model_name=model_name, device=device)

    # ------------------------------------------------------------------
    # Semantic Similarity Metrics
    # ------------------------------------------------------------------

    def semantic_similarity(
        self,
        output_sentences: List[str],
        reference_sentences: List[str],
        metric: str = "cosine",
        pooling: str = "mean",
    ) -> MetricResult:
        """Sentence-level semantic similarity.

        Citations:
            - Vadapalli et al. (2021). GEM @ ACL 2021.
            - Manning et al. (2008). Introduction to Information Retrieval.
        """
        from SSEM.semantic import SemanticSimilarityMetric

        m = SemanticSimilarityMetric(engine=self.engine, metric=metric)
        return m.evaluate(output_sentences, reference_sentences, pooling=pooling)

    def bertscore(
        self,
        output_sentences: List[str],
        reference_sentences: List[str],
    ) -> MetricResult:
        """BERTScore: Token-level precision, recall, and F1.

        Citations:
            Zhang et al. (2020). BERTScore. ICLR 2020.
        """
        from SSEM.semantic import BERTScoreMetric

        m = BERTScoreMetric(engine=self.engine)
        return m.evaluate(output_sentences, reference_sentences)

    # ------------------------------------------------------------------
    # Faithfulness & Hallucination
    # ------------------------------------------------------------------

    def faithfulness(
        self,
        output_text: str,
        source_context: str,
        nli_model: Optional[Any] = None,
        threshold: float = 0.5,
    ) -> MetricResult:
        """Faithfulness: Are output claims grounded in the source?

        Citations:
            - Kryscinski et al. (2020). EMNLP 2020.
            - Williams et al. (2018). MultiNLI. NAACL 2018.
        """
        from SSEM.faithfulness import FaithfulnessMetric

        m = FaithfulnessMetric(
            engine=self.engine, nli_model=nli_model, threshold=threshold
        )
        return m.evaluate(output_text, source_context)

    def hallucination(
        self,
        output_text: str,
        source_context: str,
        nli_model: Optional[Any] = None,
        threshold: float = 0.5,
    ) -> MetricResult:
        """Hallucination score: Fraction of unsupported claims.

        Citations:
            - Kryscinski et al. (2020). EMNLP 2020.
            - Manakul et al. (2023). SelfCheckGPT. EMNLP 2023.
        """
        from SSEM.faithfulness import HallucinationMetric

        m = HallucinationMetric(
            engine=self.engine, nli_model=nli_model, threshold=threshold
        )
        return m.evaluate(output_text, source_context)

    # ------------------------------------------------------------------
    # Answer Relevancy
    # ------------------------------------------------------------------

    def answer_relevancy(
        self,
        questions: List[str],
        answers: List[str],
    ) -> MetricResult:
        """Answer relevancy: Does the answer address the question?

        Citations:
            - Es et al. (2024). RAGAS. EACL 2024.
        """
        from SSEM.relevancy import AnswerRelevancyMetric

        m = AnswerRelevancyMetric(engine=self.engine)
        return m.evaluate(questions, answers)

    # ------------------------------------------------------------------
    # Agentic Evaluation
    # ------------------------------------------------------------------

    def reasoning_coherence(
        self,
        reasoning_steps: List[str],
        goal: Optional[str] = None,
        contradiction_threshold: float = 0.3,
        goal_weight: float = 0.3,
    ) -> MetricResult:
        """Evaluate coherence of a multi-step reasoning chain.

        Citations:
            - Xia et al. (2024). ReasonEval.
            - Liu et al. (2023). AgentBench. ICLR 2024.
        """
        from SSEM.agentic import ReasoningCoherenceMetric

        m = ReasoningCoherenceMetric(
            engine=self.engine,
            contradiction_threshold=contradiction_threshold,
            goal_weight=goal_weight,
        )
        return m.evaluate(reasoning_steps, goal=goal)

    def tool_accuracy(
        self,
        predicted_calls: List[Dict[str, Any]],
        expected_calls: List[Dict[str, Any]],
        selection_weight: float = 0.5,
        param_weight: float = 0.3,
        order_weight: float = 0.2,
    ) -> MetricResult:
        """Evaluate AI agent tool call accuracy.

        Citations:
            - Liu et al. (2023). AgentBench. ICLR 2024.
        """
        from SSEM.agentic import ToolCallAccuracyMetric

        m = ToolCallAccuracyMetric(
            selection_weight=selection_weight,
            param_weight=param_weight,
            order_weight=order_weight,
        )
        return m.evaluate(predicted_calls, expected_calls)

    def task_completion(
        self,
        agent_output: str,
        expected_criteria: Optional[List[str]] = None,
        reference_output: Optional[str] = None,
        threshold: float = 0.6,
    ) -> MetricResult:
        """Graded task completion evaluation for AI agents.

        Citations:
            - Liu et al. (2023). AgentBench. ICLR 2024.
        """
        from SSEM.agentic import TaskCompletionMetric

        m = TaskCompletionMetric(engine=self.engine, threshold=threshold)
        return m.evaluate(
            agent_output,
            expected_criteria=expected_criteria,
            reference_output=reference_output,
        )

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def multi_turn_consistency(
        self,
        responses: List[str],
        contradiction_threshold: float = 0.3,
    ) -> MetricResult:
        """Multi-turn conversation consistency.

        Citations:
            - Zheng et al. (2023). MT-Bench. NeurIPS 2023.
        """
        from SSEM.consistency import MultiTurnConsistencyMetric

        m = MultiTurnConsistencyMetric(
            engine=self.engine,
            contradiction_threshold=contradiction_threshold,
        )
        return m.evaluate(responses)

    def selfcheck(
        self,
        main_response: str,
        sampled_responses: List[str],
        threshold: float = 0.5,
    ) -> MetricResult:
        """SelfCheck consistency — hallucination detection via sampling.

        Citations:
            - Manakul et al. (2023). SelfCheckGPT. EMNLP 2023.
        """
        from SSEM.consistency import SelfCheckConsistencyMetric

        m = SelfCheckConsistencyMetric(engine=self.engine, threshold=threshold)
        return m.evaluate(main_response, sampled_responses)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def toxicity(
        self,
        texts: List[str],
        model_name: str = "unitary/toxic-bert",
    ) -> MetricResult:
        """Toxicity detection using a classifier.

        Citations:
            - Gehman et al. (2020). RealToxicityPrompts. EMNLP 2020.
        """
        from SSEM.safety import ToxicityMetric

        m = ToxicityMetric(model_name=model_name)
        return m.evaluate(texts)

    # ------------------------------------------------------------------
    # Code Evaluation
    # ------------------------------------------------------------------

    def code_correctness(
        self,
        code_samples: List[str],
        test_code: str,
        timeout: int = 10,
        k_values: Optional[List[int]] = None,
    ) -> MetricResult:
        """Code correctness via execution-based Pass@k.

        Citations:
            - Chen et al. (2021). Codex / HumanEval.
        """
        from SSEM.code_eval import CodeCorrectnessMetric

        m = CodeCorrectnessMetric(timeout=timeout, k_values=k_values)
        return m.evaluate(code_samples, test_code)

    # ------------------------------------------------------------------
    # Combined Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        output_sentences: Optional[List[str]] = None,
        reference_sentences: Optional[List[str]] = None,
        source_context: Optional[str] = None,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        reasoning_steps: Optional[List[str]] = None,
        reasoning_goal: Optional[str] = None,
        predicted_tool_calls: Optional[List[Dict[str, Any]]] = None,
        expected_tool_calls: Optional[List[Dict[str, Any]]] = None,
        agent_output: Optional[str] = None,
        expected_criteria: Optional[List[str]] = None,
        conversation_turns: Optional[List[str]] = None,
        texts_for_toxicity: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """Run all applicable metrics based on provided inputs.

        Only runs metrics for which the required inputs are provided.
        Returns a comprehensive EvaluationReport.

        Example:
            report = evaluator.evaluate_all(
                output_sentences=["The cat sat on the mat."],
                reference_sentences=["A cat was sitting on a mat."],
                source_context="A cat was observed sitting on a mat in the room.",
            )
            print(report.summary())
            print(report.explain())  # Full transparency
        """
        report = EvaluationReport()

        # Semantic similarity
        if output_sentences and reference_sentences:
            report.add(self.semantic_similarity(output_sentences, reference_sentences))
            if len(output_sentences) == len(reference_sentences):
                report.add(self.bertscore(output_sentences, reference_sentences))

        # Faithfulness & hallucination
        if output_sentences and source_context:
            combined_output = " ".join(output_sentences)
            report.add(self.faithfulness(combined_output, source_context))
            report.add(self.hallucination(combined_output, source_context))

        # Answer relevancy
        if questions and answers:
            report.add(self.answer_relevancy(questions, answers))

        # Agentic
        if reasoning_steps:
            report.add(self.reasoning_coherence(reasoning_steps, goal=reasoning_goal))

        if predicted_tool_calls is not None and expected_tool_calls is not None:
            report.add(self.tool_accuracy(predicted_tool_calls, expected_tool_calls))

        if agent_output and expected_criteria:
            report.add(self.task_completion(agent_output, expected_criteria=expected_criteria))

        # Consistency
        if conversation_turns:
            report.add(self.multi_turn_consistency(conversation_turns))

        # Safety
        if texts_for_toxicity:
            report.add(self.toxicity(texts_for_toxicity))

        return report

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_metrics(self) -> List[str]:
        """List all available metrics."""
        return [
            "semantic_similarity",
            "bertscore",
            "faithfulness",
            "hallucination",
            "answer_relevancy",
            "reasoning_coherence",
            "tool_accuracy",
            "task_completion",
            "multi_turn_consistency",
            "selfcheck",
            "toxicity",
            "code_correctness",
        ]

    def list_citations(self) -> Dict[str, str]:
        """List all research citations used in SSEM."""
        return {key: str(citation) for key, citation in CITATIONS.items()}

    def __repr__(self) -> str:
        return f"SSEM(model={self.engine.model_name!r}, device={self.engine.device!r})"
