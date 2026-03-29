"""
Answer Relevancy Metric
=======================

Measures how well an answer addresses the given question, using
embedding similarity between the question and the answer.

Research Basis:
    Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024).
    RAGAS: Automated Evaluation of Retrieval Augmented Generation.
    EACL 2024. https://arxiv.org/abs/2309.15217

How Scoring Works:
    Unlike RAG-specific implementations that require an LLM to generate
    synthetic questions from the answer, SSEM uses a lightweight
    embedding-based approach:

    1. Encode the question and the answer into dense vectors.
    2. Compute cosine similarity between the two embeddings.
    3. Higher similarity means the answer is semantically on-topic.

    This avoids LLM-as-judge costs while preserving the core intuition:
    a relevant answer should be semantically close to its question.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from SSEM.core import (
    BaseMetric,
    Citation,
    CITATIONS,
    EmbeddingEngine,
    MetricResult,
)


class AnswerRelevancyMetric(BaseMetric):
    """Embedding-based answer relevancy scoring.

    Research Basis:
        - Es et al. (2024). RAGAS. EACL 2024. (answer relevancy concept)
        - Manning et al. (2008). Introduction to Information Retrieval. (cosine similarity)

    How Scoring Works:
        1. Encode the question(s) into dense sentence embeddings.
        2. Encode the answer(s) into dense sentence embeddings.
        3. For each (question, answer) pair, compute cosine similarity.
        4. Average across all pairs.

    Score Range: [0.0, 1.0] where 1.0 = answer perfectly addresses the question.

    Transparency:
        - Per-pair similarity scores are included in the result.
        - The model used for encoding is explicitly stated.
        - The method section explains the exact computation steps.

    Parameters:
        engine: Pre-initialized EmbeddingEngine.
    """

    def __init__(self, engine: EmbeddingEngine) -> None:
        self.engine = engine

    @property
    def metric_name(self) -> str:
        return "Answer Relevancy"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["answer_relevancy"], CITATIONS["cosine_similarity"]]

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
    ) -> MetricResult:
        """Evaluate how relevant each answer is to its question.

        Args:
            questions: List of questions.
            answers:   List of answers (same length as questions).

        Returns:
            MetricResult with per-pair relevancy scores.
        """
        if len(questions) != len(answers):
            raise ValueError(
                f"Lists must have equal length. "
                f"Got {len(questions)} questions and {len(answers)} answers."
            )

        start = time.perf_counter()

        q_emb = self.engine.encode(questions)
        a_emb = self.engine.encode(answers)

        # Per-pair cosine similarity
        pair_scores = []
        for i in range(len(questions)):
            sim = cosine_similarity(
                q_emb[i : i + 1], a_emb[i : i + 1]
            )[0, 0]
            # Clamp to [0, 1] — negative cosine means anti-correlated
            pair_scores.append(float(max(0.0, sim)))

        mean_score = float(np.mean(pair_scores))
        elapsed = time.perf_counter() - start

        method = (
            f"1. Encoded {len(questions)} questions and {len(answers)} answers "
            f"using '{self.engine.model_name}' (mean-pooled embeddings).\n"
            f"2. Computed cosine similarity between each (question, answer) pair.\n"
            f"3. Clamped negative similarities to 0 (anti-correlated = irrelevant).\n"
            f"4. Averaged across {len(questions)} pairs: mean={mean_score:.4f}."
        )

        return MetricResult(
            score=mean_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(mean_score),
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "pair_scores": pair_scores,
                "per_pair": [
                    {"question": q, "answer": a, "relevancy": s}
                    for q, a, s in zip(questions, answers, pair_scores)
                ],
            },
            elapsed_sec=elapsed,
        )

    def _interpret(self, score: float) -> str:
        if score >= 0.85:
            return "Highly relevant — answers directly address the questions."
        elif score >= 0.65:
            return "Moderately relevant — answers are on-topic but may miss specifics."
        elif score >= 0.4:
            return "Weakly relevant — answers partially address the questions."
        else:
            return "Irrelevant — answers do not address the questions."
