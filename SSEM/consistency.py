"""
Multi-Turn Consistency & Self-Check Metrics
============================================

Metrics for evaluating consistency in multi-turn conversations and
detecting potential hallucinations via self-consistency checking.

1. **MultiTurnConsistencyMetric** — Measures whether a model contradicts
   itself across conversation turns.
2. **SelfCheckConsistencyMetric** — Samples multiple responses and flags
   inconsistent claims as likely hallucinations.

Research Basis:
    - Zheng et al. (2023). MT-Bench. NeurIPS 2023.
    - Manakul et al. (2023). SelfCheckGPT. EMNLP 2023.
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


class MultiTurnConsistencyMetric(BaseMetric):
    """Measures self-consistency across multiple conversation turns.

    Research Basis:
        Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench
        and Chatbot Arena. NeurIPS 2023.
        https://arxiv.org/abs/2306.05685

    How Scoring Works:
        Given a list of model responses across conversation turns:

        1. Encode all responses into dense sentence embeddings.
        2. Compute an all-pairs cosine similarity matrix.
        3. **Sequential Consistency**: Average similarity between consecutive
           turns. Measures topic/stance drift.
        4. **Global Consistency**: Average similarity across ALL turn pairs.
           Measures overall coherence of the conversation.
        5. **Contradiction Detection**: Any pair with similarity below a
           threshold is flagged as a potential self-contradiction.
        6. Final Score = weighted combination of sequential and global.

    Score Range: [0.0, 1.0] where 1.0 = perfectly consistent across all turns.

    Parameters:
        engine:              EmbeddingEngine for encoding turns.
        contradiction_threshold: Similarity below this flags a contradiction.
        sequential_weight:   Weight for sequential consistency in final score.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        contradiction_threshold: float = 0.3,
        sequential_weight: float = 0.6,
    ) -> None:
        self.engine = engine
        self.contradiction_threshold = contradiction_threshold
        self.sequential_weight = sequential_weight

    @property
    def metric_name(self) -> str:
        return "Multi-Turn Consistency"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["multi_turn"]]

    def evaluate(
        self,
        responses: List[str],
    ) -> MetricResult:
        """Evaluate consistency across conversation turns.

        Args:
            responses: Ordered list of model responses in a conversation.

        Returns:
            MetricResult with per-turn and pairwise consistency details.
        """
        if len(responses) < 2:
            return MetricResult(
                score=1.0,
                metric_name=self.metric_name,
                score_range=(0.0, 1.0),
                interpretation="Single response — consistency is trivially 1.0.",
                method="Only one response provided; no turns to compare.",
                model_used=self.engine.model_name,
                citations=self.citations,
            )

        start = time.perf_counter()

        embs = self.engine.encode(responses)
        sim_matrix = cosine_similarity(embs, embs)

        # Sequential consistency
        sequential_sims = []
        for i in range(len(responses) - 1):
            sequential_sims.append(float(sim_matrix[i, i + 1]))
        seq_consistency = float(np.mean(sequential_sims))

        # Global consistency (upper triangle, excluding diagonal)
        n = len(responses)
        global_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                global_sims.append(float(sim_matrix[i, j]))
        global_consistency = float(np.mean(global_sims))

        # Contradiction detection
        contradictions = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] < self.contradiction_threshold:
                    contradictions.append({
                        "turn_pair": (i, j),
                        "similarity": float(sim_matrix[i, j]),
                        "turn_a": responses[i][:100] + ("..." if len(responses[i]) > 100 else ""),
                        "turn_b": responses[j][:100] + ("..." if len(responses[j]) > 100 else ""),
                    })

        # Final score
        final_score = (
            self.sequential_weight * seq_consistency
            + (1 - self.sequential_weight) * global_consistency
        )
        # Penalty for contradictions
        penalty = len(contradictions) * 0.05
        final_score = max(0.0, final_score - penalty)

        elapsed = time.perf_counter() - start

        method = (
            f"1. Encoded {len(responses)} conversation turns using "
            f"'{self.engine.model_name}'.\n"
            f"2. Sequential Consistency: mean cosine similarity between "
            f"consecutive turns = {seq_consistency:.4f}.\n"
            f"3. Global Consistency: mean cosine similarity across all "
            f"{len(global_sims)} turn pairs = {global_consistency:.4f}.\n"
            f"4. Contradiction Detection: {len(contradictions)} turn pairs "
            f"below threshold {self.contradiction_threshold}.\n"
            f"5. Final = {self.sequential_weight:.1f}*sequential + "
            f"{1-self.sequential_weight:.1f}*global - "
            f"{len(contradictions)}*0.05 penalty = {final_score:.4f}."
        )

        return MetricResult(
            score=final_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(final_score, len(contradictions)),
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "sequential_consistency": seq_consistency,
                "global_consistency": global_consistency,
                "sequential_similarities": sequential_sims,
                "contradictions": contradictions,
                "similarity_matrix": sim_matrix.tolist(),
            },
            elapsed_sec=elapsed,
        )

    def _interpret(self, score: float, n_contradictions: int) -> str:
        parts = []
        if score >= 0.85:
            parts.append("Highly consistent across turns.")
        elif score >= 0.65:
            parts.append("Moderately consistent — some drift between turns.")
        elif score >= 0.4:
            parts.append("Low consistency — significant stance or topic changes.")
        else:
            parts.append("Very inconsistent — the model contradicts itself.")
        if n_contradictions > 0:
            parts.append(f" {n_contradictions} potential self-contradiction(s).")
        return "".join(parts)


class SelfCheckConsistencyMetric(BaseMetric):
    """Zero-resource hallucination detection via sampling consistency.

    Research Basis:
        Manakul, P., Liusie, A., & Gales, M. J. F. (2023).
        SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection.
        EMNLP 2023. https://arxiv.org/abs/2303.08896

    How Scoring Works:
        Given a "main" response and multiple sampled responses to the
        same prompt:

        1. Split the main response into claims (sentences).
        2. For each claim, compute its cosine similarity to each
           sampled response.
        3. A claim is considered **consistent** if its average similarity
           across all samples exceeds the threshold.
        4. **Inconsistency Score** = fraction of inconsistent claims.
           High inconsistency suggests hallucination.

    Intuition:
        Factual statements tend to be consistent across samples (the model
        "knows" them). Hallucinated content varies between samples because
        it was fabricated rather than retrieved from learned knowledge.

    Score Range: [0.0, 1.0] where 0.0 = fully consistent (no hallucination
    signal), 1.0 = fully inconsistent (high hallucination risk).

    Parameters:
        engine:     EmbeddingEngine for encoding claims and samples.
        threshold:  Consistency threshold. Default 0.5.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        threshold: float = 0.5,
    ) -> None:
        self.engine = engine
        self.threshold = threshold

    @property
    def metric_name(self) -> str:
        return "SelfCheck Consistency"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["selfcheckgpt"]]

    def evaluate(
        self,
        main_response: str,
        sampled_responses: List[str],
    ) -> MetricResult:
        """Evaluate self-consistency for hallucination detection.

        Args:
            main_response:     The primary response to evaluate.
            sampled_responses: Multiple alternative responses to the same prompt.

        Returns:
            MetricResult where score = inconsistency rate (0 = consistent, 1 = inconsistent).
        """
        import re

        start = time.perf_counter()

        # Split main response into claims
        claims = [s.strip() for s in re.split(r'(?<=[.!?])\s+', main_response.strip()) if s.strip()]

        if not claims:
            elapsed = time.perf_counter() - start
            return MetricResult(
                score=0.0,
                metric_name=self.metric_name,
                score_range=(0.0, 1.0),
                interpretation="No claims extracted — no inconsistency detected.",
                method="Main response contained no extractable claims.",
                model_used=self.engine.model_name,
                citations=self.citations,
                elapsed_sec=elapsed,
            )

        if not sampled_responses:
            elapsed = time.perf_counter() - start
            return MetricResult(
                score=0.0,
                metric_name=self.metric_name,
                score_range=(0.0, 1.0),
                interpretation="No sampled responses provided for comparison.",
                method="Cannot assess consistency without sampled responses.",
                model_used=self.engine.model_name,
                citations=self.citations,
                elapsed_sec=elapsed,
            )

        # Encode claims and samples
        claim_embs = self.engine.encode(claims)
        sample_embs = self.engine.encode(sampled_responses)

        # For each claim, compute average similarity to all samples
        claim_results = []
        inconsistent_count = 0
        for i, (claim, claim_emb) in enumerate(zip(claims, claim_embs)):
            sims = cosine_similarity(claim_emb.reshape(1, -1), sample_embs).flatten()
            avg_sim = float(np.mean(sims))
            consistent = avg_sim >= self.threshold
            if not consistent:
                inconsistent_count += 1
            claim_results.append({
                "claim": claim,
                "avg_similarity_to_samples": avg_sim,
                "per_sample_similarities": sims.tolist(),
                "consistent": consistent,
            })

        inconsistency_score = inconsistent_count / len(claims)
        elapsed = time.perf_counter() - start

        method = (
            f"1. Extracted {len(claims)} claims from the main response.\n"
            f"2. Encoded claims and {len(sampled_responses)} sampled responses "
            f"using '{self.engine.model_name}'.\n"
            f"3. For each claim, computed average cosine similarity to all "
            f"sampled responses.\n"
            f"4. A claim is inconsistent if avg_similarity < {self.threshold}.\n"
            f"5. Inconsistency = {inconsistent_count}/{len(claims)} = "
            f"{inconsistency_score:.4f}.\n"
            f"\n"
            f"Interpretation: Claims that are factual tend to appear consistently "
            f"across multiple samples. Claims that vary between samples are likely "
            f"hallucinated (Manakul et al., 2023)."
        )

        return MetricResult(
            score=inconsistency_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(inconsistency_score),
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "claims": claims,
                "claim_results": claim_results,
                "inconsistent_count": inconsistent_count,
                "total_claims": len(claims),
                "n_samples": len(sampled_responses),
            },
            elapsed_sec=elapsed,
        )

    def _interpret(self, score: float) -> str:
        if score <= 0.05:
            return "Highly consistent — low hallucination risk."
        elif score <= 0.2:
            return "Mostly consistent — a few claims may be unreliable."
        elif score <= 0.5:
            return "Moderate inconsistency — notable hallucination risk."
        else:
            return "Highly inconsistent — strong hallucination signal."
