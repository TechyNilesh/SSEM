"""
Faithfulness & Hallucination Detection Metrics
===============================================

Two NLI-based metrics that evaluate whether generated text is grounded
in source material — critical for any LLM or agent deployment.

1. **FaithfulnessMetric** — Decomposes output into atomic claims and checks
   each for entailment against the source context.
2. **HallucinationMetric** — Measures the fraction of output claims that are
   NOT supported by the source (inverse of faithfulness).

Both metrics use a Natural Language Inference (NLI) model to classify
each claim as ENTAILMENT, NEUTRAL, or CONTRADICTION.

Research Basis:
    - Kryscinski et al. (2020). Evaluating the Factual Consistency of
      Abstractive Text Summarization. EMNLP 2020.
    - Williams, Nangia & Bowman (2018). MultiNLI. NAACL 2018.

Scoring Transparency:
    The result includes every extracted claim, its NLI label, and its
    individual score, so users can trace exactly why a text was scored
    as faithful or hallucinated.
"""

from __future__ import annotations

import re
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


def _split_into_claims(text: str) -> List[str]:
    """Split text into atomic claims (sentences).

    This is a lightweight sentence splitter. For production use, consider
    spaCy or a dedicated claim extraction model.

    Method:
        Split on sentence-ending punctuation (.!?) followed by whitespace
        or end-of-string. Filter out empty strings.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class FaithfulnessMetric(BaseMetric):
    """Measures whether generated output is factually grounded in the source context.

    Research Basis:
        Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020).
        Evaluating the Factual Consistency of Abstractive Text Summarization.
        EMNLP 2020. https://arxiv.org/abs/1910.12840

    How Scoring Works:
        1. **Claim Extraction**: The output text is split into individual
           atomic claims (sentences).
        2. **Entailment Checking**: Each claim is compared against the full
           source context using one of two strategies:

           a) **NLI-based** (if an NLI model is provided): Each claim is
              classified as ENTAILMENT / NEUTRAL / CONTRADICTION against
              the source. Score = fraction of claims entailed.

           b) **Embedding-based** (default, no extra model needed): Each claim
              is embedded alongside the source. The cosine similarity between
              claim and source embeddings serves as a soft entailment score.
              Claims with similarity >= threshold are considered faithful.

        3. **Aggregation**: Faithfulness = (# faithful claims) / (# total claims).

    Score Range: [0.0, 1.0] where 1.0 = all claims are faithful to the source.

    Parameters:
        engine:     EmbeddingEngine for encoding claims and context.
        nli_model:  Optional HuggingFace NLI pipeline for hard entailment.
                    If None, falls back to embedding-based soft scoring.
        threshold:  Cosine similarity threshold for embedding-based mode.
                    Default 0.5.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        nli_model: Optional[Any] = None,
        threshold: float = 0.5,
    ) -> None:
        self.engine = engine
        self.nli_model = nli_model
        self.threshold = threshold

    @property
    def metric_name(self) -> str:
        mode = "NLI" if self.nli_model else "Embedding"
        return f"Faithfulness ({mode}-based)"

    @property
    def citations(self) -> List[Citation]:
        cites = [CITATIONS["nli_faithfulness"]]
        if self.nli_model:
            cites.append(CITATIONS["nli_models"])
        else:
            cites.append(CITATIONS["cosine_similarity"])
        return cites

    def evaluate(
        self,
        output_text: str,
        source_context: str,
    ) -> MetricResult:
        """Evaluate faithfulness of output against source context.

        Args:
            output_text:    The generated text to evaluate.
            source_context: The source/reference context (ground truth).

        Returns:
            MetricResult with per-claim breakdown.
        """
        start = time.perf_counter()

        claims = _split_into_claims(output_text)
        if not claims:
            elapsed = time.perf_counter() - start
            return MetricResult(
                score=1.0,
                metric_name=self.metric_name,
                score_range=(0.0, 1.0),
                interpretation="No claims found in output — trivially faithful.",
                method="Output text contained no extractable claims.",
                model_used=self.engine.model_name,
                citations=self.citations,
                details={"claims": [], "claim_scores": []},
                elapsed_sec=elapsed,
            )

        if self.nli_model:
            claim_results = self._evaluate_nli(claims, source_context)
            mode_description = (
                "Each claim was classified against the source using an NLI model.\n"
                "   Labels: ENTAILMENT (faithful), CONTRADICTION (hallucinated), "
                "NEUTRAL (unsupported).\n"
                "   A claim is considered faithful if the NLI label is ENTAILMENT."
            )
        else:
            claim_results = self._evaluate_embedding(claims, source_context)
            mode_description = (
                f"Each claim was embedded alongside the source context using "
                f"'{self.engine.model_name}'.\n"
                f"   Cosine similarity between claim and source embeddings was computed.\n"
                f"   A claim is considered faithful if similarity >= {self.threshold}."
            )

        faithful_count = sum(1 for c in claim_results if c["faithful"])
        score = faithful_count / len(claims)
        elapsed = time.perf_counter() - start

        method = (
            f"1. Extracted {len(claims)} claims from the output text.\n"
            f"2. {mode_description}\n"
            f"3. Faithfulness = {faithful_count}/{len(claims)} faithful claims = {score:.4f}."
        )

        return MetricResult(
            score=score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(score),
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "claims": claims,
                "claim_results": claim_results,
                "faithful_count": faithful_count,
                "total_claims": len(claims),
                "threshold": self.threshold if not self.nli_model else None,
            },
            elapsed_sec=elapsed,
        )

    def _evaluate_nli(
        self, claims: List[str], source: str
    ) -> List[Dict[str, Any]]:
        """Evaluate claims using an NLI model."""
        results = []
        for claim in claims:
            nli_input = f"{source} [SEP] {claim}"
            prediction = self.nli_model(nli_input)
            # HuggingFace pipeline returns [{"label": ..., "score": ...}]
            if isinstance(prediction, list):
                prediction = prediction[0]
            label = prediction.get("label", "").upper()
            nli_score = prediction.get("score", 0.0)
            faithful = "ENTAIL" in label
            results.append({
                "claim": claim,
                "nli_label": label,
                "nli_confidence": nli_score,
                "faithful": faithful,
            })
        return results

    def _evaluate_embedding(
        self, claims: List[str], source: str
    ) -> List[Dict[str, Any]]:
        """Evaluate claims using embedding cosine similarity."""
        source_emb = self.engine.encode([source])
        claim_embs = self.engine.encode(claims)
        similarities = cosine_similarity(claim_embs, source_emb).flatten()

        results = []
        for claim, sim in zip(claims, similarities):
            results.append({
                "claim": claim,
                "similarity_to_source": float(sim),
                "threshold": self.threshold,
                "faithful": float(sim) >= self.threshold,
            })
        return results

    def _interpret(self, score: float) -> str:
        if score >= 0.95:
            return "Highly faithful — nearly all claims are grounded in the source."
        elif score >= 0.8:
            return "Mostly faithful — most claims are supported, minor unsupported content."
        elif score >= 0.5:
            return "Partially faithful — significant portion of claims lack source support."
        else:
            return "Low faithfulness — majority of claims are not grounded in the source."


class HallucinationMetric(BaseMetric):
    """Measures the fraction of generated claims NOT supported by source context.

    This is the complement of Faithfulness: Hallucination = 1 - Faithfulness.

    Research Basis:
        - Kryscinski et al. (2020). EMNLP 2020. (claim-level verification)
        - Manakul et al. (2023). SelfCheckGPT. EMNLP 2023. (hallucination framing)

    How Scoring Works:
        1. Identical to FaithfulnessMetric (claim extraction + entailment check).
        2. Hallucination Score = 1 - Faithfulness Score.
        3. Each unsupported claim is flagged individually.

    Score Range: [0.0, 1.0] where 0.0 = no hallucination, 1.0 = fully hallucinated.

    Parameters:
        engine:     EmbeddingEngine for encoding claims and context.
        nli_model:  Optional HuggingFace NLI pipeline.
        threshold:  Cosine similarity threshold for embedding-based mode.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        nli_model: Optional[Any] = None,
        threshold: float = 0.5,
    ) -> None:
        self._faithfulness = FaithfulnessMetric(engine, nli_model, threshold)

    @property
    def metric_name(self) -> str:
        return "Hallucination Score"

    @property
    def citations(self) -> List[Citation]:
        return [
            CITATIONS["nli_faithfulness"],
            CITATIONS["selfcheckgpt"],
        ]

    def evaluate(
        self,
        output_text: str,
        source_context: str,
    ) -> MetricResult:
        """Evaluate hallucination level.

        Args:
            output_text:    The generated text to evaluate.
            source_context: The source/reference context.

        Returns:
            MetricResult where score = fraction of hallucinated claims.
        """
        faith_result = self._faithfulness.evaluate(output_text, source_context)

        hallucination_score = 1.0 - faith_result.score
        details = faith_result.details or {}

        # Flag hallucinated claims
        hallucinated_claims = []
        for cr in details.get("claim_results", []):
            if not cr["faithful"]:
                hallucinated_claims.append(cr["claim"])

        method = (
            f"1. Computed Faithfulness score using {self._faithfulness.metric_name}.\n"
            f"2. Hallucination = 1 - Faithfulness = 1 - {faith_result.score:.4f} "
            f"= {hallucination_score:.4f}.\n"
            f"3. Flagged {len(hallucinated_claims)}/{details.get('total_claims', 0)} "
            f"claims as hallucinated (not grounded in source)."
        )

        return MetricResult(
            score=hallucination_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(hallucination_score),
            method=method,
            model_used=self._faithfulness.engine.model_name,
            citations=self.citations,
            details={
                "faithfulness_score": faith_result.score,
                "hallucinated_claims": hallucinated_claims,
                "total_claims": details.get("total_claims", 0),
                "claim_results": details.get("claim_results", []),
            },
            elapsed_sec=faith_result.elapsed_sec,
        )

    def _interpret(self, score: float) -> str:
        if score <= 0.05:
            return "Minimal hallucination — output is well-grounded in the source."
        elif score <= 0.2:
            return "Low hallucination — a few unsupported claims detected."
        elif score <= 0.5:
            return "Moderate hallucination — notable portion of claims lack source support."
        else:
            return "High hallucination — majority of output is not grounded in the source."
