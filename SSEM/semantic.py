"""
Semantic Similarity & BERTScore Metrics
=======================================

Two complementary metrics for measuring text generation quality:

1. **SemanticSimilarity** — Sentence-level, token-level, or LSI-based similarity
   between generated and reference text.  This is the original SSEM v1 metric,
   now refactored with transparency and citation support.

2. **BERTScore** — Token-level precision, recall, and F1 using contextual
   embeddings, as proposed by Zhang et al. (2020, ICLR).

Scoring Transparency:
    Both metrics return ``MetricResult`` objects that fully document:
    - The exact computation steps
    - Which model produced the embeddings
    - The score range and interpretation
    - The research papers behind the approach
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import correlation

from SSEM.core import (
    BaseMetric,
    Citation,
    CITATIONS,
    EmbeddingEngine,
    MetricResult,
)


class SemanticSimilarityMetric(BaseMetric):
    """Sentence-level semantic similarity between generated and reference text.

    Computes the pairwise similarity of sentence embeddings using one of
    several distance metrics.

    Research Basis:
        - Beken Fikri, F., Oflazer, K., & Yanıkoğlu, B. (2021). Semantic
          Similarity Based Evaluation for Abstractive News Summarization.
          GEM 2021.
        - Manning et al. (2008). Introduction to Information Retrieval.
          (cosine similarity foundation)

    How Scoring Works:
        1. Each sentence is encoded into a dense vector via a transformer model.
        2. Pairwise similarity is computed between all output-reference pairs.
        3. The diagonal (matched pairs) or full matrix is aggregated.

    Parameters:
        engine:  Pre-initialized ``EmbeddingEngine``.
        metric:  One of "cosine", "euclidean", "pearson".
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        metric: str = "cosine",
    ) -> None:
        self.engine = engine
        self.metric = metric

    @property
    def metric_name(self) -> str:
        return f"Semantic Similarity ({self.metric})"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["semantic_similarity"], CITATIONS["cosine_similarity"]]

    def _compute_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> np.ndarray:
        if self.metric == "cosine":
            return cosine_similarity(emb1, emb2)
        elif self.metric == "euclidean":
            # Convert distance to similarity: 1 / (1 + d)
            dist = euclidean_distances(emb1, emb2)
            return 1.0 / (1.0 + dist)
        elif self.metric == "pearson":
            n = emb1.shape[0]
            m = emb2.shape[0]
            sim = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    # scipy correlation returns distance; similarity = 1 - distance
                    sim[i, j] = 1.0 - correlation(emb1[i], emb2[j])
            return sim
        else:
            raise ValueError(
                f"Unknown metric '{self.metric}'. Choose from: cosine, euclidean, pearson."
            )

    def evaluate(
        self,
        output_sentences: List[str],
        reference_sentences: List[str],
        pooling: str = "mean",
    ) -> MetricResult:
        """Evaluate semantic similarity.

        Args:
            output_sentences:    Generated texts.
            reference_sentences: Reference texts.
            pooling:             "mean" for mean-pooled embeddings,
                                 "cls" for CLS-token embeddings (legacy).

        Returns:
            MetricResult with full transparency.
        """
        import time

        start = time.perf_counter()

        if pooling == "cls":
            out_emb = self.engine.encode_cls(output_sentences)
            ref_emb = self.engine.encode_cls(reference_sentences)
        else:
            out_emb = self.engine.encode(output_sentences)
            ref_emb = self.engine.encode(reference_sentences)

        sim_matrix = self._compute_similarity(out_emb, ref_emb)

        # Diagonal = matched pair scores (output[i] vs reference[i])
        n_pairs = min(len(output_sentences), len(reference_sentences))
        paired_scores = np.array([sim_matrix[i, i] for i in range(n_pairs)])
        mean_score = float(np.mean(paired_scores))
        std_score = float(np.std(paired_scores))

        elapsed = time.perf_counter() - start

        # Score range depends on metric
        if self.metric == "cosine":
            score_range = (-1.0, 1.0)
        elif self.metric == "euclidean":
            score_range = (0.0, 1.0)
        else:
            score_range = (-1.0, 1.0)

        method = (
            f"1. Encoded {len(output_sentences)} output and {len(reference_sentences)} "
            f"reference sentences using '{self.engine.model_name}' ({pooling}-pooling).\n"
            f"2. Computed pairwise {self.metric} similarity matrix "
            f"(shape {sim_matrix.shape[0]}x{sim_matrix.shape[1]}).\n"
            f"3. Extracted diagonal (matched pair) scores.\n"
            f"4. Aggregated: mean={mean_score:.4f}, std={std_score:.4f}."
        )

        interpretation = self._interpret(mean_score)

        return MetricResult(
            score=mean_score,
            metric_name=self.metric_name,
            score_range=score_range,
            interpretation=interpretation,
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "paired_scores": paired_scores.tolist(),
                "std": std_score,
                "similarity_matrix": sim_matrix.tolist(),
                "pooling": pooling,
                "metric": self.metric,
            },
            elapsed_sec=elapsed,
        )

    def _interpret(self, score: float) -> str:
        if self.metric == "cosine":
            if score >= 0.9:
                return "Very high similarity — outputs closely match references."
            elif score >= 0.7:
                return "High similarity — outputs are semantically close to references."
            elif score >= 0.5:
                return "Moderate similarity — partial semantic overlap."
            elif score >= 0.3:
                return "Low similarity — limited semantic overlap."
            else:
                return "Very low similarity — outputs diverge significantly from references."
        else:
            if score >= 0.8:
                return "Very high similarity."
            elif score >= 0.5:
                return "Moderate similarity."
            else:
                return "Low similarity."


class BERTScoreMetric(BaseMetric):
    """BERTScore: Token-level precision, recall, and F1 using contextual embeddings.

    Research Basis:
        Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020).
        BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
        https://arxiv.org/abs/1904.09675

    How Scoring Works:
        1. Encode each sentence into per-token contextual embeddings.
        2. For each output-reference pair, compute a cosine similarity matrix
           between all output tokens and all reference tokens.
        3. **Precision**: For each output token, take the max similarity to any
           reference token, then average across output tokens.
           → "How much of the output is supported by the reference?"
        4. **Recall**: For each reference token, take the max similarity to any
           output token, then average across reference tokens.
           → "How much of the reference is captured by the output?"
        5. **F1**: Harmonic mean of precision and recall.
        6. Average across all sentence pairs.

    Score Range: [0, 1] where 1 = perfect token-level alignment.
    """

    def __init__(self, engine: EmbeddingEngine) -> None:
        self.engine = engine

    @property
    def metric_name(self) -> str:
        return "BERTScore"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["bertscore"]]

    def _compute_bertscore_pair(
        self, output_tokens: np.ndarray, reference_tokens: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute P, R, F1 for a single sentence pair.

        Args:
            output_tokens:    shape (n_out, hidden_dim)
            reference_tokens: shape (n_ref, hidden_dim)

        Returns:
            (precision, recall, f1)
        """
        # Cosine similarity matrix: (n_out, n_ref)
        sim = cosine_similarity(output_tokens, reference_tokens)

        # Precision: for each output token, best match in reference
        precision = float(np.mean(np.max(sim, axis=1)))

        # Recall: for each reference token, best match in output
        recall = float(np.mean(np.max(sim, axis=0)))

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    def evaluate(
        self,
        output_sentences: List[str],
        reference_sentences: List[str],
    ) -> MetricResult:
        """Evaluate BERTScore for paired output-reference sentences.

        Args:
            output_sentences:    Generated texts.
            reference_sentences: Reference texts (must be same length).

        Returns:
            MetricResult with precision, recall, F1, and per-pair details.
        """
        import time

        if len(output_sentences) != len(reference_sentences):
            raise ValueError(
                f"Sentence lists must have equal length. "
                f"Got {len(output_sentences)} outputs and {len(reference_sentences)} references."
            )

        start = time.perf_counter()

        out_tokens = self.engine.encode_tokens(output_sentences)
        ref_tokens = self.engine.encode_tokens(reference_sentences)

        precisions, recalls, f1s = [], [], []
        for o_tok, r_tok in zip(out_tokens, ref_tokens):
            p, r, f = self._compute_bertscore_pair(o_tok, r_tok)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        mean_p = float(np.mean(precisions))
        mean_r = float(np.mean(recalls))
        mean_f1 = float(np.mean(f1s))
        elapsed = time.perf_counter() - start

        method = (
            f"1. Encoded {len(output_sentences)} sentence pairs into per-token "
            f"contextual embeddings using '{self.engine.model_name}'.\n"
            f"2. For each pair, built a cosine similarity matrix between "
            f"output tokens and reference tokens.\n"
            f"3. Precision = mean of row-wise max similarities "
            f"(each output token's best reference match).\n"
            f"4. Recall = mean of column-wise max similarities "
            f"(each reference token's best output match).\n"
            f"5. F1 = harmonic mean of precision and recall.\n"
            f"6. Averaged across {len(output_sentences)} pairs: "
            f"P={mean_p:.4f}, R={mean_r:.4f}, F1={mean_f1:.4f}."
        )

        interpretation = self._interpret(mean_f1)

        return MetricResult(
            score=mean_f1,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=interpretation,
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "precision": mean_p,
                "recall": mean_r,
                "f1": mean_f1,
                "per_pair": [
                    {"precision": p, "recall": r, "f1": f}
                    for p, r, f in zip(precisions, recalls, f1s)
                ],
            },
            elapsed_sec=elapsed,
        )

    def _interpret(self, f1: float) -> str:
        if f1 >= 0.95:
            return "Near-perfect token-level alignment between output and reference."
        elif f1 >= 0.85:
            return "Strong token-level overlap — output captures most reference content."
        elif f1 >= 0.70:
            return "Moderate overlap — some reference content missing or extra content present."
        elif f1 >= 0.50:
            return "Weak overlap — significant divergence between output and reference."
        else:
            return "Poor alignment — output and reference share little semantic content."
